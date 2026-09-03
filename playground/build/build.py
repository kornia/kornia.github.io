"""Build the static assets behind kornia.org/playground.

Runs once, offline, from a checkout of kornia; nothing here runs on the deployed site.

1. ``specs.SPECS`` says how to call each demoable operator. For every spec the builder exports an
   ONNX graph with the live parameters as graph inputs (one graph per value of a discrete
   parameter), first with dynamic batch/height/width and, when the operator specialises its
   shapes, at the fixed demo size. Every graph passes ``onnx.checker`` and is executed once with
   onnxruntime against eager PyTorch before it is published.
2. When export fails the builder renders frames along the first parameter instead, so the page
   still has a working demo, and records why.
3. Every public name of the image packages (and every operator the export survey knows in the
   other packages) is listed in the catalog; names without a spec are greyed out with a reason.
4. Output: ``playground/registry.json`` (read by ``playground.js``), ``graphs/``, ``frames/``,
   ``images/`` and one static page per demoable operator under ``ops/<slug>/``.

Usage, from the repository root, with a kornia environment that has onnx, onnxscript, onnxruntime::

    python playground/build/build.py                       # everything
    python playground/build/build.py gaussian_blur2d rotate  # only these (matched on the id suffix)
    python playground/build/build.py --pages-only          # regenerate registry + pages from cached graphs
"""

from __future__ import annotations

import inspect
import io
import json
import shutil
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torch.export import Dim

import kornia as K

sys.path.insert(0, str(Path(__file__).resolve().parent))
from specs import HIDDEN, MODULE_FORMS, SPECS, UNSUPPORTED_REASONS, Live, Select, Spec, doc_url  # noqa: E402

warnings.filterwarnings("ignore")
torch.manual_seed(0)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # playground/
SIZE = 256
FRAME_STEPS = 16
ORT_WEB_VERSION = "1.29.0"  # pinned in the HTML too; keep them in sync
OPSET = 18
KORNIA_REPO = Path.home() / "git" / "kornia"  # for the export survey snapshot (optional)

# (id, file, label, selectable). Non-selectable samples are fixed auxiliary inputs (a guidance image).
SAMPLES = [
    ("parrots", "kodim23.png", "Parrots", True),
    ("motorcycles", "kodim05.png", "Motorcycles", True),
    ("lighthouse", "kodim19.png", "Lighthouse", True),
    ("barn", "kodim22.png", "Barn", False),
]

PACKAGES = [
    ("kornia.filters", "Filters"),
    ("kornia.color", "Color"),
    ("kornia.enhance", "Enhance"),
    ("kornia.morphology", "Morphology"),
    ("kornia.geometry.transform", "Geometry: transforms"),
    ("kornia.augmentation", "Augmentation"),
]
OTHER_PACKAGE_TITLES = {
    "kornia.feature": "Feature detection and matching",
    "kornia.geometry": "Geometry: other",
    "kornia.losses": "Losses",
    "kornia.metrics": "Metrics",
    "kornia.models": "Models",
    "kornia.contrib": "Contrib",
    "kornia.utils": "Utils",
    "kornia.image": "Image container",
    "kornia.sensors": "Sensors",
    "kornia.io": "I/O",
    "kornia.tracking": "Tracking",
    "kornia.nerf": "NeRF",
    "kornia.x": "Training utilities (x)",
}


def slug_of(op_id: str) -> str:
    return op_id.replace("kornia.", "").replace(".", "-")


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- images


def _to_tensor(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def _display(t: torch.Tensor, mode: str) -> Image.Image:
    t = t[0].detach().float()
    if t.dim() == 2:
        t = t[None]
    if t.shape[0] == 4:
        t = t[:3]
    if mode == "normalize":
        lo, hi = t.amin(), t.amax()
        t = (t - lo) / (hi - lo + 1e-8)
    t = t.clamp(0, 1)
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def prepare_images() -> list[dict[str, str]]:
    out_dir = ROOT / "images"
    out_dir.mkdir(exist_ok=True)
    entries = []
    for sid, filename, label, selectable in SAMPLES:
        im = Image.open(HERE / "samples" / filename).convert("RGB")
        side = min(im.size)
        left, top = (im.width - side) // 2, (im.height - side) // 2
        sq = im.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)
        sq.save(out_dir / f"{sid}.png", optimize=True)
        sq.resize((96, 96), Image.LANCZOS).save(out_dir / f"{sid}_thumb.jpg", quality=85)
        entries.append({"id": sid, "label": label, "file": f"images/{sid}.png", "thumb": f"images/{sid}_thumb.jpg", "selectable": selectable})
    return entries


# --------------------------------------------------------------------------- export


class _WrapperBase(nn.Module):
    """Adapts a spec's ``call`` to ``forward(x[, guide], <one tensor per live param>)`` with the selects baked in.

    ``torch.export`` matches ``dynamic_shapes`` against the *signature*, so the forward must take
    its tensors as named positional arguments rather than ``*args``; ``_Wrapper`` builds a subclass
    with a forward of the right arity.
    """

    def __init__(self, spec: Spec, selects: dict[str, Any]) -> None:
        super().__init__()
        self.spec = spec
        self.selects = selects
        self.live_names = [p.name for p in spec.params if isinstance(p, Live)]

    def _call(self, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        kwargs = dict(self.selects)
        rest = args[1:]
        if self.spec.guidance:
            kwargs["guide"] = rest[0]
            rest = rest[1:]
        kwargs.update(zip(self.live_names, rest))
        return self.spec.call(args[0], **kwargs)


_WRAPPERS: dict[tuple[bool, int], type] = {}


def _Wrapper(spec: Spec, selects: dict[str, Any]) -> _WrapperBase:  # noqa: N802 - factory used like a class
    n = sum(isinstance(p, Live) for p in spec.params)
    key = (bool(spec.guidance), n)
    if key not in _WRAPPERS:
        names = ["x"] + (["guide"] if spec.guidance else []) + [f"p{i}" for i in range(n)]
        ns: dict[str, Any] = {}
        exec(f"def forward(self, {', '.join(names)}):\n    return self._call(({', '.join(names)},))", ns)  # noqa: S102
        _WRAPPERS[key] = type(f"_Wrapper_{int(key[0])}_{n}", (_WrapperBase,), {"forward": ns["forward"]})
    return _WRAPPERS[key](spec, selects)


_GUIDES: dict[str, torch.Tensor] = {}


def _guide(spec: Spec) -> tuple[torch.Tensor, ...]:
    """The fixed guidance image of a spec as a (1, 3, SIZE, SIZE) tensor, or an empty tuple."""
    if not spec.guidance:
        return ()
    if spec.guidance not in _GUIDES:
        _GUIDES[spec.guidance] = _to_tensor(Image.open(ROOT / "images" / f"{spec.guidance}.png"))
    return (_GUIDES[spec.guidance],)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _output_info(spec: Spec, sample: torch.Tensor, selects: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Eager output at defaults plus how to display it (clamp or normalise) and its shape."""
    live = [torch.tensor([p.default], dtype=torch.float32) for p in spec.params if isinstance(p, Live)]
    module = _Wrapper(spec, selects).eval()
    with torch.no_grad():
        torch.manual_seed(0)
        out = module(sample, *_guide(spec), *live)
    if not isinstance(out, torch.Tensor) or out.dim() != 4 or out.shape[1] not in (1, 3, 4):
        raise ValueError(f"output is not an image tensor: {getattr(out, 'shape', type(out))}")
    lo, hi = float(out.amin()), float(out.amax())
    display = "clamp" if lo >= -0.02 and hi <= 1.02 else "normalize"
    info = {"channels": int(out.shape[1]), "display": display, "height": int(out.shape[-2]), "width": int(out.shape[-1])}
    return out, info


folded_runs: set[bool] = set()  # set by _export when a stochastic graph does not vary between runs


def _export(spec: Spec, selects: dict[str, Any], sample: torch.Tensor, ref: torch.Tensor, dynamic: bool) -> tuple[bytes, list[str], float] | None:
    live_params = [p for p in spec.params if isinstance(p, Live)]
    module = _Wrapper(spec, selects).eval()
    live = tuple(torch.tensor([p.default], dtype=torch.float32) for p in live_params)
    guide = _guide(spec)
    inputs = (sample, *guide, *live)
    kwargs: dict[str, Any] = {}
    if dynamic:
        B, H, W = Dim("B", min=1, max=64), Dim("H", min=16, max=4096), Dim("W", min=16, max=4096)
        image_dims = {0: B, 2: H, 3: W}
        kwargs["dynamic_shapes"] = (image_dims,) + tuple(image_dims for _ in guide) + tuple({} for _ in live)
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            program = torch.onnx.export(module, inputs, dynamo=True, opset_version=OPSET, verbose=False, **kwargs)
        buf = io.BytesIO()
        program.save(buf)
        data = buf.getvalue()
        onnx.checker.check_model(onnx.load_from_string(data))
        sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        if len(names) != len(inputs):
            _log(f"      {'dynamic' if dynamic else 'static'}: exporter folded an input away ({names})")
            return None
        feeds = {n: t.numpy() for n, t in zip(names, inputs)}
        t0 = time.perf_counter()
        out = sess.run(None, feeds)[0]
        latency = (time.perf_counter() - t0) * 1000
        if spec.stochastic and float(np.abs(sess.run(None, feeds)[0] - out).max()) < 1e-7:
            _log(f"      {'dynamic' if dynamic else 'static'}: two runs are identical; the random draw was folded at export")
            folded_runs.add(True)
        if dynamic:
            # the graph must really accept another size and batch
            x2 = torch.rand(2, 3, 96, 128)
            guide2 = tuple(torch.rand(2, 3, 96, 128) for _ in guide)
            feeds2 = dict(feeds)
            feeds2[names[0]] = x2.numpy()
            for name, g in zip(names[1 : 1 + len(guide2)], guide2):
                feeds2[name] = g.numpy()
            out2 = sess.run(None, feeds2)[0]
            with torch.no_grad():
                torch.manual_seed(0)
                ref2 = module(x2, *guide2, *live)
            if tuple(out2.shape) != tuple(ref2.shape):
                _log(f"      dynamic: output shape {out2.shape} vs eager {tuple(ref2.shape)} at 2x3x96x128")
                return None
            if not spec.stochastic and float(np.abs(out2 - ref2.numpy()).max()) > 2e-3:
                _log(f"      dynamic: onnxruntime disagrees with eager at 2x3x96x128 (max {np.abs(out2 - ref2.numpy()).max():.3g})")
                return None
        if not spec.stochastic and float(np.abs(out - ref.numpy()).max()) > 2e-3:
            _log(f"      {'dynamic' if dynamic else 'static'}: onnxruntime disagrees with eager (max {np.abs(out - ref.numpy()).max():.3g})")
            return None
        return data, names, latency
    except Exception as e:  # noqa: BLE001 - any failure means the next fallback
        first = str(e).splitlines()[0][:140] if str(e) else type(e).__name__
        _log(f"      {'dynamic' if dynamic else 'static'}: {type(e).__name__}: {first}")
        return None


def render_frames(spec: Spec, images: list[dict[str, str]], display: str) -> dict[str, Any] | None:
    numeric = [p for p in spec.params if isinstance(p, Live)] or [p for p in spec.params if isinstance(p, Select)]
    slug = slug_of(spec.id)
    rel_dir = f"frames/{slug}"
    if (ROOT / rel_dir).exists():
        shutil.rmtree(ROOT / rel_dir)
    if not numeric:
        primary, values = None, [None]
    else:
        primary = numeric[0]
        values = (
            [round(primary.min + (primary.max - primary.min) * i / (FRAME_STEPS - 1), 4) for i in range(FRAME_STEPS)]
            if isinstance(primary, Live)
            else list(primary.choices)
        )
    fixed = {p.name: p.default for p in spec.params if p is not primary}
    try:
        for im in images:
            (ROOT / rel_dir / im["id"]).mkdir(parents=True, exist_ok=True)
            x = _to_tensor(Image.open(ROOT / im["file"]))
            for i, v in enumerate(values):
                kwargs = dict(fixed)
                if primary is not None:
                    kwargs[primary.name] = torch.tensor([v], dtype=torch.float32) if isinstance(primary, Live) else v
                if spec.guidance:
                    kwargs["guide"] = _guide(spec)[0]
                with torch.no_grad():
                    torch.manual_seed(0)
                    y = spec.call(x, **kwargs)
                _display(y, display).save(ROOT / rel_dir / im["id"] / f"{i:02d}.webp", quality=88)
    except Exception as e:  # noqa: BLE001
        _log(f"      frames: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        return None
    return {"param": primary.name if primary else None, "values": values, "dir": rel_dir, "fixed": fixed, "note": spec.frames_note}


def _param_dict(p: Live | Select) -> dict[str, Any]:
    d = asdict(p)
    d["kind"] = p.kind
    d["type"] = p.type
    if getattr(p, "derived", None):
        d["derived"] = p.derived
    return d


def build_spec(spec: Spec, images: list[dict[str, str]], sample: torch.Tensor) -> dict[str, Any]:
    _log(f"== {spec.id}")
    slug = slug_of(spec.id)
    entry: dict[str, Any] = {
        "id": spec.id,
        "slug": slug,
        "name": spec.id.rsplit(".", 1)[1],
        "package": spec.id.rsplit(".", 1)[0],
        "status": "unsupported",
        "summary": spec.summary,
        "doc_url": doc_url(spec.id),
        "snippet": spec.snippet,
        "module_snippet": spec.module,
        "rust": asdict(spec.rust) if spec.rust else None,
        "stochastic": spec.stochastic,
        "pre": spec.pre,
        "guidance": next((im for im in images if im["id"] == spec.guidance), None) if spec.guidance else None,
        "params": [_param_dict(p) for p in spec.params],
    }
    selects = [p for p in spec.params if isinstance(p, Select)]
    combos: list[dict[str, Any]] = [{}]
    for p in selects:
        combos = [{**c, p.name: v} for c in combos for v in p.choices]
    default_sel = {p.name: p.default for p in selects}
    try:
        ref, out_info = _output_info(spec, sample, default_sel)
    except Exception as e:  # noqa: BLE001
        entry["reason"] = f"failed to run eagerly: {type(e).__name__}: {str(e).splitlines()[0][:100]}"
        _log(f"      {entry['reason']}")
        return entry
    entry["output"] = out_info

    if not spec.frames_only:
        graphs, graph_kb, latencies, inputs = {}, {}, [], []
        dynamic_all = True
        ok = True
        folded_runs.clear()
        graph_dir = ROOT / "graphs" / slug
        if graph_dir.exists():
            shutil.rmtree(graph_dir)
        for sel in combos:
            try:
                ref_sel = ref if sel == default_sel else _output_info(spec, sample, sel)[0]
            except Exception as e:  # noqa: BLE001 - one bad variant disqualifies the graph set
                _log(f"      {sel}: eager run failed: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
                ok = False
                break
            result = None
            dyn = False
            if not spec.static_only:
                result = _export(spec, sel, sample, ref_sel, dynamic=True)
                dyn = result is not None
            if result is None:
                result = _export(spec, sel, sample, ref_sel, dynamic=False)
            if result is None:
                ok = False
                break
            dynamic_all = dynamic_all and dyn
            data, inputs, latency = result
            key = "|".join(_fmt(sel[p.name]) for p in selects) or "default"
            path = graph_dir / (key.replace("|", "_").replace(".", "p") + ".onnx")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            graphs[key] = str(path.relative_to(ROOT))
            graph_kb[key] = round(len(data) / 1024, 1)
            latencies.append(latency)
            _log(f"      {key:>10s}: {len(data) / 1024:7.1f} KB  {'dynamic' if dyn else 'static 256px'}  onnxruntime {latency:.1f} ms")
        if ok:
            entry.update(
                status="live",
                mode="onnx",
                graphs=graphs,
                graph_kb=graph_kb,
                inputs=inputs,
                select_order=[p.name for p in selects],
                dynamic=dynamic_all,
                cpu_latency_ms=round(float(np.median(latencies)), 1),
            )
            if spec.stochastic and folded_runs:
                # no Re-roll button: every run of this graph gives the same result
                entry["stochastic"] = False
                entry["sampling_folded"] = True
            return entry
        if graph_dir.exists():
            shutil.rmtree(graph_dir)
    frames = render_frames(spec, images, out_info["display"])
    if frames is None:
        entry["reason"] = "could not export nor render frames"
        return entry
    entry.update(status="frames", mode="frames", frames=frames)
    _log(f"      frames: {len(frames['values']) * len(images)} webp along {frames['param']}")
    return entry


def _recheck_variability(entry: dict[str, Any], sample: torch.Tensor) -> None:
    """Run every graph of a cached stochastic entry twice; drop the Re-roll when nothing changes."""
    varies = False
    for path in entry["graphs"].values():
        sess = ort.InferenceSession(str(ROOT / path), providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        feeds = {names[0]: sample.numpy()}
        if entry.get("guidance"):
            feeds[names[1]] = sample.numpy()
        live = [p for p in entry["params"] if p["kind"] == "live"]
        for name, p in zip(names[len(feeds):], live):
            feeds[name] = np.array([p["default"]], dtype=np.float32)
        a, b = sess.run(None, feeds)[0], sess.run(None, feeds)[0]
        varies = varies or float(np.abs(a - b).max()) > 1e-7
    entry["stochastic"] = varies
    entry["sampling_folded"] = not varies
    if not varies:
        _log(f"      {entry['id']}: random draw folded at export, no Re-roll")


# --------------------------------------------------------------------------- catalog


def _public_names(package: str) -> list[tuple[str, str]]:
    mod = __import__(package, fromlist=["_"])
    names = getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]
    out = []
    for n in sorted(set(names)):
        obj = getattr(mod, n, None)
        if obj is None or inspect.ismodule(obj):
            continue
        if not (inspect.isclass(obj) or callable(obj)):
            continue
        if package == "kornia.augmentation" and (n.endswith("Base2D") or n.endswith("Base3D") or n.endswith("BaseV2") or "Dispather" in n):
            continue
        out.append((n, "class" if inspect.isclass(obj) else "function"))
    return out


def _survey() -> dict[str, dict[str, str]]:
    """Operator id -> {onnx status, package} from kornia's export survey snapshot, when available."""
    path = KORNIA_REPO / "docs" / "source" / "_data" / "export_support.json"
    if not path.exists():
        _log(f"no export survey at {path}; the catalog lists only the image packages")
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict[str, str]] = {}
    for c in data["cases"]:
        if c.get("onnx") == "n/a":
            continue
        op_id = f"{c['package']}.{c['operator']}"
        prev = out.get(op_id)
        status = c["onnx"]
        # an operator counts as exporting when any configuration does
        if prev is None or (status.startswith("ok") and not prev["onnx"].startswith("ok")):
            out[op_id] = {"onnx": status, "package": c["package"], "section": c.get("section", "")}
    return out


def build_catalog(built: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every public image operator of PACKAGES: built demos first-class, the rest greyed out with a reason.

    Kernel/matrix builders, enums and tuple-returning helpers (``HIDDEN``) are left out entirely, and
    the nn.Module form of a demoed function is folded into that function's entry (shown on its page
    as the second Python block) instead of being listed twice.
    """
    reasons = {name: why for why, names in UNSUPPORTED_REASONS.items() for name in names.split()}
    survey = _survey()
    ops: list[dict[str, Any]] = []

    for package, _title in PACKAGES:
        names = _public_names(package)
        # MotionBlur3D ~ motion_blur3d, Rgb255ToNormals ~ rgb255_to_normals: same letters once
        # underscores and case are dropped
        functions_norm = {n.replace("_", "").lower() for n, k in names if k == "function"}
        for name, kind in names:
            op_id = f"{package}.{name}"
            if name in HIDDEN or op_id in MODULE_FORMS:
                continue
            # a class whose function form is also public is the module form of that function
            # (JointBilateralBlur / joint_bilateral_blur): list the function only
            if kind == "class" and name.lower() in functions_norm:
                continue
            if op_id in built:
                entry = {**built[op_id], "kind": kind}
                module_names = [m for m, fn in MODULE_FORMS.items() if fn == op_id]
                if module_names:
                    entry["module_name"] = module_names[0]
                ops.append(entry)
                continue
            entry: dict[str, Any] = {"id": op_id, "slug": slug_of(op_id), "name": name, "package": package, "kind": kind, "doc_url": doc_url(op_id), "status": "unsupported"}
            if name in reasons:
                entry["reason"] = reasons[name]
            elif op_id in survey and not survey[op_id]["onnx"].startswith("ok"):
                entry["reason"] = "does not export to ONNX yet"
            else:
                entry["reason"] = "no demo yet"
            ops.append(entry)
    return ops


# --------------------------------------------------------------------------- pages


def write_pages(registry: dict[str, Any]) -> int:
    template = (HERE / "op_template.html").read_text()
    ops_dir = ROOT / "ops"
    if ops_dir.exists():
        shutil.rmtree(ops_dir)
    n = 0
    for op in registry["ops"]:
        if op["status"] == "unsupported" or op.get("alias_of"):
            continue
        page = template
        for key, value in {
            "{{id}}": op["id"],
            "{{slug}}": op["slug"],
            "{{name}}": op["name"],
            "{{title}}": op["name"].replace("_", " "),
            "{{summary}}": op.get("summary", ""),
            "{{doc_url}}": op["doc_url"],
            "{{ort_version}}": ORT_WEB_VERSION,
        }.items():
            page = page.replace(key, value)
        out = ops_dir / op["slug"] / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page)
        n += 1
    return n


# --------------------------------------------------------------------------- main


def main(argv: list[str]) -> None:
    pages_only = "--pages-only" in argv
    wanted = [a for a in argv if not a.startswith("--")] or None
    images = prepare_images()
    sample = _to_tensor(Image.open(ROOT / images[0]["file"]))
    registry_path = ROOT / "registry.json"
    previous: dict[str, dict[str, Any]] = {}
    if registry_path.exists() and (wanted or pages_only):
        previous = {o["id"]: o for o in json.loads(registry_path.read_text())["ops"] if o["status"] != "unsupported" and not o.get("alias_of")}
    specs = [s for s in SPECS if not wanted or any(s.id.endswith(w) for w in wanted)]
    if wanted and not specs:
        sys.exit(f"no spec matches {wanted}")
    built = dict(previous)
    if pages_only:
        for entry in built.values():
            if entry.get("mode") == "onnx" and (entry.get("stochastic") or entry.get("sampling_folded")):
                _recheck_variability(entry, sample)
    if not pages_only:
        for spec in specs:
            built[spec.id] = build_spec(spec, images, sample)
    ops = build_catalog(built)
    registry = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "kornia": K.__version__,
        "torch": torch.__version__,
        "onnx_opset": OPSET,
        "onnxruntime_web": ORT_WEB_VERSION,
        "size": SIZE,
        "images": images,
        "videos": json.loads((HERE / "samples" / "videos" / "videos.json").read_text()) if (HERE / "samples" / "videos" / "videos.json").exists() else [],
        "packages": [{"id": p, "title": t} for p, t in PACKAGES],
        "ops": ops,
    }
    registry_path.write_text(json.dumps(registry, indent=1) + "\n")
    n_pages = write_pages(registry)
    counts = {s: sum(1 for o in ops if o["status"] == s) for s in ("live", "frames", "unsupported")}
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file() and f.suffix in (".onnx", ".webp", ".png", ".jpg"))
    _log(f"wrote registry.json: {len(ops)} operators listed ({counts}), {n_pages} pages, {total / 1e6:.1f} MB of assets")


if __name__ == "__main__":
    main(sys.argv[1:])
