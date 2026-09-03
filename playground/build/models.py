#!/usr/bin/env python
"""Build the Models section of the playground: ready-to-use neural networks, separate from the operators.

Small models are exported here and vendored under ``playground/models/``. Large ones are referenced by
URL on the Hugging Face hub (``kornia/ONNX_models`` and ``onnx-community``), which serves them with
CORS headers, so the browser downloads them straight from the hub and this repository stays small.

    python playground/build/models.py            # export, verify, write models/index.json and the pages

Outputs: ``playground/models/index.json`` (the model registry the widget reads), one ``.onnx`` per vendored
model, ``playground/models/coco_labels.json``, and ``playground/models/<slug>/index.html`` deep-link pages.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # playground/
OUT = ROOT / "models"
BUILD = ROOT / "build"

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

HF_KORNIA = "https://huggingface.co/kornia/ONNX_models/resolve/main/"
HF_ONNX_COMMUNITY = "https://huggingface.co/onnx-community/"


def inline_external_data(path: Path) -> None:
    """The dynamo exporter may park big constants in a .data side file; the browser fetches one file."""
    import onnx

    proto = onnx.load(str(path), load_external_data=True)
    onnx.save_model(proto, str(path), save_as_external_data=False)
    side = path.with_name(path.name + ".data")
    if side.exists():
        side.unlink()


def sample_image(size: int) -> np.ndarray:
    img = Image.open(ROOT / "images" / sorted(p.name for p in (ROOT / "images").glob("*.png"))[0]).convert("RGB").resize((size, size))
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


def run_onnx(path: Path, feeds: dict[str, np.ndarray]) -> tuple[list[np.ndarray], float]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    t0 = time.time()
    outs = sess.run(None, feeds)
    return outs, time.time() - t0


def face_sample() -> dict | None:
    """A still with a face for the YuNet page, cut from the astronaut clip (NASA, public domain)."""
    import subprocess

    source = BUILD / "samples" / "videos" / "astronaut_source.webm"
    if not source.exists():
        print("  (no astronaut_source.webm; the YuNet page keeps the default samples)")
        return None
    vf = "crop='min(iw,ih)':'min(iw,ih)',scale={s}:{s}:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "15", "-i", str(source), "-frames:v", "1", "-vf", vf.format(s=320), "-q:v", "3", str(OUT / "astronaut_320.jpg")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "15", "-i", str(source), "-frames:v", "1", "-vf", vf.format(s=96), "-q:v", "4", str(OUT / "astronaut_thumb.jpg")], check=True)
    return {"file": "models/astronaut_320.jpg", "thumb": "models/astronaut_thumb.jpg", "label": "Astronaut on the ISS (NASA)"}


def export_yunet() -> dict:
    """YuNet backbone only: the box decoding and NMS live in models.js (the traced NMS loop does not generalise)."""
    from kornia.contrib import FaceDetector

    fd = FaceDetector().eval()
    core = fd.model
    size = 320
    x = torch.from_numpy(sample_image(size) * 255.0)  # FaceDetector works on 0..255 floats
    path = OUT / "yunet.onnx"
    torch.onnx.export(core, (x,), str(path), dynamo=False, opset_version=17, input_names=["image"],
                      output_names=["loc", "conf", "iou"],
                      dynamic_axes={"image": {0: "B", 2: "H", 3: "W"}, "loc": {0: "B", 1: "N"}, "conf": {0: "B", 1: "N"}, "iou": {0: "B", 1: "N"}})
    with torch.no_grad():
        ref = core(x)
    outs, dt = run_onnx(path, {"image": x.numpy()})
    for got, key in zip(outs, ("loc", "conf", "iou")):
        diff = float(np.abs(got - ref[key].numpy()).max())
        assert diff < 1e-3, (key, diff)
    print(f"  yunet.onnx: {path.stat().st_size / 1e6:.2f} MB, {dt * 1000:.0f} ms, verified vs eager")
    sample = face_sample()
    return {
        "id": "kornia.contrib.FaceDetector",
        "slug": "face-detection-yunet",
        "name": "YuNet face detection",
        "title": "YuNet",
        "subtitle": "face detection",
        "task": "Detection",
        "summary": "Tiny face detector with five facial keypoints, a few milliseconds per image.",
        "url": "models/yunet.onnx",
        "sample": sample,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "hosted": "kornia.org",
        "input": {"name": "image", "size": size, "scale": 255.0, "mean": None, "std": None},
        "pipeline": {"imports": ["from kornia.contrib import FaceDetector"], "build": "FaceDetector()", "call": "model(out * 255.0)[0]  # (N, 15): box, 5 keypoints, score"},
        "output": {"kind": "faces_yunet", "threshold": 0.5, "nms": 0.3, "variance": [0.1, 0.2],
                   "min_sizes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]], "steps": [8, 16, 32, 64]},
        "params": "76 k parameters",
        "paper": {"title": "YuNet: A Tiny Millisecond-level Face Detector", "url": "https://link.springer.com/article/10.1007/s11633-023-1423-y"},
        "license": "MIT (libfacedetection weights)",
        "source": "https://github.com/ShiqiYu/libfacedetection",
        "doc_url": "https://kornia.readthedocs.io/en/latest/contrib.html#kornia.contrib.FaceDetector",
        "snippets": [
            {"label": "PyTorch", "code": "import kornia as K\nfrom kornia.contrib import FaceDetector, FaceDetectorResult\n\ndetector = FaceDetector()                 # YuNet, pretrained\nfaces = detector(x * 255.0)[0]            # (N, 15): box, 5 keypoints, score; x in [0, 1]\nfor face in [FaceDetectorResult(f) for f in faces]:\n    print(face.top_left, face.bottom_right, face.score)"},
        ],
        "note": "Backbone exported from kornia with dynamic height and width; box decoding and NMS run in JavaScript on the page.",
    }


def export_small_sr() -> dict:
    from kornia.models.small_sr import SmallSRNetWrapper

    sr = SmallSRNetWrapper(upscale_factor=3, pretrained=True).eval()
    size = 192
    x = torch.from_numpy(sample_image(size))
    path = OUT / "small_sr_x3.onnx"
    torch.onnx.export(sr, (x,), str(path), dynamo=False, opset_version=17, input_names=["image"], output_names=["upscaled"],
                      dynamic_axes={"image": {0: "B", 2: "H", 3: "W"}, "upscaled": {0: "B", 2: "H3", 3: "W3"}})
    with torch.no_grad():
        ref = sr(x).numpy()
    outs, dt = run_onnx(path, {"image": x.numpy()})
    diff = float(np.abs(outs[0] - ref).max())
    assert diff < 1e-3, diff
    print(f"  small_sr_x3.onnx: {path.stat().st_size / 1e6:.2f} MB, {dt * 1000:.0f} ms, verified vs eager")
    return {
        "id": "kornia.models.small_sr.SmallSRNetWrapper",
        "slug": "super-resolution-espcn",
        "name": "Super-resolution x3 (ESPCN)",
        "title": "ESPCN ×3",
        "subtitle": "super-resolution",
        "task": "Enhancement",
        "summary": "Sub-pixel convolution network that triples the resolution of an image.",
        "url": "models/small_sr_x3.onnx",
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "hosted": "kornia.org",
        "input": {"name": "image", "size": size, "scale": 1.0, "mean": None, "std": None},
        "output": {"kind": "image", "scale": 3},
        "pipeline": {"imports": ["from kornia.models.small_sr import SmallSRNetWrapper"], "build": "SmallSRNetWrapper(upscale_factor=3)", "call": "model(out)             # (B, 3, 3H, 3W)"},
        "params": "60 k parameters",
        "paper": {"title": "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN", "url": "https://arxiv.org/abs/1609.05158"},
        "license": "BSD-3 (PyTorch tutorial weights)",
        "source": "https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html",
        "doc_url": "https://kornia.readthedocs.io/en/latest/models.html",
        "snippets": [
            {"label": "PyTorch", "code": "from kornia.models.small_sr import SmallSRNetWrapper\n\nupscale = SmallSRNetWrapper(upscale_factor=3)   # pretrained\ny = upscale(x)                                  # (B, 3, 3H, 3W), x in [0, 1]"},
        ],
        "note": "Dynamic height and width: the downloaded graph runs on any image size.",
    }


# ---------------------------------------------------------------- local features and classification


def _nms_numpy(heat: np.ndarray, k: int, thr: float) -> np.ndarray:
    """(y, x) positions that are the maximum of their k x k window and above thr; mirrors kornia's nms2d."""
    from numpy.lib.stride_tricks import sliding_window_view

    pad = k // 2
    padded = np.pad(heat, pad, mode="constant", constant_values=-np.inf)
    win = sliding_window_view(padded, (k, k)).max(axis=(-1, -2))
    return np.argwhere((heat == win) & (heat > thr))


def _sample_bilinear(fm: np.ndarray, xs: np.ndarray, ys: np.ndarray, W: int, H: int) -> np.ndarray:
    """grid_sample(bilinear, align_corners=False, zeros) at XFeat's normgrid (x / (W-1)) coordinates. fm: (C, h, w)."""
    C, h, w = fm.shape
    gx = 2.0 * xs / (W - 1) - 1.0
    gy = 2.0 * ys / (H - 1) - 1.0
    ix = ((gx + 1) * w - 1) / 2
    iy = ((gy + 1) * h - 1) / 2
    x0 = np.floor(ix).astype(int)
    y0 = np.floor(iy).astype(int)
    out = np.zeros((len(xs), C), dtype=np.float32)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi, yi = x0 + dx, y0 + dy
        wgt = (1 - np.abs(ix - xi)) * (1 - np.abs(iy - yi))
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out[ok] += (fm[:, yi[ok], xi[ok]] * wgt[ok]).T
    return out


def export_xfeat() -> dict:
    """XFeat's network only; heatmap NMS, scoring and descriptor sampling are re-done in models.js.

    The build re-implements that post-processing in numpy and checks it against kornia's detectAndCompute,
    so the JavaScript port has a verified reference.
    """
    import torch.nn.functional as F
    from kornia.feature import XFeat

    xf = XFeat.from_pretrained(top_k=512).eval()

    class Dense(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = xf.net

        def forward(self, image: torch.Tensor):
            feats, kpts, heat = self.net(image)
            return XFeat._get_kpts_heatmap(kpts), heat, F.normalize(feats, dim=1)

    size = 256
    img = torch.from_numpy(np.asarray(Image.open(OUT / "astronaut_320.jpg").convert("RGB").resize((size, size)), dtype=np.float32).transpose(2, 0, 1)[None] / 255.0) if (OUT / "astronaut_320.jpg").exists() else torch.from_numpy(sample_image(size))
    path = OUT / "xfeat.onnx"
    # the legacy exporter rejects XFeat's InstanceNorm under dynamic shapes; the dynamo exporter at a fixed
    # 256x256 is fine (the page always feeds 256), with its side-file constants folded back into one file
    torch.onnx.export(Dense().eval(), (img,), str(path), dynamo=True, opset_version=18, input_names=["image"],
                      output_names=["keypoint_heatmap", "reliability", "descriptors"])
    inline_external_data(path)
    outs, dt = run_onnx(path, {"image": img.numpy()})
    heat, rel, desc = outs[0][0, 0], outs[1][0], outs[2][0]
    # numpy post-processing == what models.js does
    pos = _nms_numpy(heat, 5, 0.05)
    ys, xs = pos[:, 0].astype(np.float32), pos[:, 1].astype(np.float32)
    score = heat[pos[:, 0], pos[:, 1]] * _sample_bilinear(rel, xs, ys, size, size)[:, 0]
    order = np.argsort(-score)[:512]
    order = order[score[order] > 0]
    kp = np.stack([xs[order], ys[order]], 1)
    d = _sample_bilinear(desc, kp[:, 0], kp[:, 1], size, size)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    with torch.no_grad():
        ref = xf.detectAndCompute(img, top_k=512)[0]
    ref_kp = ref["keypoints"].numpy()
    common = {tuple(map(int, k)) for k in ref_kp} & {tuple(map(int, k)) for k in kp}
    agree = len(common) / max(1, len(ref_kp))
    ref_d = ref["descriptors"].numpy()
    lookup = {tuple(map(int, k)): i for i, k in enumerate(kp)}
    cos = np.mean([float(ref_d[i] @ d[lookup[tuple(map(int, k))]]) for i, k in enumerate(ref_kp) if tuple(map(int, k)) in lookup])
    assert agree > 0.95 and cos > 0.99, (agree, cos)
    print(f"  xfeat.onnx: {path.stat().st_size / 1e6:.2f} MB, {dt * 1000:.0f} ms; post-processing check: {len(ref_kp)} keypoints, {agree:.0%} identical, descriptor cosine {cos:.4f}")
    return {
        "id": "kornia.feature.XFeat",
        "slug": "local-features-xfeat",
        "name": "XFeat keypoints and matching",
        "title": "XFeat",
        "subtitle": "keypoints + matching",
        "task": "Local features",
        "summary": "Accelerated features: keypoints with 64-d descriptors, matched against a warped copy of the same frame.",
        "url": "models/xfeat.onnx",
        "graphs": {"warp": "home/affine_256.onnx"},
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "hosted": "kornia.org",
        "input": {"name": "image", "size": size, "scale": 1.0, "mean": None, "std": None},
        "output": {"kind": "features_xfeat", "nms": 5, "threshold": 0.05, "top_k": 512, "min_cossim": 0.82},
        "params": "660 k parameters",
        "paper": {"title": "XFeat: Accelerated Features for Lightweight Image Matching (CVPR 2024)", "url": "https://arxiv.org/abs/2404.19174"},
        "license": "Apache-2.0",
        "source": "https://github.com/verlab/accelerated_features",
        "doc_url": "https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.XFeat",
        "snippets": [
            {"label": "PyTorch", "code": "from kornia.feature import XFeat\n\nxfeat = XFeat.from_pretrained(top_k=512)\nfeats = xfeat.detectAndCompute(x)[0]                  # keypoints (N, 2), scores (N,), descriptors (N, 64)\nidx0, idx1 = xfeat.match(feats[\"descriptors\"], feats_other[\"descriptors\"], min_cossim=0.82)"},
        ],
        "note": "The network runs on the frame and on an affine-warped copy; NMS, scoring, descriptor sampling and mutual-nearest-neighbour matching run in JavaScript, checked against kornia at build time.",
    }


def export_keynet_hardnet() -> dict:
    from kornia.feature import HardNet, KeyNet

    keynet = KeyNet(pretrained=True).eval()
    hardnet = HardNet(pretrained=True).eval()
    size = 256
    gray = torch.from_numpy(sample_image(size)).mean(1, keepdim=True)
    kp_path = OUT / "keynet.onnx"
    # fixed 256x256 with the legacy exporter: under dynamic shapes it rejects KeyNet's reflect padding, and the
    # dynamo exporter produces a graph whose response disagrees with eager by half its range (a kornia bug to file)
    torch.onnx.export(keynet, (gray,), str(kp_path), dynamo=False, opset_version=17, input_names=["image"], output_names=["response"])
    with torch.no_grad():
        ref = keynet(gray).numpy()
    outs, dt1 = run_onnx(kp_path, {"image": gray.numpy()})
    diff, scale = float(np.abs(outs[0] - ref).max()), float(np.abs(ref).max())
    assert diff < 1e-2 * scale, (diff, scale)   # relative: the response map is not bounded to [0, 1]
    print(f"  keynet response: max |diff| {diff:.2e} on a range of {scale:.2e}")
    patches = torch.rand(40, 1, 32, 32)
    hn_path = OUT / "hardnet.onnx"
    torch.onnx.export(hardnet, (patches,), str(hn_path), dynamo=False, opset_version=17, input_names=["patches"], output_names=["descriptors"],
                      dynamic_axes={"patches": {0: "N"}, "descriptors": {0: "N"}})
    with torch.no_grad():
        ref = hardnet(patches).numpy()
    outs, dt2 = run_onnx(hn_path, {"patches": patches.numpy()})
    assert float(np.abs(outs[0] - ref).max()) < 1e-3
    total = (kp_path.stat().st_size + hn_path.stat().st_size) / 1e6
    print(f"  keynet.onnx + hardnet.onnx: {total:.2f} MB, {dt1 * 1000:.0f} + {dt2 * 1000:.0f} ms, both verified vs eager")
    return {
        "id": "kornia.feature.KeyNet+HardNet",
        "slug": "local-features-keynet-hardnet",
        "name": "KeyNet + HardNet",
        "title": "KeyNet + HardNet",
        "subtitle": "detector + descriptor",
        "task": "Local features",
        "summary": "The classic two-stage pipeline: a learned keypoint detector, then a 128-d patch descriptor, matched against a warped copy.",
        "url": "models/keynet.onnx",
        "graphs": {"descriptor": "models/hardnet.onnx", "warp": "home/affine_256.onnx"},
        "size_mb": round(total, 2),
        "hosted": "kornia.org",
        "input": {"name": "image", "size": size, "scale": 1.0, "mean": None, "std": None, "gray": True},
        "output": {"kind": "features_keynet_hardnet", "nms": 7, "top_k": 300, "patch": 32, "support": 40, "min_cossim": 0.55},
        "params": "5.9 k + 1.3 M parameters",
        "paper": {"title": "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters (ICCV 2019); HardNet: Working hard to know your neighbor's margins (NeurIPS 2017)", "url": "https://arxiv.org/abs/1904.00889"},
        "license": "MIT (Key.Net) and MIT (HardNet) weights",
        "source": "https://github.com/axelBarroso/Key.Net-Pytorch",
        "doc_url": "https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.KeyNet",
        "snippets": [
            {"label": "PyTorch", "code": "import kornia.feature as KF\n\ndetector = KF.KeyNetDetector(pretrained=True, num_features=300)\ndescriptor = KF.LAFDescriptor(KF.HardNet(pretrained=True))\nlafs, responses = detector(gray)                        # local affine frames on a (B, 1, H, W) image\ndescs = descriptor(gray, lafs)                          # (B, 300, 128)\nmatch_dists, idxs = KF.match_mnn(descs[0], descs_other[0])"},
        ],
        "note": "Single-scale on the page: NMS on the response map, a 48-px window around each keypoint resized to a 32x32 patch, then the descriptor. kornia's KeyNetDetector adds a scale pyramid and affine frames.",
    }


def export_tinyvit() -> dict:
    from kornia.models.tiny_vit import TinyViT

    model = TinyViT.from_config("5m", pretrained=True).eval()
    size = 224
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    x = torch.from_numpy((sample_image(size) - mean) / std)
    path = OUT / "tinyvit_5m.onnx"
    torch.onnx.export(model, (x,), str(path), dynamo=False, opset_version=17, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "B"}, "logits": {0: "B"}})
    with torch.no_grad():
        ref = model(x).numpy()
    outs, dt = run_onnx(path, {"image": x.numpy()})
    assert float(np.abs(outs[0] - ref).max()) < 1e-2, float(np.abs(outs[0] - ref).max())
    labels = [line.strip() for line in (BUILD / "imagenet_classes.txt").read_text().splitlines() if line.strip()]
    assert len(labels) == 1000, len(labels)
    (OUT / "imagenet_labels.json").write_text(json.dumps(labels))
    print(f"  tinyvit_5m.onnx: {path.stat().st_size / 1e6:.1f} MB, {dt * 1000:.0f} ms, verified vs eager; top-1 on the sample: {labels[int(outs[0][0].argmax())]}")
    return {
        "id": "kornia.models.tiny_vit.TinyViT",
        "slug": "classification-tinyvit",
        "name": "TinyViT-5M classification",
        "title": "TinyViT-5M",
        "subtitle": "ImageNet classification",
        "task": "Classification",
        "summary": "A 5-million-parameter vision transformer distilled on ImageNet-22k, fine-tuned on ImageNet-1k: top-5 classes for a frame.",
        "url": "models/tinyvit_5m.onnx",
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "hosted": "kornia.org",
        "input": {"name": "image", "size": size, "scale": 1.0, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "fixed": True},
        "output": {"kind": "classification", "labels": "imagenet", "top": 5},
        "pipeline": {"imports": ["from kornia.models.tiny_vit import TinyViT"], "build": 'TinyViT.from_config("5m", pretrained=True).eval()', "call": "model(out).softmax(-1)   # (B, 1000); normalisation is inside the exported graph, apply kornia.enhance.normalize in PyTorch"},
        "params": "5.4 M parameters, 79.1 % ImageNet-1k top-1",
        "paper": {"title": "TinyViT: Fast Pretraining Distillation for Small Vision Transformers (ECCV 2022)", "url": "https://arxiv.org/abs/2207.10666"},
        "license": "MIT (Microsoft Cream weights)",
        "source": "https://github.com/microsoft/Cream/tree/main/TinyViT",
        "doc_url": "https://kornia.readthedocs.io/en/latest/models/tiny_vit.html",
        "snippets": [
            {"label": "PyTorch", "code": "import torch\nfrom kornia.models.tiny_vit import TinyViT\nfrom kornia.enhance import normalize\n\nmodel = TinyViT.from_config(\"5m\", pretrained=True).eval()     # ImageNet-1k head\nx = normalize(x224, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))\nprobs = model(x).softmax(-1)                             # (B, 1000)"},
        ],
        "note": "224x224 input, ImageNet normalisation; the page shows the five most likely ImageNet classes.",
    }


def rtdetr() -> dict:
    return {
        "id": "kornia.contrib.object_detection.RTDETRDetectorBuilder",
        "slug": "object-detection-rtdetr",
        "name": "RT-DETR r18 object detection",
        "title": "RT-DETR r18",
        "subtitle": "object detection",
        "task": "Detection",
        "summary": "Real-time transformer detector trained on COCO: 80 object classes with boxes and scores.",
        "url": HF_KORNIA + "models/kornia.models.detection.rtdetr_r18vd_480x480.onnx",
        "size_mb": 80.8,
        "hosted": "huggingface.co/kornia/ONNX_models",
        "input": {"name": "input", "size": 480, "scale": 1.0, "mean": None, "std": None},
        "output": {"kind": "detections", "format": "cls_score_xywh", "labels": "coco", "threshold": 0.4, "coord_size": 480},
        "pipeline": {"imports": ["from kornia.contrib.object_detection import RTDETRDetectorBuilder"], "build": 'RTDETRDetectorBuilder.build("rtdetr_r18vd", image_size=480)', "call": "model(out)[0]        # (N, 6): class_id, score, x, y, w, h"},
        "params": "20 M parameters, ResNet-18 backbone",
        "paper": {"title": "DETRs Beat YOLOs on Real-time Object Detection", "url": "https://arxiv.org/abs/2304.08069"},
        "license": "Apache-2.0",
        "source": "https://github.com/lyuwenyu/RT-DETR",
        "doc_url": "https://kornia.readthedocs.io/en/latest/models.html#rtdetrdetectorbuilder",
        "snippets": [
            {"label": "PyTorch", "code": "from kornia.contrib.object_detection import RTDETRDetectorBuilder\n\ndetector = RTDETRDetectorBuilder.build(\"rtdetr_r18vd\", image_size=480)\ndetections = detector(x)[0]        # (N, 6): class_id, score, x, y, w, h; x in [0, 1]"},
            {"label": "ONNX from the hub", "code": "from kornia.onnx import ONNXSequential\n\n# the same graph this page runs, downloaded from huggingface.co/kornia/ONNX_models\nmodel = ONNXSequential(\"hf://models/kornia.models.detection.rtdetr_r18vd_480x480\")\nout = model(x.numpy())[0]          # (1, 300, 6): class_id, score, x, y, w, h"},
        ],
        "note": "Resizes to 480x480 inside the graph, so any input size works; boxes come back in those coordinates.",
    }


def depth_anything() -> dict:
    return {
        "id": "kornia.models.depth_estimation.DepthAnythingONNXBuilder",
        "slug": "depth-estimation-depth-anything-v2",
        "name": "Depth Anything V2 small",
        "title": "Depth Anything V2 small",
        "subtitle": "depth estimation",
        "task": "Depth",
        "summary": "Monocular relative depth from a single image, ViT-S encoder, 8-bit quantised.",
        "url": HF_ONNX_COMMUNITY + "depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx",
        "size_mb": 27.3,
        "hosted": "huggingface.co/onnx-community",
        "input": {"name": "pixel_values", "size": 364, "scale": 1.0, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": {"kind": "depth"},
        "pipeline": {"imports": ["import onnxruntime as ort"], "build": 'ort.InferenceSession("model_quantized.onnx")   # from huggingface.co/onnx-community/depth-anything-v2-small', "call": 'model.run(None, {"pixel_values": out.numpy()})[0]', "note": "the ImageNet normalisation is part of the exported pipeline graph; in PyTorch apply kornia.enhance.normalize first"},
        "params": "25 M parameters, quantised to int8",
        "paper": {"title": "Depth Anything V2", "url": "https://arxiv.org/abs/2406.09414"},
        "license": "Apache-2.0 (small model)",
        "source": "https://github.com/DepthAnything/Depth-Anything-V2",
        "doc_url": "https://kornia.readthedocs.io/en/latest/models.html",
        "snippets": [
            {"label": "ONNX Runtime", "code": "import onnxruntime as ort\nimport torch\nimport kornia as K\n\n# model_quantized.onnx from huggingface.co/onnx-community/depth-anything-v2-small\nsess = ort.InferenceSession(\"model_quantized.onnx\")\nx = K.geometry.resize(x, (364, 364))                       # a multiple of 14\nx = K.enhance.normalize(x, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))\ndepth = sess.run(None, {\"pixel_values\": x.numpy()})[0]    # (1, 364, 364) relative inverse depth"},
        ],
        "note": "Input side must be a multiple of 14; the page uses 364. Output is relative inverse depth, shown with a colour map.",
    }


def write_pages(models: list[dict]) -> int:
    template = (BUILD / "op_template.html").read_text()
    ort_version = json.loads((ROOT / "registry.json").read_text())["onnxruntime_web"]
    n = 0
    for m in models:
        page = template.replace('data-base="../../" data-op="{{id}}"', 'data-base="../../" data-model="{{id}}"')
        page = page.replace("ops/{{slug}}/", "models/{{slug}}/").replace("{{slug}}", m["slug"])
        page = page.replace("{{id}}", m["id"]).replace("{{name}}", m["name"]).replace("{{summary}}", m["summary"])
        page = page.replace("{{ort_version}}", ort_version)
        d = OUT / m["slug"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(page)
        n += 1
    return n


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print("exporting")
    models = [rtdetr(), depth_anything(), export_yunet(), export_xfeat(), export_keynet_hardnet(), export_tinyvit(), export_small_sr()]
    (OUT / "coco_labels.json").write_text(json.dumps(COCO))
    stale = OUT / "yunet_320.onnx"
    if stale.exists():
        stale.unlink()
    index = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "kornia": __import__("kornia").__version__,
        "onnxruntime": ort.__version__,
        "tasks": ["Detection", "Local features", "Depth", "Classification", "Enhancement"],
        "models": models,
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"models/index.json: {len(models)} models; pages: {write_pages(models)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
