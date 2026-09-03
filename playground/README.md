# kornia.org/playground

Kornia's image operators running live in the visitor's browser. There is no server: every demo is
the real PyTorch operator exported to ONNX and executed with
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) (WebAssembly) on the page. Every
graph can be downloaded, and every demo shows the call in Python (function and module form) and,
where kornia-rs has the operator, in Rust.

```
playground/
  index.html          the shell: operator list on the left, the selected demo in the middle, page tools right
  ops/<slug>/         GENERATED: the same shell with one operator preselected (deep links, SEO)
  pipelines/          the pipeline builder (same shell; see below)
  playground.css      styles on top of ../styles.css tokens
  playground.js       sidebar, routing and the widget, driven by registry.json
  models.js           the Models section: loading with progress, task-specific decoding and drawing
  pipeline.js         the pipeline builder: compose, run, export
  vendor/onnx.proto   the ONNX schema protobuf.js parses in the browser
  registry.json       GENERATED: operators, status, parameters, graph paths, frame sweeps, snippets
  graphs/<slug>/      GENERATED: one .onnx per discrete-parameter value
  frames/<slug>/      GENERATED: fallback frames for operators that cannot export
  images/             GENERATED: 256 px sample images + thumbnails
  videos/             GENERATED: three 4-second 256 px clips (mp4 + webm) + posters, see build/videos.py
  models/             GENERATED: models/index.json, the small vendored .onnx files, COCO labels, one page per model
  build/specs.py      the operator specs (call, parameters, Python/Rust snippets) and catalog rules
  build/models.py     the Models section: exports the small models, references the large ones on the hub
  build/home.py       the homepage demo's affine graph (parameters as inputs, so one draw holds for a whole clip)
  home/               GENERATED: affine_256.onnx for the homepage demo
  build/build.py      the generator; build/op_template.html the per-operator page template
  build/samples/      the Kodak originals
```

## How a widget works

- Slider parameters (`Live`) are graph inputs, so one graph serves every slider position.
- Discrete parameters (`Select`, e.g. `kernel_size`) cannot be graph inputs; each value is its own
  graph and the widget downloads only the one selected.
- Graphs are exported with dynamic batch, height and width when the operator allows it, so the
  downloaded file runs on real images. Operators that specialise their shapes fall back to a fixed
  `1x3x256x256` graph; the page says which.
- Operators that read tensor values in Python (canny's threshold check) cannot export. `build.py`
  renders `FRAME_STEPS` frames along their first parameter instead, and the widget scrubs them.
- Random augmentations keep their sampling inside the graph (`RandomUniformLike`), so "Re-roll"
  runs it again with new parameters. The ONNX RNG is not seedable from the page.
- Outputs outside `[0, 1]` (Lab, HSV, Laplacian, integral image) are min-max normalised for display
  and the caption says so.
- The sample image is drawn to a 256x256 canvas, converted to a `(1, 3, 256, 256)` float tensor,
  and the output tensor is written back to a canvas. Nothing leaves the browser.

## Catalog rules

`build.py` lists every public callable of `kornia.filters`, `kornia.color`, `kornia.enhance`,
`kornia.morphology`, `kornia.geometry.transform` and `kornia.augmentation`, with these rules:

- A function and its `nn.Module` form are one entry, the function. The module form appears as the
  second Python block on the page (`MODULE_FORMS` in `specs.py`, plus a name match for classes
  whose snake_case function is also public).
- Kernel and matrix builders, enums, base classes and tuple-returning helpers are left out
  (`HIDDEN`): they do not process an image.
- Everything else without a spec is greyed out with a reason from `UNSUPPORTED_REASONS`, "does not
  export to ONNX yet" when kornia's export survey (`docs/source/_data/export_support.json` in the
  kornia repo, if checked out next to this one) says so, or "no demo yet". 3D operators and the
  augmentations are listed for later phases.

## Rust snippets

`Rust` in a spec holds the kornia-rs call; the page adds the prelude that loads the sample with
`kornia_io` and allocates the output image. Signatures follow `kornia-imgproc` 0.1.14 on docs.rs
and have not been compiled here; operators without a kornia-rs counterpart say so in the tab.

## Models

The sidebar switches between **Operators** (the catalog above) and **Models**: ready-to-use neural networks,
kept apart from the operators because they are heavy, task-specific and mostly hosted elsewhere.
`build/models.py` writes `models/index.json`, and each entry says where its graph lives:

- small models are exported here and vendored, each verified against eager kornia at build time: YuNet
  (0.35 MB), XFeat (2.8 MB), KeyNet + HardNet (5.4 MB), TinyViT-5M (27 MB) and the ESPCN super-resolution
  network (0.25 MB);
- large models are fetched by the browser straight from the Hugging Face hub, which serves them with CORS
  headers: RT-DETR r18 from `kornia/ONNX_models` (81 MB) and Depth Anything V2 small from `onnx-community`
  (27 MB, int8). Nothing is downloaded until the visitor presses **Load**.

The page shows a progress bar during the download. Nothing runs by itself: the visitor picks a sample
image and presses **Run on this frame** (the clips are not offered here, the large models are too slow
for video). `models.js` decodes the output. Local-feature models run twice, on the frame and on a copy warped by the
homepage's affine graph (a gentle one: up to 8 degrees, 5 % translation and 5 % scale, no shear), and the page draws the mutual-nearest-neighbour matches between the two; XFeat's
NMS, reliability scoring and descriptor sampling are a port of `detectAndCompute` whose numpy twin the
build checks against kornia (99 % identical keypoints), while KeyNet + HardNet is single-scale with a
40-px window around each peak resized to a 32x32 patch. Classification shows the top-5 ImageNet classes.
The other outputs: COCO boxes and labels
for RT-DETR, YuNet's priors, box decoding and NMS (a port of `kornia.contrib.FaceDetector`, because the
traced NMS loop does not generalise to other images), a colour map for relative depth, and the upscaled
image for super-resolution. Each page also carries the Python call, a Download ONNX link and a model card
(size, licence, paper, source). Left out, and why: DexiNed (140 MB graph, belongs on the hub first), DISK
(its keypoint selection does not export; XFeat covers the use case), LoFTR and DeDoDe (dense matchers,
too slow without threads), MobileSAM (1024-px encoder), SuperPoint + LightGlue (the fused ONNX is a
matcher only, and its GitHub release asset is not served with CORS headers), Real-ESRGAN (needs the
optional `basicsr` dependency). Two kornia issues surfaced: the dynamo exporter produces a wrong KeyNet
response map (the legacy exporter is used instead), and `DepthAnythingONNXBuilder.build` fails on its
preprocessor export.

## Pipelines

`pipelines/` lets a visitor chain operators: pick a container (`torch.nn.Sequential`,
`kornia.augmentation.ImageSequential` or `AugmentationSequential`), add steps from the catalog,
set their parameters, and run the chain on the sample images. Everything happens in the browser:

- Each step's exported graph is fetched, decoded with protobuf.js against `vendor/onnx.proto`,
  renamed with an `s<i>/` prefix and spliced to the previous step's output. Live parameters stay
  graph inputs, so one composed graph serves every slider position; "parameters baked" turns them
  into initializers at export time.
- The composed graph runs through onnxruntime-web for the preview and downloads as one `.onnx`.
  The Python block prints the equivalent container with the module form of every step.
- Eligible steps are live ONNX operators with a module form (or augmentations) and a single image
  input. Channel and size mismatches, and fixed-shape steps, are listed as warnings.
- Under the steps sits an optional **Model** dropdown (detection, faces, depth, classification,
  super-resolution; the two-pass feature models are excluded). The steps are the preprocessing; the model
  always runs after them. Its graph is fetched (from this site or the Hugging Face hub), its input
  normalisation (`scale`, `mean`, `std`) is spliced in as constant nodes, and the whole thing runs and
  downloads as one standalone ONNX graph. Graphs exported at opset 14 to 17 are lifted
  to 18 when composed (`Reduce*` axes become an input, `Split` gets `num_outputs`). Fixed-size models such
  as TinyViT warn until a `resize` to their size precedes them; the generated Python builds the model after
  the container and calls it on the container's output.
- Pipelines are stored in `localStorage`; "Copy share link" encodes one into the URL.

## Theme

`theme.js` (loaded in every page's `<head>`, before paint) applies the stored light/dark choice, or nothing,
in which case the CSS follows the system preference. The colours are tokens in `styles.css`; the dark theme
redefines them under `[data-theme="dark"]` and a `prefers-color-scheme` block. Code blocks and canvases
keep their own colours in both themes.

## Regenerating

Run from a checkout of kornia (the exported graphs are only as current as that checkout):

```bash
python -m venv .venv && .venv/bin/pip install -r playground/build/requirements.txt
.venv/bin/python playground/build/build.py              # everything (~10 min on CPU)
.venv/bin/python playground/build/build.py rotate hue   # only matching specs, merged into the registry
.venv/bin/python playground/build/build.py --pages-only # regenerate registry + pages from cached graphs
```

Then serve the repository root (`python -m http.server`) and open `/playground/`; `file://` will
not work because the page fetches `registry.json` and the graphs.

`build.py` refuses to publish a graph that fails `onnx.checker` or whose onnxruntime output
disagrees with eager PyTorch (also at a second batch and image size for dynamic graphs); such an
operator falls back to frames and the reason is printed. The onnxruntime-web version is pinned in
`index.html`, `build/op_template.html` and `ORT_WEB_VERSION` in `build.py`; bump all three together.

## Adding an operator

Append a `Spec` to `SPECS` in `build/specs.py`: the operator id, a one-line summary, the parameter
list, a lambda that calls kornia on the image (live parameters arrive as `(1,)` tensors, selects as
Python values), the call as a snippet template, optionally its module form and Rust call. Run the
build for that operator and open its page.

## Homepage demo

The homepage runs three things on the clips with `hero-demo.js`: YuNet face detection (the model from
`models/`, decoded with the helpers `models.js` exports), an affine augmentation, and Sobel edges. Random
augmentations draw their parameters inside the graph on every run, which would jitter a video, so
`build/home.py` exports `warp_affine` with angle, translation, scale and shear as graph inputs; the page
draws them once per pass of the clip from RandomAffine's default ranges and holds them until the clip
wraps or the visitor presses Re-roll.

## Sample clips

Three 4-second clips from Wikimedia Commons (public domain and CC0; authors and links in
`build/samples/videos/README.md`). `build/videos.py` cuts them to a 256 px centre crop with
ffmpeg; the source downloads are git-ignored. On a live operator or a pipeline the clip is drawn
frame by frame to the input canvas and the graph runs on every frame, as fast as it allows; the
status line shows the resulting frame rate. Frame-mode operators have no clip buttons.

## Sample images

From the [Kodak Lossless True Color Image Suite](https://r0k.us/graphics/kodak/), released for
unrestricted usage.
