"""Declarative specs for the operators the playground can run.

Every spec says how to call one kornia operator on an RGB image ``x`` of shape ``(B, 3, H, W)`` in
``[0, 1]``, which parameters a visitor can move, and how to print the call as Python (function
form, optional module form) and, where kornia-rs has the operator, as Rust. ``build.py`` turns each
spec into ONNX graphs (or, when export fails, frames) and lists every public operator of the image
packages, greyed out when no spec exists for it.

Parameter kinds:

- ``Live``: a float that becomes a graph input, so one graph serves every slider position.
- ``Select``: a discrete value (kernel size, bits, enum); each value is its own graph.

``call`` receives the image and one keyword per parameter: tensors of shape ``[1]`` for live
parameters, plain Python values for selects. Snippet templates use ``{img}`` and ``{param}``
placeholders. ``rust`` holds the kornia-rs call (``kornia-imgproc`` 0.1.14 signatures from
docs.rs); the page adds the shared prelude that loads ``img`` and allocates ``out``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

import kornia as K
from kornia.color import CFA


@dataclass
class Live:
    name: str
    min: float
    max: float
    step: float
    default: float
    label: str = ""

    kind: str = field(default="live", init=False)
    type: str = field(default="float", init=False)


@dataclass
class Select:
    name: str
    choices: list[Any]
    default: Any
    label: str = ""
    labels: list[str] | None = None  # display text per choice (enums)
    literals: dict[str, str] | None = None  # choice -> Python literal for the ``{name_literal}`` placeholder

    kind: str = field(default="select", init=False)

    @property
    def type(self) -> str:
        if all(isinstance(c, int) for c in self.choices):
            return "int"
        return "float" if all(isinstance(c, (int, float)) for c in self.choices) else "str"


@dataclass
class Rust:
    use: list[str]  # ``use`` paths
    call: list[str]  # statements after the prelude; ``out`` is allocated with ``channels`` channels
    channels: int = 3
    out_size: str = ""  # expression for the output ImageSize when it differs from the input
    note: str = ""


@dataclass
class Spec:
    id: str
    summary: str
    params: list[Live | Select]
    call: Callable[..., torch.Tensor]
    snippet: str
    module: str = ""  # the same call through the nn.Module form, if kornia has one
    rust: Rust | None = None
    stochastic: bool = False
    pre: str = ""  # human note about input conversion
    guidance: str = ""  # id of a fixed, non-selectable sample fed as a second image input (``guide=``)
    static_only: bool = False  # skip the dynamic-shape attempt (documented failure)
    frames_only: bool = False  # skip the export attempt (documented failure)
    frames_note: str = ""


DOCS = "https://kornia.readthedocs.io/en/latest"
PI = math.pi

KS = Select("kernel_size", [3, 5, 7, 9, 11, 15, 21], 5)
KS_SMALL = Select("kernel_size", [3, 5, 7, 9, 11], 5)


def _pair(v: torch.Tensor) -> torch.Tensor:
    """(1,) live value -> (1, 2) as the (y, x) pair many filters take."""
    return torch.stack([v, v], -1)


def _ones_kernel(k: int, like: torch.Tensor) -> torch.Tensor:
    return torch.ones(k, k, dtype=like.dtype, device=like.device)


def _like_image(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """(1,) live value -> (B, 1, H, W) so it broadcasts like a per-pixel weight."""
    return v.reshape(1, 1, 1, 1).expand(x.shape[0], 1, x.shape[-2], x.shape[-1])


_MODULES: dict[tuple, torch.nn.Module] = {}


def _module(key: tuple, factory: Callable[[], torch.nn.Module]) -> torch.nn.Module:
    """Construct an nn.Module once per parameter set; building one inside the traced call fails."""
    if key not in _MODULES:
        _MODULES[key] = factory().eval()
    return _MODULES[key]


# Predefined kernels for the operators that take a user kernel.
KERNELS_2D: dict[str, torch.Tensor] = {
    "sharpen": torch.tensor([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]),
    "emboss": torch.tensor([[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]),
    "edge": torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]),
    "sobel_x": torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
    "box3": torch.full((3, 3), 1.0 / 9),
    "gaussian5": torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]).outer(torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])) / 256,
}
KERNELS_1D: dict[str, torch.Tensor] = {
    "box3": torch.full((3,), 1.0 / 3),
    "gaussian5": torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16,
    "gaussian7": torch.tensor([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0]) / 64,
    "derivative": torch.tensor([-1.0, 0.0, 1.0]) / 2,
}


def _lits(kernels: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: str([[round(v, 4) for v in row] for row in k.tolist()] if k.dim() == 2 else [round(v, 4) for v in k.tolist()]) for name, k in kernels.items()}


def _affine_rust(matrix_expr: str) -> Rust:
    return Rust(
        ["kornia_imgproc::warp::{get_rotation_matrix2d, warp_affine}", "kornia_imgproc::interpolation::InterpolationMode"],
        [
            "let (w, h) = (img.size().width as f32, img.size().height as f32);",
            f"let m = {matrix_expr};",
            "warp_affine(&img, &mut out, &m, InterpolationMode::Bilinear)?;",
        ],
    )


SPECS: list[Spec] = [
    # ------------------------------------------------------------------------------ filters
    Spec(
        "kornia.filters.gaussian_blur2d",
        "Separable Gaussian filter. The kernel size is fixed per graph; sigma drives the graph live.",
        [KS, Live("sigma", 0.1, 8.0, 0.1, 1.5)],
        lambda x, kernel_size, sigma: K.filters.gaussian_blur2d(x, (kernel_size, kernel_size), _pair(sigma)),
        "kornia.filters.gaussian_blur2d({img}, ({kernel_size}, {kernel_size}), ({sigma}, {sigma}))",
        module="kornia.filters.GaussianBlur2d(({kernel_size}, {kernel_size}), ({sigma}, {sigma}))({img})",
        rust=Rust(["kornia_imgproc::filter::gaussian_blur"], ["gaussian_blur(&img, &mut out, ({kernel_size}, {kernel_size}), ({sigma}, {sigma}))?;"]),
    ),
    Spec(
        "kornia.filters.box_blur",
        "Mean filter over a square window.",
        [KS],
        lambda x, kernel_size: K.filters.box_blur(x, (kernel_size, kernel_size)),
        "kornia.filters.box_blur({img}, ({kernel_size}, {kernel_size}))",
        module="kornia.filters.BoxBlur(({kernel_size}, {kernel_size}))({img})",
        rust=Rust(["kornia_imgproc::filter::box_blur"], ["box_blur(&img, &mut out, ({kernel_size}, {kernel_size}))?;"]),
    ),
    Spec(
        "kornia.filters.median_blur",
        "Median over a square window; exports through a sort-based selection.",
        [KS_SMALL],
        lambda x, kernel_size: K.filters.median_blur(x, (kernel_size, kernel_size)),
        "kornia.filters.median_blur({img}, ({kernel_size}, {kernel_size}))",
        module="kornia.filters.MedianBlur(({kernel_size}, {kernel_size}))({img})",
    ),
    Spec(
        "kornia.filters.bilateral_blur",
        "Edge-preserving blur with live colour and spatial sigmas. The graph unrolls the window, so it is the slowest demo here.",
        [Select("kernel_size", [3, 5, 7], 5), Live("sigma_color", 0.01, 1.0, 0.01, 0.1), Live("sigma_space", 0.5, 5.0, 0.1, 1.5)],
        lambda x, kernel_size, sigma_color, sigma_space: K.filters.bilateral_blur(x, (kernel_size, kernel_size), sigma_color, _pair(sigma_space)),
        "kornia.filters.bilateral_blur({img}, ({kernel_size}, {kernel_size}), {sigma_color}, ({sigma_space}, {sigma_space}))",
        module="kornia.filters.BilateralBlur(({kernel_size}, {kernel_size}), {sigma_color}, ({sigma_space}, {sigma_space}))({img})",
    ),
    Spec(
        "kornia.filters.guided_blur",
        "Guided filter using the image as its own guidance; eps controls the edge preservation.",
        [KS_SMALL, Live("eps", 0.001, 0.5, 0.001, 0.01)],
        lambda x, kernel_size, eps: K.filters.guided_blur(x, x, (kernel_size, kernel_size), eps),
        "kornia.filters.guided_blur({img}, {img}, ({kernel_size}, {kernel_size}), {eps})",
        module="kornia.filters.GuidedBlur(({kernel_size}, {kernel_size}), {eps})({img}, {img})",
    ),
    Spec(
        "kornia.filters.motion_blur",
        "Directional blur along a line kernel; angle and direction are live.",
        [KS, Live("angle", 0.0, 180.0, 1.0, 45.0, "angle (deg)"), Live("direction", -1.0, 1.0, 0.1, 0.0)],
        lambda x, kernel_size, angle, direction: K.filters.motion_blur(x, kernel_size, angle, direction),
        "kornia.filters.motion_blur({img}, {kernel_size}, {angle}, {direction})",
        module="kornia.filters.MotionBlur({kernel_size}, {angle}, {direction})({img})",
    ),
    Spec(
        "kornia.filters.unsharp_mask",
        "Sharpening by subtracting a Gaussian blur; sigma is live.",
        [KS_SMALL, Live("sigma", 0.1, 5.0, 0.1, 1.0)],
        lambda x, kernel_size, sigma: K.filters.unsharp_mask(x, (kernel_size, kernel_size), _pair(sigma)),
        "kornia.filters.unsharp_mask({img}, ({kernel_size}, {kernel_size}), ({sigma}, {sigma}))",
        module="kornia.filters.UnsharpMask(({kernel_size}, {kernel_size}), ({sigma}, {sigma}))({img})",
    ),
    Spec(
        "kornia.filters.laplacian",
        "Laplacian of the image; the output is signed and shown normalised.",
        [KS_SMALL],
        lambda x, kernel_size: K.filters.laplacian(x, kernel_size),
        "kornia.filters.laplacian({img}, {kernel_size})",
        module="kornia.filters.Laplacian({kernel_size})({img})",
    ),
    Spec(
        "kornia.filters.sobel",
        "Sobel edge magnitude per channel.",
        [],
        lambda x: K.filters.sobel(x),
        "kornia.filters.sobel({img})",
        module="kornia.filters.Sobel()({img})",
        rust=Rust(["kornia_imgproc::filter::sobel"], ["sobel(&img, &mut out, 3)?;"]),
    ),
    Spec(
        "kornia.filters.blur_pool2d",
        "Anti-aliased downsampling (BlurPool): blur then stride-2 subsample. The output is half size.",
        [Select("kernel_size", [3, 5, 7], 3)],
        lambda x, kernel_size: K.filters.blur_pool2d(x, kernel_size),
        "kornia.filters.blur_pool2d({img}, {kernel_size})",
        module="kornia.filters.BlurPool2D({kernel_size})({img})",
    ),
    Spec(
        "kornia.filters.max_blur_pool2d",
        "Max pooling followed by BlurPool; the output is half size.",
        [Select("kernel_size", [3, 5, 7], 3)],
        lambda x, kernel_size: K.filters.max_blur_pool2d(x, kernel_size),
        "kornia.filters.max_blur_pool2d({img}, {kernel_size})",
        module="kornia.filters.MaxBlurPool2D({kernel_size})({img})",
    ),
    Spec(
        "kornia.filters.edge_aware_blur_pool2d",
        "BlurPool that keeps edges above a threshold sharp; the output is half size.",
        # edge_threshold is compared in Python (``> 0``), so it cannot be a live input
        [Select("kernel_size", [3, 5, 7], 3), Select("edge_threshold", [0.5, 1.25, 2.5], 1.25)],
        lambda x, kernel_size, edge_threshold: K.filters.edge_aware_blur_pool2d(x, kernel_size, float(edge_threshold)),
        "kornia.filters.edge_aware_blur_pool2d({img}, {kernel_size}, edge_threshold={edge_threshold})",
        module="kornia.filters.EdgeAwareBlurPool2D({kernel_size}, edge_threshold={edge_threshold})({img})",
    ),
    Spec(
        "kornia.filters.canny",
        "Canny edge detector. Its threshold check reads tensor values in Python, so it cannot export with live "
        "thresholds; the frames were rendered offline along the low threshold.",
        [Live("low_threshold", 0.02, 0.18, 0.01, 0.1), Select("high_threshold", [0.2], 0.2)],
        lambda x, low_threshold, high_threshold: K.filters.canny(x, low_threshold=float(low_threshold), high_threshold=float(high_threshold))[1],
        "kornia.filters.canny({img}, low_threshold={low_threshold}, high_threshold={high_threshold})[1]",
        module="kornia.filters.Canny(low_threshold={low_threshold}, high_threshold={high_threshold})({img})[1]",
        frames_only=True,
        frames_note="high_threshold is fixed at 0.2 in this demo",
    ),
    Spec(
        "kornia.filters.filter2d",
        "Convolution with a 2D kernel; pick one of the predefined kernels.",
        [Select("kernel", list(KERNELS_2D), "sharpen", literals=_lits(KERNELS_2D))],
        lambda x, kernel: K.filters.filter2d(x, KERNELS_2D[kernel].to(x)[None]),
        "kornia.filters.filter2d({img}, torch.tensor({kernel_literal})[None])",
    ),
    Spec(
        "kornia.filters.filter2d_separable",
        "Separable convolution: the same 1D kernel along x and y.",
        [Select("kernel", list(KERNELS_1D), "gaussian5", literals=_lits(KERNELS_1D))],
        lambda x, kernel: K.filters.filter2d_separable(x, KERNELS_1D[kernel].to(x)[None], KERNELS_1D[kernel].to(x)[None]),
        "kornia.filters.filter2d_separable({img}, torch.tensor({kernel_literal})[None], torch.tensor({kernel_literal})[None])",
    ),
    Spec(
        "kornia.filters.fft_conv",
        "The same 2D convolution computed through the FFT.",
        [Select("kernel", list(KERNELS_2D), "sharpen", literals=_lits(KERNELS_2D))],
        lambda x, kernel: K.filters.fft_conv(x, KERNELS_2D[kernel].to(x)[None]),
        "kornia.filters.fft_conv({img}, torch.tensor({kernel_literal})[None])",
    ),
    Spec(
        "kornia.filters.joint_bilateral_blur",
        "Bilateral blur whose range weights come from a separate guidance image (the fixed reference under the output).",
        [Select("kernel_size", [3, 5, 7], 5), Live("sigma_color", 0.01, 1.0, 0.01, 0.1), Live("sigma_space", 0.5, 5.0, 0.1, 1.5)],
        lambda x, guide, kernel_size, sigma_color, sigma_space: K.filters.joint_bilateral_blur(x, guide, (kernel_size, kernel_size), sigma_color, _pair(sigma_space)),
        "kornia.filters.joint_bilateral_blur({img}, guide, ({kernel_size}, {kernel_size}), {sigma_color}, ({sigma_space}, {sigma_space}))",
        module="kornia.filters.JointBilateralBlur(({kernel_size}, {kernel_size}), {sigma_color}, ({sigma_space}, {sigma_space}))({img}, guide)",
        guidance="barn",
    ),
    Spec(
        "kornia.filters.in_range",
        "Mask of the pixels whose every channel lies between two bounds.",
        [Live("lower", 0.0, 1.0, 0.01, 0.2), Live("upper", 0.0, 1.0, 0.01, 0.8)],
        lambda x, lower, upper: K.filters.in_range(
            x, lower.reshape(1, 1, 1, 1).expand(x.shape[0], 3, 1, 1), upper.reshape(1, 1, 1, 1).expand(x.shape[0], 3, 1, 1), return_mask=True
        ),
        "kornia.filters.in_range({img}, ({lower}, {lower}, {lower}), ({upper}, {upper}, {upper}), return_mask=True)",
        module="kornia.filters.InRange(({lower}, {lower}, {lower}), ({upper}, {upper}, {upper}), return_mask=True)({img})",
    ),
    # ------------------------------------------------------------------------------ color
    Spec(
        "kornia.color.rgb_to_grayscale", "Luma-weighted grayscale.", [], lambda x: K.color.rgb_to_grayscale(x),
        "kornia.color.rgb_to_grayscale({img})", module="kornia.color.RgbToGrayscale()({img})",
        rust=Rust(["kornia_imgproc::color::gray_from_rgb"], ["gray_from_rgb(&img, &mut out)?;"], channels=1),
    ),
    Spec("kornia.color.rgb_to_y", "The Y (luma) channel of YUV.", [], lambda x: K.color.rgb_to_y(x), "kornia.color.rgb_to_y({img})"),
    Spec(
        "kornia.color.rgb_to_bgr", "Swaps the red and blue channels.", [], lambda x: K.color.rgb_to_bgr(x),
        "kornia.color.rgb_to_bgr({img})", module="kornia.color.RgbToBgr()({img})",
        rust=Rust(["kornia_imgproc::color::bgr_from_rgb"], ["bgr_from_rgb(&img, &mut out)?;"]),
    ),
    Spec(
        "kornia.color.bgr_to_rgb", "Swaps the blue and red channels.", [], lambda x: K.color.bgr_to_rgb(x),
        "kornia.color.bgr_to_rgb({img})", module="kornia.color.BgrToRgb()({img})",
        rust=Rust(["kornia_imgproc::color::bgr_from_rgb"], ["bgr_from_rgb(&img, &mut out)?;  // the same channel swap"]),
    ),
    Spec(
        "kornia.color.rgb_to_hsv", "RGB to HSV; hue is in radians, so the output is shown normalised.", [], lambda x: K.color.rgb_to_hsv(x),
        "kornia.color.rgb_to_hsv({img})", module="kornia.color.RgbToHsv()({img})",
        rust=Rust(["kornia_imgproc::color::hsv_from_rgb"], ["hsv_from_rgb(&img, &mut out)?;  // H, S, V in [0, 255], unlike kornia's radians"]),
    ),
    Spec("kornia.color.rgb_to_hls", "RGB to HLS; shown normalised.", [], lambda x: K.color.rgb_to_hls(x), "kornia.color.rgb_to_hls({img})", module="kornia.color.RgbToHls()({img})"),
    Spec("kornia.color.rgb_to_lab", "RGB to CIE Lab; L is in [0, 100] so the output is shown normalised.", [], lambda x: K.color.rgb_to_lab(x), "kornia.color.rgb_to_lab({img})", module="kornia.color.RgbToLab()({img})"),
    Spec("kornia.color.rgb_to_luv", "RGB to CIE Luv; shown normalised.", [], lambda x: K.color.rgb_to_luv(x), "kornia.color.rgb_to_luv({img})", module="kornia.color.RgbToLuv()({img})"),
    Spec("kornia.color.rgb_to_xyz", "RGB to CIE XYZ.", [], lambda x: K.color.rgb_to_xyz(x), "kornia.color.rgb_to_xyz({img})", module="kornia.color.RgbToXyz()({img})"),
    Spec("kornia.color.rgb_to_ycbcr", "RGB to YCbCr.", [], lambda x: K.color.rgb_to_ycbcr(x), "kornia.color.rgb_to_ycbcr({img})", module="kornia.color.RgbToYcbcr()({img})"),
    Spec("kornia.color.rgb_to_yuv", "RGB to YUV; U and V are signed, shown normalised.", [], lambda x: K.color.rgb_to_yuv(x), "kornia.color.rgb_to_yuv({img})", module="kornia.color.RgbToYuv()({img})"),
    Spec("kornia.color.rgb_to_linear_rgb", "Removes the sRGB gamma curve.", [], lambda x: K.color.rgb_to_linear_rgb(x), "kornia.color.rgb_to_linear_rgb({img})", module="kornia.color.RgbToLinearRgb()({img})"),
    Spec("kornia.color.linear_rgb_to_rgb", "Applies the sRGB gamma curve.", [], lambda x: K.color.linear_rgb_to_rgb(x), "kornia.color.linear_rgb_to_rgb({img})", module="kornia.color.LinearRgbToRgb()({img})"),
    Spec("kornia.color.sepia", "Sepia tone.", [], lambda x: K.color.sepia(x), "kornia.color.sepia({img})", module="kornia.color.Sepia()({img})"),
    Spec(
        "kornia.color.rgb_to_rgba",
        "Adds a constant alpha channel; only RGB is displayed.",
        [Live("alpha_val", 0.0, 1.0, 0.05, 1.0)],
        lambda x, alpha_val: K.color.rgb_to_rgba(x, _like_image(alpha_val, x)),
        "kornia.color.rgb_to_rgba({img}, {alpha_val})",
        module="kornia.color.RgbToRgba({alpha_val})({img})",
    ),
    Spec(
        "kornia.color.rgb_to_raw",
        "Simulates a Bayer mosaic with the given colour filter array.",
        [Select("cfa", ["BG", "GB", "RG", "GR"], "BG")],
        lambda x, cfa: K.color.rgb_to_raw(x, getattr(CFA, cfa)),
        "kornia.color.rgb_to_raw({img}, kornia.color.CFA.{cfa})",
        module="kornia.color.RgbToRaw(kornia.color.CFA.{cfa})({img})",
    ),
    Spec(
        "kornia.color.grayscale_to_rgb",
        "Repeats a single channel three times.",
        [],
        lambda x: K.color.grayscale_to_rgb(K.color.rgb_to_grayscale(x)),
        "kornia.color.grayscale_to_rgb(kornia.color.rgb_to_grayscale({img}))",
        module="kornia.color.GrayscaleToRgb()(kornia.color.rgb_to_grayscale({img}))",
        rust=Rust(
            ["kornia_imgproc::color::{gray_from_rgb, rgb_from_gray}"],
            ["let mut gray = Image::<f32, 1, _>::from_size_val(img.size(), 0.0, CpuAllocator)?;", "gray_from_rgb(&img, &mut gray)?;", "rgb_from_gray(&gray, &mut out)?;"],
        ),
        pre="the sample is converted to grayscale first",
    ),
    Spec(
        "kornia.color.apply_colormap",
        "Maps a grayscale image through a colour map.",
        [Select("colormap", ["viridis", "plasma", "cividis", "turbo", "jet", "hot", "bone", "ocean", "seismic", "twilight"], "viridis")],
        lambda x, colormap: K.color.apply_colormap(K.color.rgb_to_grayscale(x), K.color.ColorMap(base=getattr(K.color.ColorMapType, colormap))),
        "kornia.color.apply_colormap(kornia.color.rgb_to_grayscale({img}), kornia.color.ColorMap(base=kornia.color.ColorMapType.{colormap}))",
        module="kornia.color.ApplyColorMap(kornia.color.ColorMap(base=kornia.color.ColorMapType.{colormap}))(kornia.color.rgb_to_grayscale({img}))",
        pre="the sample is converted to grayscale first",
    ),
    # ------------------------------------------------------------------------------ enhance
    Spec("kornia.enhance.adjust_brightness", "Adds a constant to every pixel.", [Live("factor", -0.5, 0.5, 0.01, 0.2)], lambda x, factor: K.enhance.adjust_brightness(x, factor), "kornia.enhance.adjust_brightness({img}, {factor})", module="kornia.enhance.AdjustBrightness({factor})({img})"),
    Spec(
        "kornia.enhance.adjust_brightness_accumulative",
        "Brightness scaled by the factor, accumulating over repeated calls as in PIL.",
        [Live("factor", 0.0, 2.0, 0.01, 1.3)],
        lambda x, factor: K.enhance.adjust_brightness_accumulative(x, factor),
        "kornia.enhance.adjust_brightness_accumulative({img}, {factor})",
        module="kornia.enhance.AdjustBrightnessAccumulative({factor})({img})",
    ),
    Spec("kornia.enhance.adjust_contrast", "Multiplies every pixel by a factor.", [Live("factor", 0.0, 2.0, 0.01, 1.3)], lambda x, factor: K.enhance.adjust_contrast(x, factor), "kornia.enhance.adjust_contrast({img}, {factor})", module="kornia.enhance.AdjustContrast({factor})({img})"),
    Spec("kornia.enhance.adjust_contrast_with_mean_subtraction", "Contrast around the image mean, as in PIL.", [Live("factor", 0.0, 2.0, 0.01, 1.3)], lambda x, factor: K.enhance.adjust_contrast_with_mean_subtraction(x, factor), "kornia.enhance.adjust_contrast_with_mean_subtraction({img}, {factor})", module="kornia.enhance.AdjustContrastWithMeanSubtraction({factor})({img})"),
    Spec("kornia.enhance.adjust_gamma", "Power-law correction with a gain.", [Live("gamma", 0.2, 3.0, 0.05, 1.0), Live("gain", 0.5, 2.0, 0.05, 1.0)], lambda x, gamma, gain: K.enhance.adjust_gamma(x, gamma, gain), "kornia.enhance.adjust_gamma({img}, {gamma}, {gain})", module="kornia.enhance.AdjustGamma({gamma}, {gain})({img})"),
    Spec("kornia.enhance.adjust_hue", "Rotates every pixel's hue by the given angle in radians.", [Live("factor", -PI, PI, 0.01, 0.5, "factor (rad)")], lambda x, factor: K.enhance.adjust_hue(x, factor), "kornia.enhance.adjust_hue({img}, {factor})", module="kornia.enhance.AdjustHue({factor})({img})"),
    Spec("kornia.enhance.adjust_saturation", "Scales the HSV saturation.", [Live("factor", 0.0, 3.0, 0.05, 1.5)], lambda x, factor: K.enhance.adjust_saturation(x, factor), "kornia.enhance.adjust_saturation({img}, {factor})", module="kornia.enhance.AdjustSaturation({factor})({img})"),
    Spec("kornia.enhance.adjust_saturation_with_gray_subtraction", "Saturation by blending with the grayscale image, as in PIL.", [Live("factor", 0.0, 3.0, 0.05, 1.5)], lambda x, factor: K.enhance.adjust_saturation_with_gray_subtraction(x, factor), "kornia.enhance.adjust_saturation_with_gray_subtraction({img}, {factor})", module="kornia.enhance.AdjustSaturationWithGraySubtraction({factor})({img})"),
    Spec("kornia.enhance.adjust_log", "Logarithmic tone mapping.", [Live("gain", 0.1, 3.0, 0.05, 1.0)], lambda x, gain: K.enhance.adjust_log(x, gain), "kornia.enhance.adjust_log({img}, {gain})", module="kornia.enhance.AdjustLog({gain})({img})"),
    Spec("kornia.enhance.adjust_sigmoid", "Sigmoid contrast curve with a cutoff and a gain.", [Live("cutoff", 0.0, 1.0, 0.01, 0.5), Live("gain", 1.0, 20.0, 0.5, 10.0)], lambda x, cutoff, gain: K.enhance.adjust_sigmoid(x, cutoff, gain), "kornia.enhance.adjust_sigmoid({img}, {cutoff}, {gain})", module="kornia.enhance.AdjustSigmoid({cutoff}, {gain})({img})"),
    Spec("kornia.enhance.sharpness", "Sharpness blend, as in PIL.", [Live("factor", 0.0, 3.0, 0.05, 1.5)], lambda x, factor: K.enhance.sharpness(x, factor), "kornia.enhance.sharpness({img}, {factor})"),
    Spec("kornia.enhance.solarize", "Inverts every pixel above the threshold.", [Live("thresholds", 0.0, 1.0, 0.01, 0.5)], lambda x, thresholds: K.enhance.solarize(x, thresholds), "kornia.enhance.solarize({img}, {thresholds})"),
    Spec("kornia.enhance.posterize", "Keeps only the given number of bits per channel.", [Select("bits", [1, 2, 3, 4, 5, 6, 7, 8], 3)], lambda x, bits: K.enhance.posterize(x, bits), "kornia.enhance.posterize({img}, {bits})"),
    Spec("kornia.enhance.invert", "Inverts the image.", [], lambda x: K.enhance.invert(x), "kornia.enhance.invert({img})", module="kornia.enhance.Invert()({img})"),
    Spec("kornia.enhance.equalize", "Histogram equalisation per channel.", [], lambda x: K.enhance.equalize(x), "kornia.enhance.equalize({img})"),
    Spec(
        "kornia.enhance.equalize_clahe",
        "Contrast-limited adaptive histogram equalisation.",
        [Select("clip_limit", [2, 5, 10, 40], 10)],
        lambda x, clip_limit: K.enhance.equalize_clahe(x, clip_limit=float(clip_limit)),
        "kornia.enhance.equalize_clahe({img}, clip_limit={clip_limit})",
    ),
    Spec(
        "kornia.enhance.normalize_min_max", "Stretches each image to [0, 1].", [], lambda x: K.enhance.normalize_min_max(x),
        "kornia.enhance.normalize_min_max({img})",
        rust=Rust(["kornia_imgproc::normalize::normalize_min_max"], ["normalize_min_max(&img, &mut out, 0.0, 1.0)?;"]),
    ),
    Spec(
        "kornia.enhance.normalize",
        "Subtracts a mean and divides by a standard deviation; shown normalised.",
        [Live("mean", 0.0, 1.0, 0.01, 0.45), Live("std", 0.05, 1.0, 0.01, 0.25)],
        lambda x, mean, std: K.enhance.normalize(x, mean.expand(3), std.expand(3)),
        "kornia.enhance.normalize({img}, torch.tensor([{mean}, {mean}, {mean}]), torch.tensor([{std}, {std}, {std}]))",
        module="kornia.enhance.Normalize(torch.tensor([{mean}, {mean}, {mean}]), torch.tensor([{std}, {std}, {std}]))({img})",
    ),
    Spec(
        "kornia.enhance.threshold",
        "Binary threshold; the threshold is live.",
        [Live("thresh", 0.0, 1.0, 0.01, 0.5)],
        lambda x, thresh: K.enhance.threshold(x, thresh, maxval=1.0),
        "kornia.enhance.threshold({img}, {thresh}, maxval=1.0)",
        module="kornia.enhance.Threshold({thresh}, maxval=1.0)({img})",
        rust=Rust(["kornia_imgproc::threshold::threshold_binary"], ["threshold_binary(&img, &mut out, {thresh}, 1.0)?;"]),
    ),
    Spec(
        "kornia.enhance.shift_rgb",
        "Adds a per-channel offset.",
        [Live("r_shift", -0.5, 0.5, 0.01, 0.1), Live("g_shift", -0.5, 0.5, 0.01, 0.0), Live("b_shift", -0.5, 0.5, 0.01, -0.1)],
        lambda x, r_shift, g_shift, b_shift: K.enhance.shift_rgb(x, r_shift, g_shift, b_shift),
        "kornia.enhance.shift_rgb({img}, {r_shift}, {g_shift}, {b_shift})",
    ),
    Spec(
        "kornia.enhance.add_weighted",
        "Weighted sum of the image and its horizontal mirror.",
        [Live("alpha", 0.0, 1.0, 0.05, 0.6), Live("beta", 0.0, 1.0, 0.05, 0.4), Live("gamma", -0.5, 0.5, 0.05, 0.0)],
        lambda x, alpha, beta, gamma: K.enhance.add_weighted(
            x, _like_image(alpha, x).expand_as(x), K.geometry.transform.hflip(x), _like_image(beta, x).expand_as(x), _like_image(gamma, x).expand_as(x)
        ),
        "kornia.enhance.add_weighted({img}, {alpha}, kornia.geometry.transform.hflip({img}), {beta}, {gamma})",
        module="kornia.enhance.AddWeighted({alpha}, {beta}, {gamma})({img}, kornia.geometry.transform.hflip({img}))",
        rust=Rust(
            ["kornia_imgproc::enhance::add_weighted", "kornia_imgproc::flip::horizontal_flip"],
            ["let mut mirror = Image::<f32, 3, _>::from_size_val(img.size(), 0.0, CpuAllocator)?;", "horizontal_flip(&img, &mut mirror)?;", "add_weighted(&img, {alpha}, &mirror, {beta}, {gamma}, &mut out)?;"],
        ),
    ),
    Spec(
        "kornia.enhance.jpeg_codec_differentiable",
        "Differentiable JPEG encode/decode at the given quality.",
        [Live("jpeg_quality", 1.0, 100.0, 1.0, 20.0)],
        lambda x, jpeg_quality: K.enhance.jpeg_codec_differentiable(x, jpeg_quality),
        "kornia.enhance.jpeg_codec_differentiable({img}, torch.tensor([{jpeg_quality}]))",
        module="kornia.enhance.JPEGCodecDifferentiable()({img}, torch.tensor([{jpeg_quality}]))",
    ),
    Spec("kornia.enhance.integral_image", "Summed-area table; shown normalised.", [], lambda x: K.enhance.integral_image(x), "kornia.enhance.integral_image({img})", module="kornia.enhance.IntegralImage()({img})"),
    # ------------------------------------------------------------------------------ morphology
    *[
        Spec(
            f"kornia.morphology.{name}",
            f"Morphological {name.replace('_', ' ')} with a square structuring element.",
            [KS_SMALL],
            (lambda name: lambda x, kernel_size: getattr(K.morphology, name)(x, _ones_kernel(kernel_size, x)))(name),
            f"kornia.morphology.{name}({{img}}, torch.ones({{kernel_size}}, {{kernel_size}}))",
        )
        for name in ("dilation", "erosion", "opening", "closing", "gradient", "top_hat", "bottom_hat")
    ],
    # ------------------------------------------------------------------------------ geometry.transform
    Spec(
        "kornia.geometry.transform.hflip", "Horizontal flip.", [], lambda x: K.geometry.transform.hflip(x),
        "kornia.geometry.transform.hflip({img})", module="kornia.geometry.transform.Hflip()({img})",
        rust=Rust(["kornia_imgproc::flip::horizontal_flip"], ["horizontal_flip(&img, &mut out)?;"]),
    ),
    Spec(
        "kornia.geometry.transform.vflip", "Vertical flip.", [], lambda x: K.geometry.transform.vflip(x),
        "kornia.geometry.transform.vflip({img})", module="kornia.geometry.transform.Vflip()({img})",
        rust=Rust(["kornia_imgproc::flip::vertical_flip"], ["vertical_flip(&img, &mut out)?;"]),
    ),
    Spec("kornia.geometry.transform.rot180", "Rotation by 180 degrees.", [], lambda x: K.geometry.transform.rot180(x), "kornia.geometry.transform.rot180({img})", module="kornia.geometry.transform.Rot180()({img})"),
    Spec(
        "kornia.geometry.transform.rotate",
        "Rotation about the image centre with bilinear sampling; the angle is live.",
        [Live("angle", -180.0, 180.0, 1.0, 30.0, "angle (deg)")],
        lambda x, angle: K.geometry.transform.rotate(x, angle),
        "kornia.geometry.transform.rotate({img}, torch.tensor([{angle}]))",
        module="kornia.geometry.transform.Rotate(torch.tensor([{angle}]))({img})",
        rust=_affine_rust("get_rotation_matrix2d((w / 2.0, h / 2.0), {angle}, 1.0)"),
    ),
    Spec(
        "kornia.geometry.transform.translate",
        "Translation in pixels.",
        [Live("tx", -100.0, 100.0, 1.0, 20.0, "tx (px)"), Live("ty", -100.0, 100.0, 1.0, -10.0, "ty (px)")],
        lambda x, tx, ty: K.geometry.transform.translate(x, torch.stack([tx, ty], -1)),
        "kornia.geometry.transform.translate({img}, torch.tensor([[{tx}, {ty}]]))",
        module="kornia.geometry.transform.Translate(torch.tensor([[{tx}, {ty}]]))({img})",
        rust=Rust(
            ["kornia_imgproc::warp::warp_affine", "kornia_imgproc::interpolation::InterpolationMode"],
            ["let m = [1.0, 0.0, {tx}, 0.0, 1.0, {ty}];", "warp_affine(&img, &mut out, &m, InterpolationMode::Bilinear)?;"],
        ),
    ),
    Spec(
        "kornia.geometry.transform.scale",
        "Isotropic scale about the centre.",
        [Live("scale_factor", 0.25, 3.0, 0.05, 1.5)],
        lambda x, scale_factor: K.geometry.transform.scale(x, scale_factor),
        "kornia.geometry.transform.scale({img}, torch.tensor([{scale_factor}]))",
        module="kornia.geometry.transform.Scale(torch.tensor([{scale_factor}]))({img})",
        rust=_affine_rust("get_rotation_matrix2d((w / 2.0, h / 2.0), 0.0, {scale_factor})"),
    ),
    Spec(
        "kornia.geometry.transform.shear",
        "Shear along x and y.",
        [Live("sx", -1.0, 1.0, 0.02, 0.3), Live("sy", -1.0, 1.0, 0.02, 0.0)],
        lambda x, sx, sy: K.geometry.transform.shear(x, torch.stack([sx, sy], -1)),
        "kornia.geometry.transform.shear({img}, torch.tensor([[{sx}, {sy}]]))",
        module="kornia.geometry.transform.Shear(torch.tensor([[{sx}, {sy}]]))({img})",
    ),
    Spec(
        "kornia.geometry.transform.resize",
        "Resizes the image to a square of the given side.",
        [Select("size", [64, 128, 192, 224, 256, 384, 512], 128)],
        lambda x, size: K.geometry.transform.resize(x, (size, size)),
        "kornia.geometry.transform.resize({img}, ({size}, {size}))",
        module="kornia.geometry.transform.Resize(({size}, {size}))({img})",
        rust=Rust(
            ["kornia_imgproc::resize::resize_native", "kornia_imgproc::interpolation::InterpolationMode"],
            ["resize_native(&img, &mut out, InterpolationMode::Bilinear)?;"],
            out_size="ImageSize { width: {size}, height: {size} }",
        ),
    ),
    Spec(
        "kornia.geometry.transform.rescale",
        "Rescales the image by a factor.",
        [Select("factor", [0.25, 0.5, 0.75, 1.5, 2.0], 0.5)],
        lambda x, factor: K.geometry.transform.rescale(x, float(factor)),
        "kornia.geometry.transform.rescale({img}, {factor})",
        module="kornia.geometry.transform.Rescale({factor})({img})",
    ),
    Spec(
        "kornia.geometry.transform.center_crop",
        "Crops a centred square of the given side.",
        [Select("size", [64, 96, 128, 160, 192, 224], 128)],
        lambda x, size: K.geometry.transform.center_crop(x, (size, size)),
        "kornia.geometry.transform.center_crop({img}, ({size}, {size}))",
        module="kornia.geometry.transform.CenterCrop2D(({size}, {size}))({img})",
    ),
    Spec(
        "kornia.geometry.transform.crop_by_indices",
        "Crops a square window given by its corner coordinates; the corner is live, the window size is fixed per graph.",
        [Select("size", [64, 96, 128], 96), Live("x0", 0.0, 160.0, 1.0, 40.0, "x0 (px)"), Live("y0", 0.0, 160.0, 1.0, 20.0, "y0 (px)")],
        lambda x, size, x0, y0: K.geometry.transform.crop_by_indices(
            x,
            torch.stack([torch.stack([x0, y0], -1), torch.stack([x0 + (size - 1), y0], -1), torch.stack([x0 + (size - 1), y0 + (size - 1)], -1), torch.stack([x0, y0 + (size - 1)], -1)], 1),
            size=(size, size),
        ),
        "kornia.geometry.transform.crop_by_indices({img}, torch.tensor([[[{x0}, {y0}], [{x0} + {size} - 1, {y0}], [{x0} + {size} - 1, {y0} + {size} - 1], [{x0}, {y0} + {size} - 1]]]), size=({size}, {size}))",
    ),
    Spec("kornia.geometry.transform.pyrdown", "Gaussian blur then 2x downsample.", [], lambda x: K.geometry.transform.pyrdown(x), "kornia.geometry.transform.pyrdown({img})", module="kornia.geometry.transform.PyrDown()({img})"),
    Spec("kornia.geometry.transform.pyrup", "2x upsample then Gaussian blur.", [], lambda x: K.geometry.transform.pyrup(x), "kornia.geometry.transform.pyrup({img})", module="kornia.geometry.transform.PyrUp()({img})"),
    Spec(
        "kornia.geometry.transform.elastic_transform2d",
        "Elastic deformation driven by smoothed random noise sampled inside the graph.",
        [Select("kernel_size", [33, 63], 63), Live("alpha", 0.0, 3.0, 0.1, 1.0)],
        lambda x, kernel_size, alpha: K.geometry.transform.elastic_transform2d(
            x, torch.rand(x.shape[0], 2, x.shape[-2], x.shape[-1], dtype=x.dtype, device=x.device) * 2 - 1,
            kernel_size=(kernel_size, kernel_size), alpha=torch.cat([alpha, alpha]),
        ),
        "kornia.geometry.transform.elastic_transform2d({img}, torch.rand(1, 2, 256, 256) * 2 - 1, kernel_size=({kernel_size}, {kernel_size}), alpha=({alpha}, {alpha}))",
        stochastic=True,
    ),
    # ------------------------------------------------------------------------------ augmentation
]


# ------------------------------------------------------------------------------ augmentation
# Every augmentation is built once per parameter choice (``_module``) with p=1.0 unless the effect
# itself is the coin flip, and its sampling stays inside the graph: "Re-roll" draws new parameters.
# ``AUGS`` rows: class name, summary, the one Select the visitor gets, factory(value) -> module, and
# the constructor arguments as they appear in the snippet ({v} is the chosen value).
# ``AUGS`` rows: class name, summary, the Selects the visitor gets (one graph per combination, so
# keep the product small for the 0.5-2 MB geometric graphs), factory(*values in Select order) ->
# module, and the constructor arguments as they appear in the snippet ({name}, {name_one_minus},
# {name_one_plus} and {name_plus_200} placeholders).
A = K.augmentation
AUGS: list[tuple[str, str, list[Select], Callable[..., torch.nn.Module], str]] = [
    ("CenterCrop", "Centre crop to a fixed square.", [Select("size", [96, 128, 192], 128)], lambda v: A.CenterCrop(v, p=1.0), "{size}, p=1.0"),
    ("ColorJiggle", "Random brightness, contrast, saturation and hue in one go, applied in a fixed order.", [Select("strength", [0.1, 0.3, 0.5], 0.3)], lambda v: A.ColorJiggle(v, v, v, min(v, 0.5), p=1.0), "{strength}, {strength}, {strength}, {strength}, p=1.0"),
    ("ColorJitter", "Like ColorJiggle but the four adjustments are applied in a random order, which cannot be traced.", [Select("strength", [0.1, 0.3, 0.5], 0.3)], lambda v: A.ColorJitter(v, v, v, min(v, 0.5), p=1.0), "{strength}, {strength}, {strength}, {strength}, p=1.0"),
    ("Denormalize", "Undoes a mean/std normalisation; shown normalised.", [Select("mean", [0.5, 0.45], 0.5)], lambda v: A.Denormalize(torch.tensor([v] * 3), torch.tensor([0.25] * 3), p=1.0), "torch.tensor([{mean}, {mean}, {mean}]), torch.tensor([0.25, 0.25, 0.25]), p=1.0"),
    ("Normalize", "Mean/std normalisation; shown normalised.", [Select("mean", [0.5, 0.45], 0.5)], lambda v: A.Normalize(torch.tensor([v] * 3), torch.tensor([0.25] * 3), p=1.0), "torch.tensor([{mean}, {mean}, {mean}]), torch.tensor([0.25, 0.25, 0.25]), p=1.0"),
    ("LongestMaxSize", "Rescales so the longer side equals max_size.", [Select("max_size", [128, 192, 256], 128)], lambda v: A.LongestMaxSize(v, p=1.0), "{max_size}, p=1.0"),
    ("SmallestMaxSize", "Rescales so the shorter side equals max_size.", [Select("max_size", [128, 192, 256], 128)], lambda v: A.SmallestMaxSize(v, p=1.0), "{max_size}, p=1.0"),
    ("PadTo", "Pads to a fixed size.", [Select("size", [288, 320, 384], 320)], lambda v: A.PadTo((v, v)), "({size}, {size})"),
    ("Resize", "Resizes to a fixed square.", [Select("size", [128, 192, 256], 128)], lambda v: A.Resize((v, v), p=1.0), "({size}, {size}), p=1.0"),
    ("RandomAffine", "Random rotation, translation, scale and shear.",
     [Select("degrees", [15, 30, 90], 30), Select("translate", [0.0, 0.2], 0.2), Select("scale_jitter", [0.0, 0.3], 0.3)],
     lambda d, t, sj: A.RandomAffine(degrees=float(d), translate=t, scale=(1 - sj, 1 + sj), shear=10.0, p=1.0),
     "degrees={degrees}, translate={translate}, scale=({scale_jitter_one_minus}, {scale_jitter_one_plus}), shear=10.0, p=1.0"),
    ("RandomAutoContrast", "Stretches each channel to full range (no randomness besides p).", [Select("p", [1.0], 1.0)], lambda v: A.RandomAutoContrast(p=1.0), "p=1.0"),
    ("RandomBoxBlur", "Box blur with a fixed kernel.", [Select("kernel_size", [3, 5, 7], 5)], lambda v: A.RandomBoxBlur((v, v), p=1.0), "({kernel_size}, {kernel_size}), p=1.0"),
    ("RandomBrightness", "Random brightness factor in [1-s, 1+s].", [Select("strength", [0.1, 0.3, 0.5], 0.3)], lambda v: A.RandomBrightness((1 - v, 1 + v), p=1.0), "({strength_one_minus}, {strength_one_plus}), p=1.0"),
    ("RandomChannelDropout", "Zeroes random channels.", [Select("num_drop_channels", [1, 2], 1), Select("fill_value", [0.0, 0.5, 1.0], 0.0)], lambda n, f: A.RandomChannelDropout(n, fill_value=f, p=1.0), "{num_drop_channels}, fill_value={fill_value}, p=1.0"),
    ("RandomChannelShuffle", "Permutes the colour channels.", [Select("p", [1.0], 1.0)], lambda v: A.RandomChannelShuffle(p=1.0), "p=1.0"),
    ("RandomContrast", "Random contrast factor in [1-s, 1+s].", [Select("strength", [0.1, 0.3, 0.5], 0.3)], lambda v: A.RandomContrast((1 - v, 1 + v), p=1.0), "({strength_one_minus}, {strength_one_plus}), p=1.0"),
    ("RandomCrop", "Random square crop.", [Select("size", [96, 128, 192], 128)], lambda v: A.RandomCrop((v, v), p=1.0), "({size}, {size}), p=1.0"),
    ("RandomElasticTransform", "Elastic deformation with random displacement noise.", [Select("alpha", [0.5, 1.0, 2.0], 1.0), Select("sigma", [16.0, 32.0], 32.0)], lambda a, sg: A.RandomElasticTransform(sigma=(sg, sg), alpha=(a, a), p=1.0), "sigma=({sigma}, {sigma}), alpha=({alpha}, {alpha}), p=1.0"),
    ("RandomEqualize", "Histogram equalisation.", [Select("p", [1.0], 1.0)], lambda v: A.RandomEqualize(p=1.0), "p=1.0"),
    ("RandomErasing", "Erases a random rectangle.", [Select("max_scale", [0.1, 0.33, 0.5], 0.33), Select("value", [0.0, 0.5, 1.0], 0.0)], lambda m, val: A.RandomErasing(scale=(0.02, m), value=val, p=1.0), "scale=(0.02, {max_scale}), value={value}, p=1.0"),
    ("RandomFisheye", "Random fisheye distortion.", [Select("max_gamma", [0.5, 1.0, 2.0], 1.0)], lambda v: A.RandomFisheye(torch.tensor([-0.3, 0.3]), torch.tensor([-0.3, 0.3]), torch.tensor([0.9, v]), p=1.0), "torch.tensor([-0.3, 0.3]), torch.tensor([-0.3, 0.3]), torch.tensor([0.9, {max_gamma}]), p=1.0"),
    ("RandomGamma", "Random gamma in [1-s, 1+s] with a random gain.", [Select("strength", [0.2, 0.5, 0.8], 0.5), Select("gain_jitter", [0.0, 0.3], 0.0)], lambda v, g: A.RandomGamma((1 - v, 1 + v), (1 - g, 1 + g), p=1.0), "({strength_one_minus}, {strength_one_plus}), ({gain_jitter_one_minus}, {gain_jitter_one_plus}), p=1.0"),
    ("RandomGaussianBlur", "Gaussian blur with a random sigma.", [Select("kernel_size", [3, 5, 9], 5), Select("max_sigma", [1.0, 2.0, 4.0], 2.0)], lambda k, sg: A.RandomGaussianBlur((k, k), (0.1, sg), p=1.0), "({kernel_size}, {kernel_size}), (0.1, {max_sigma}), p=1.0"),
    ("RandomGaussianIllumination", "Adds a random Gaussian light spot.", [Select("max_gain", [0.15, 0.3, 0.5], 0.3)], lambda v: A.RandomGaussianIllumination(gain=(0.01, v), p=1.0), "gain=(0.01, {max_gain}), p=1.0"),
    ("RandomGaussianNoise", "Adds Gaussian noise.", [Select("mean", [-0.1, 0.0, 0.1], 0.0), Select("std", [0.05, 0.1, 0.2], 0.1)], lambda m, sd: A.RandomGaussianNoise(mean=m, std=sd, p=1.0), "mean={mean}, std={std}, p=1.0"),
    ("RandomGrayscale", "Converts to grayscale with probability p.", [Select("p", [0.5], 0.5)], lambda v: A.RandomGrayscale(p=v), "p={p}"),
    ("RandomHorizontalFlip", "Horizontal flip with probability p.", [Select("p", [0.5], 0.5)], lambda v: A.RandomHorizontalFlip(p=v), "p={p}"),
    ("RandomVerticalFlip", "Vertical flip with probability p.", [Select("p", [0.5], 0.5)], lambda v: A.RandomVerticalFlip(p=v), "p={p}"),
    ("RandomHue", "Random hue shift in [-h, h].", [Select("hue", [0.1, 0.3, 0.5], 0.3)], lambda v: A.RandomHue((-v, v), p=1.0), "(-{hue}, {hue}), p=1.0"),
    ("RandomInvert", "Inverts with probability p.", [Select("p", [0.5], 0.5)], lambda v: A.RandomInvert(p=v), "p={p}"),
    ("RandomLinearCornerIllumination", "Linear light gradient from a random corner.", [Select("max_gain", [0.1, 0.2, 0.4], 0.2)], lambda v: A.RandomLinearCornerIllumination(gain=(0.01, v), p=1.0), "gain=(0.01, {max_gain}), p=1.0"),
    ("RandomLinearIllumination", "Linear light gradient in a random direction.", [Select("max_gain", [0.1, 0.2, 0.4], 0.2)], lambda v: A.RandomLinearIllumination(gain=(0.01, v), p=1.0), "gain=(0.01, {max_gain}), p=1.0"),
    ("RandomMedianBlur", "Median blur with a fixed kernel.", [Select("kernel_size", [3, 5, 7], 5)], lambda v: A.RandomMedianBlur((v, v), p=1.0), "({kernel_size}, {kernel_size}), p=1.0"),
    ("RandomMotionBlur", "Motion blur with a random angle and direction.", [Select("kernel_size", [3, 7, 11], 7), Select("angle", [15.0, 45.0, 90.0], 45.0)], lambda k, a: A.RandomMotionBlur(k, a, 0.5, p=1.0), "{kernel_size}, {angle}, 0.5, p=1.0"),
    ("RandomPerspective", "Random perspective warp.", [Select("distortion_scale", [0.2, 0.5, 0.8], 0.5), Select("sampling_method", ["basic", "area_preserving"], "basic")], lambda d, m: A.RandomPerspective(d, sampling_method=m, p=1.0), "{distortion_scale}, sampling_method=\"{sampling_method}\", p=1.0"),
    ("RandomPlanckianJitter", "Colour temperature jitter along the Planckian locus.", [Select("mode", ["blackbody", "cied"], "blackbody")], lambda v: A.RandomPlanckianJitter(mode=v, p=1.0), "mode=\"{mode}\", p=1.0"),
    ("RandomPlasmaBrightness", "Brightness modulated by a plasma fractal.", [Select("max_intensity", [0.6], 0.6)], lambda v: A.RandomPlasmaBrightness(intensity=(0.0, v), p=1.0), "intensity=(0.0, {max_intensity}), p=1.0"),
    ("RandomPlasmaContrast", "Contrast modulated by a plasma fractal.", [Select("max_roughness", [0.5], 0.5)], lambda v: A.RandomPlasmaContrast(roughness=(0.1, v), p=1.0), "roughness=(0.1, {max_roughness}), p=1.0"),
    ("RandomPlasmaShadow", "Shadows shaped by a plasma fractal.", [Select("shade", [-0.6], -0.6)], lambda v: A.RandomPlasmaShadow(shade_intensity=(v, 0.0), p=1.0), "shade_intensity=({shade}, 0.0), p=1.0"),
    ("RandomPosterize", "Posterisation to a random number of bits.", [Select("bits", [2, 3, 5], 3)], lambda v: A.RandomPosterize(bits=v, p=1.0), "bits={bits}, p=1.0"),
    ("RandomRGBShift", "Random per-channel shift.", [Select("r_shift_limit", [0.1, 0.3], 0.3), Select("g_shift_limit", [0.1, 0.3], 0.3), Select("b_shift_limit", [0.1, 0.3], 0.3)], lambda r, g, b: A.RandomRGBShift(r, g, b, p=1.0), "{r_shift_limit}, {g_shift_limit}, {b_shift_limit}, p=1.0"),
    ("RandomRain", "Draws random rain streaks (the drop count is read in Python, so this shows frames).", [Select("drops", [300, 1000], 300)], lambda v: A.RandomRain(number_of_drops=(v, v + 200), drop_height=(5, 20), drop_width=(-5, 5), p=1.0), "number_of_drops=({drops}, {drops_plus_200}), drop_height=(5, 20), drop_width=(-5, 5), p=1.0"),
    ("RandomResizedCrop", "Random crop of random scale, resized to a square.", [Select("size", [128, 192], 128), Select("min_scale", [0.08, 0.3, 0.6], 0.3)], lambda sz, ms: A.RandomResizedCrop((sz, sz), scale=(ms, 1.0), p=1.0), "({size}, {size}), scale=({min_scale}, 1.0), p=1.0"),
    ("RandomRotation", "Random rotation within +/- degrees.", [Select("degrees", [15, 45, 90, 180], 45)], lambda v: A.RandomRotation(float(v), p=1.0), "{degrees}, p=1.0"),
    ("RandomRotation90", "Random multiple of 90 degrees.", [Select("p", [1.0], 1.0)], lambda v: A.RandomRotation90((0, 3), p=1.0), "(0, 3), p=1.0"),
    ("RandomSaltAndPepperNoise", "Salt and pepper noise.", [Select("max_amount", [0.03, 0.06, 0.12], 0.06), Select("salt_vs_pepper", [0.2, 0.5, 0.8], 0.5)], lambda a, sp: A.RandomSaltAndPepperNoise(amount=(0.01, a), salt_vs_pepper=(sp, sp), p=1.0), "amount=(0.01, {max_amount}), salt_vs_pepper=({salt_vs_pepper}, {salt_vs_pepper}), p=1.0"),
    ("RandomSaturation", "Random saturation factor in [1-s, 1+s].", [Select("strength", [0.2, 0.5, 0.8], 0.5)], lambda v: A.RandomSaturation((1 - v, 1 + v), p=1.0), "({strength_one_minus}, {strength_one_plus}), p=1.0"),
    ("RandomSharpness", "Random sharpness.", [Select("sharpness", [0.5, 1.0, 2.0], 1.0)], lambda v: A.RandomSharpness(v, p=1.0), "{sharpness}, p=1.0"),
    ("RandomShear", "Random shear.", [Select("shear", [0.2, 0.4, 0.6], 0.4)], lambda v: A.RandomShear(v, p=1.0), "{shear}, p=1.0"),
    ("RandomSnow", "Snow effect.", [Select("snow_coefficient", [0.5], 0.5)], lambda v: A.RandomSnow(snow_coefficient=(v, v), brightness=(2, 2), p=1.0), "snow_coefficient=({snow_coefficient}, {snow_coefficient}), brightness=(2, 2), p=1.0"),
    ("RandomSolarize", "Random solarisation threshold and addition.", [Select("thresholds", [0.1, 0.3], 0.1), Select("additions", [0.0, 0.1, 0.2], 0.1)], lambda t, a: A.RandomSolarize(t, a, p=1.0), "{thresholds}, {additions}, p=1.0"),
    ("RandomThinPlateSpline", "Random thin-plate-spline warp (needs a linear solver, so this shows frames).", [Select("scale", [0.1, 0.2, 0.4], 0.2)], lambda v: A.RandomThinPlateSpline(v, p=1.0), "{scale}, p=1.0"),
    ("RandomTranslate", "Random translation as a fraction of the size.", [Select("translate", [0.1, 0.2, 0.3], 0.2)], lambda v: A.RandomTranslate((-v, v), (-v, v), p=1.0), "(-{translate}, {translate}), (-{translate}, {translate}), p=1.0"),
    ("RandomJigsaw", "Shuffles the tiles of a grid (the permutation is applied in Python, so this shows frames).", [Select("grid", [2, 4], 2)], lambda v: A.RandomJigsaw((v, v), p=1.0), "({grid}, {grid}), p=1.0"),
]


def _aug_spec(name: str, summary: str, selects: list[Select], factory: Callable[..., torch.nn.Module], args: str) -> Spec:
    params = []
    for sel in selects:
        new = Select(sel.name, sel.choices, sel.default, literals=sel.literals)
        if new.type != "str":
            new.derived = {  # type: ignore[attr-defined] - carried into the registry for the snippet
                str(c): {"one_minus": f"{1 - c:g}", "one_plus": f"{1 + c:g}", "plus_200": str(c + 200) if isinstance(c, int) else ""} for c in sel.choices
            }
        params.append(new)
    names = [p.name for p in params]

    def call(x: torch.Tensor, _name: str = name, _factory: Callable[..., torch.nn.Module] = factory, _names: tuple[str, ...] = tuple(names), **kw: Any) -> torch.Tensor:
        values = tuple(kw[n] for n in _names)
        return _module((_name, values), lambda: _factory(*values))(x)

    return Spec(f"kornia.augmentation.{name}", summary, params, call, f"kornia.augmentation.{name}({args})({{img}})", stochastic=True)


SPECS += [_aug_spec(*row) for row in AUGS]

# Module forms of the functions above: not listed separately, shown as the second Python block.
MODULE_FORMS: dict[str, str] = {
    "kornia.filters.GaussianBlur2d": "kornia.filters.gaussian_blur2d",
    "kornia.filters.BoxBlur": "kornia.filters.box_blur",
    "kornia.filters.MedianBlur": "kornia.filters.median_blur",
    "kornia.filters.BilateralBlur": "kornia.filters.bilateral_blur",
    "kornia.filters.GuidedBlur": "kornia.filters.guided_blur",
    "kornia.filters.MotionBlur": "kornia.filters.motion_blur",
    "kornia.filters.UnsharpMask": "kornia.filters.unsharp_mask",
    "kornia.filters.Laplacian": "kornia.filters.laplacian",
    "kornia.filters.Sobel": "kornia.filters.sobel",
    "kornia.filters.BlurPool2D": "kornia.filters.blur_pool2d",
    "kornia.filters.MaxBlurPool2D": "kornia.filters.max_blur_pool2d",
    "kornia.filters.EdgeAwareBlurPool2D": "kornia.filters.edge_aware_blur_pool2d",
    "kornia.filters.Canny": "kornia.filters.canny",
    "kornia.filters.InRange": "kornia.filters.in_range",
    "kornia.color.RgbToGrayscale": "kornia.color.rgb_to_grayscale",
    "kornia.color.RgbToBgr": "kornia.color.rgb_to_bgr",
    "kornia.color.BgrToRgb": "kornia.color.bgr_to_rgb",
    "kornia.color.RgbToHsv": "kornia.color.rgb_to_hsv",
    "kornia.color.RgbToHls": "kornia.color.rgb_to_hls",
    "kornia.color.RgbToLab": "kornia.color.rgb_to_lab",
    "kornia.color.RgbToLuv": "kornia.color.rgb_to_luv",
    "kornia.color.RgbToXyz": "kornia.color.rgb_to_xyz",
    "kornia.color.RgbToYcbcr": "kornia.color.rgb_to_ycbcr",
    "kornia.color.RgbToYuv": "kornia.color.rgb_to_yuv",
    "kornia.color.RgbToLinearRgb": "kornia.color.rgb_to_linear_rgb",
    "kornia.color.LinearRgbToRgb": "kornia.color.linear_rgb_to_rgb",
    "kornia.color.Sepia": "kornia.color.sepia",
    "kornia.color.RgbToRgba": "kornia.color.rgb_to_rgba",
    "kornia.color.RgbToRaw": "kornia.color.rgb_to_raw",
    "kornia.color.GrayscaleToRgb": "kornia.color.grayscale_to_rgb",
    "kornia.color.ApplyColorMap": "kornia.color.apply_colormap",
    "kornia.enhance.AdjustBrightness": "kornia.enhance.adjust_brightness",
    "kornia.enhance.AdjustBrightnessAccumulative": "kornia.enhance.adjust_brightness_accumulative",
    "kornia.enhance.AdjustContrast": "kornia.enhance.adjust_contrast",
    "kornia.enhance.AdjustContrastWithMeanSubtraction": "kornia.enhance.adjust_contrast_with_mean_subtraction",
    "kornia.enhance.AdjustGamma": "kornia.enhance.adjust_gamma",
    "kornia.enhance.AdjustHue": "kornia.enhance.adjust_hue",
    "kornia.enhance.AdjustSaturation": "kornia.enhance.adjust_saturation",
    "kornia.enhance.AdjustSaturationWithGraySubtraction": "kornia.enhance.adjust_saturation_with_gray_subtraction",
    "kornia.enhance.AdjustLog": "kornia.enhance.adjust_log",
    "kornia.enhance.AdjustSigmoid": "kornia.enhance.adjust_sigmoid",
    "kornia.enhance.Invert": "kornia.enhance.invert",
    "kornia.enhance.Normalize": "kornia.enhance.normalize",
    "kornia.enhance.Threshold": "kornia.enhance.threshold",
    "kornia.enhance.AddWeighted": "kornia.enhance.add_weighted",
    "kornia.enhance.JPEGCodecDifferentiable": "kornia.enhance.jpeg_codec_differentiable",
    "kornia.enhance.IntegralImage": "kornia.enhance.integral_image",
    "kornia.geometry.transform.Hflip": "kornia.geometry.transform.hflip",
    "kornia.geometry.transform.Vflip": "kornia.geometry.transform.vflip",
    "kornia.geometry.transform.Rot180": "kornia.geometry.transform.rot180",
    "kornia.geometry.transform.Rotate": "kornia.geometry.transform.rotate",
    "kornia.geometry.transform.Translate": "kornia.geometry.transform.translate",
    "kornia.geometry.transform.Scale": "kornia.geometry.transform.scale",
    "kornia.geometry.transform.Shear": "kornia.geometry.transform.shear",
    "kornia.geometry.transform.Resize": "kornia.geometry.transform.resize",
    "kornia.geometry.transform.Rescale": "kornia.geometry.transform.rescale",
    "kornia.geometry.transform.CenterCrop2D": "kornia.geometry.transform.center_crop",
    "kornia.geometry.transform.PyrDown": "kornia.geometry.transform.pyrdown",
    "kornia.geometry.transform.PyrUp": "kornia.geometry.transform.pyrup",
    # module forms of functions that have no demo yet: still not listed twice
    "kornia.filters.MotionBlur3D": "kornia.filters.motion_blur3d",
    "kornia.filters.JointBilateralBlur": "kornia.filters.joint_bilateral_blur",
    "kornia.color.RawToRgb2x2Downscaled": "kornia.color.raw_to_rgb_2x2_downscaled",
    "kornia.geometry.transform.HomographyWarper": "kornia.geometry.transform.homography_warp",
    "kornia.enhance.Denormalize": "kornia.enhance.denormalize",
}

# Public names that are not image operators at all and are left out of the playground entirely:
# kernel and matrix builders, enums, base classes, tuple-returning helpers.
HIDDEN: set[str] = set(
    "gaussian gaussian_blur2d_t get_binary_kernel2d get_box_kernel1d get_box_kernel2d get_diff_kernel2d "
    "get_gaussian_discrete_kernel1d get_gaussian_erf_kernel1d get_gaussian_kernel1d get_gaussian_kernel1d_t "
    "get_gaussian_kernel2d get_gaussian_kernel2d_t get_gaussian_kernel3d get_gaussian_kernel3d_t get_hanning_kernel1d "
    "get_hanning_kernel2d get_laplacian_kernel1d get_laplacian_kernel2d get_motion_kernel2d get_motion_kernel3d "
    "get_sobel_kernel2d get_spatial_gradient_kernel2d get_spatial_gradient_kernel3d laplacian_1d "
    "get_affine_matrix2d get_affine_matrix3d get_perspective_transform get_perspective_transform3d "
    "get_projective_transform get_rotation_matrix2d get_shear_matrix2d get_shear_matrix3d get_tps_transform "
    "get_translation_matrix2d invert_affine_transform projection_from_Rt warp_grid warp_grid3d warp_points_tps "
    "ColorMap ColorMapType CFA RGBColor ThresholdType BaseModel BaseWarper Homography Similarity Affine "
    "rgb_to_yuv420 rgb_to_yuv422 RgbToYuv420 RgbToYuv422 yuv420_to_rgb yuv422_to_rgb Yuv420ToRgb Yuv422ToRgb "
    "otsu_threshold OtsuThreshold spatial_gradient SpatialGradient spatial_gradient3d SpatialGradient3d "
    "build_pyramid build_laplacian_pyramid ScalePyramid zca_mean zca_whiten ZCAWhitening histogram histogram2d "
    "image_histogram2d integral_tensor IntegralTensor linear_transform ImageRegistrator StableDiffusionDissolving "
    "RandomDissolving rgb_to_rgb255 RgbToRgb255 rgb255_to_rgb Rgb255ToRgb Rescale "
    "AugmentationSequential ImageSequential PatchSequential VideoSequential ManyToManyAugmentationDispather "
    "ManyToOneAugmentationDispather "
    # conversions whose input is not an RGB image cannot be shown on an RGB sample
    "hsv_to_rgb hls_to_rgb lab_to_rgb luv_to_rgb xyz_to_rgb ycbcr_to_rgb yuv_to_rgb raw_to_rgb "
    "raw_to_rgb_2x2_downscaled rgb255_to_normals normals_to_rgb255 rgba_to_rgb rgba_to_bgr bgr_to_grayscale "
    "bgr_to_rgba HsvToRgb HlsToRgb LabToRgb LuvToRgb XyzToRgb YcbcrToRgb YuvToRgb RawToRgb RawToRgb2x2Downscaled "
    "Rgb255ToNormals NormalsToRgb255 RgbaToRgb RgbaToBgr BgrToGrayscale BgrToRgba denormalize "  # (enhance.Denormalize folds into it)
    # low-level variants of demoed operators
    "adjust_hue_raw adjust_saturation_raw "
    # warps and crops driven by matrices, boxes or point sets: not slider material
    "crop_and_resize crop_and_resize3d crop_by_boxes crop_by_boxes3d crop_by_transform_mat crop_by_transform_mat3d "
    "homography_warp homography_warp3d resize_to_be_divisible warp_affine warp_affine3d warp_image_tps "
    "warp_perspective warp_perspective3d".split()
)

# Why a listed public name has no demo yet. Everything else without a spec reads "no demo yet".
UNSUPPORTED_REASONS: dict[str, str] = {
    "needs a transformation matrix or point set as input": "affine warp_affine warp_perspective homography_warp "
    "HomographyWarper warp_image_tps remap crop_and_resize crop_by_boxes crop_by_indices crop_by_transform_mat "
    "resize_to_be_divisible",
    "3D (volumetric) operator, planned for a later phase": "filter3d motion_blur3d MotionBlur3D equalize3d rotate3d affine3d "
    "warp_affine3d warp_perspective3d homography_warp3d center_crop3d crop_and_resize3d crop_by_boxes3d "
    "crop_by_transform_mat3d RandomAffine3D RandomCrop3D RandomDepthicalFlip3D RandomEqualize3D "
    "RandomHorizontalFlip3D RandomMotionBlur3D RandomPerspective3D RandomRotation3D RandomVerticalFlip3D CenterCrop3D",
    "needs labels or a batch of several images": "PatchMix RandomCutMixV2 RandomMixUpV2 RandomMosaic",
}

DOC_PAGES = {
    "kornia.filters": "filters.html",
    "kornia.color": "color.html",
    "kornia.enhance": "enhance.html",
    "kornia.morphology": "morphology.html",
    "kornia.geometry.transform": "geometry.transform.html",
    "kornia.augmentation": "augmentation.module.html",
}


def doc_url(op_id: str) -> str:
    package = op_id.rsplit(".", 1)[0]
    page = DOC_PAGES.get(package, package.replace("kornia.", "") + ".html")
    return f"{DOCS}/{page}#{op_id}"
