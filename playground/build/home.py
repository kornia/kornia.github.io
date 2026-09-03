#!/usr/bin/env python
"""Graphs for the homepage demo that the operator catalog does not provide.

The homepage plays a clip and must apply the *same* augmentation to every frame, but kornia's random
augmentations draw their parameters inside the graph on every run. So the homepage gets an affine warp
whose parameters are graph inputs: the page draws them once per pass of the clip (with RandomAffine's
default ranges) and feeds the same values to every frame.

    python playground/build/home.py     # writes playground/home/affine_256.onnx (fixed 1x3x256x256)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

import kornia
from kornia.geometry.transform import get_affine_matrix2d, warp_affine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "home"
SIZE = 256


class Affine(torch.nn.Module):
    """warp_affine driven by scalar parameters: angle (deg), translation (px), scale, shear (deg)."""

    def forward(self, image: torch.Tensor, angle: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor,
                scale: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        center = torch.tensor([[SIZE / 2.0, SIZE / 2.0]], dtype=image.dtype).expand(b, 2)
        translations = torch.stack([tx, ty], dim=-1).expand(b, 2)
        scales = torch.stack([scale, scale], dim=-1).expand(b, 2)
        matrix = get_affine_matrix2d(translations, center, scales, angle.expand(b), sx=shear.expand(b), sy=torch.zeros_like(angle).expand(b))
        return warp_affine(image, matrix[:, :2], (SIZE, SIZE), padding_mode="zeros")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    model = Affine().eval()
    x = torch.rand(1, 3, SIZE, SIZE)
    params = (torch.tensor([12.0]), torch.tensor([10.0]), torch.tensor([-6.0]), torch.tensor([1.08]), torch.tensor([5.0]))
    path = OUT / "affine_256.onnx"
    torch.onnx.export(model, (x,) + params, str(path), dynamo=True, opset_version=18,
                      input_names=["image", "angle", "tx", "ty", "scale", "shear"], output_names=["warped"])
    # the exporter may park large constants in a side file; the browser fetches one file, so inline them
    model_proto = onnx.load(str(path), load_external_data=True)
    onnx.save_model(model_proto, str(path), save_as_external_data=False)
    side = path.with_name(path.name + ".data")
    if side.exists():
        side.unlink()
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)
    with torch.no_grad():
        ref = model(x, *params).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {"image": x.numpy()}
    for name, value in zip(["angle", "tx", "ty", "scale", "shear"], params):
        feeds[name] = value.numpy()
    out = sess.run(None, feeds)[0]
    diff = float(np.abs(out - ref).max())
    if diff > 1e-3:
        print(f"mismatch vs eager: {diff}", file=sys.stderr)
        return 1
    print(f"home/affine_256.onnx: {path.stat().st_size / 1e3:.0f} kB, verified vs eager (max diff {diff:.1e}), kornia {kornia.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
