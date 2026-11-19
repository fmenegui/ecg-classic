"""Flatten a photographed page with the dense page-flattening network (hybrid/train_dewarp.py).

The network gives, for every pixel of the photo (at 512 x 704), its normalised flat-page
coordinate and whether it shows the page. The flat page is then sampled from the full-resolution
photo: for every pixel of the output page, Newton iterations find the photo position whose
predicted coordinate is that pixel (the scheme of hybrid/rendergen.rectify). The output is an
A4-landscape page image that the classical pipeline reads like a scan (its own grid geometry
still refines scale and residual distortion).
"""
from __future__ import annotations

import numpy as np

IN_HW = (512, 704)
_SESS = {}


def _session(path):
    if path not in _SESS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _SESS[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _SESS[path]


def predict_field(img_bgr, path):
    """Field (2, 512, 704) of normalised page coordinates and visibility probability (512, 704)."""
    import cv2
    x = cv2.resize(img_bgr, (IN_HW[1], IN_HW[0]), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    out = _session(path).run(None, {"x": x.transpose(2, 0, 1)[None].copy()})[0][0]
    return out[:2], 1 / (1 + np.exp(-out[2]))


def flatten(img_bgr, path, out_wh=(2970, 2100), work_wh=(1188, 840), iters=25):
    """The flat page (out_wh, BGR) and the fraction of it the photo shows; None if the network
    sees no page."""
    import torch
    import torch.nn.functional as F
    f, vis = predict_field(img_bgr, path)
    if (vis > 0.5).mean() < 0.05:
        return None, 0.0
    f = torch.from_numpy(f).float()
    h, w = f.shape[1:]
    # derivatives of the field (per normalised photo unit) for the Newton step
    fx = torch.zeros_like(f); fy = torch.zeros_like(f)
    fx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (4 / w); fx[:, :, 0] = fx[:, :, 1]; fx[:, :, -1] = fx[:, :, -2]
    fy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (4 / h); fy[:, 0] = fy[:, 1]; fy[:, -1] = fy[:, -2]
    stack = torch.cat([f, fx, fy], 0)[None]                         # (1, 6, h, w)
    Wk, Hk = work_wh
    ys, xs = torch.meshgrid(torch.arange(Hk, dtype=torch.float32), torch.arange(Wk, dtype=torch.float32), indexing="ij")
    c = torch.stack([(xs + 0.5) / Wk * 2 - 1, (ys + 0.5) / Hk * 2 - 1], -1)   # target page coordinates
    # affine start from the visible pixels
    m = torch.from_numpy(vis > 0.5)
    py_, px_ = torch.nonzero(m, as_tuple=True)
    sel = torch.randperm(len(py_))[:20000]
    P = torch.stack([(px_[sel].float() + 0.5) / w * 2 - 1, (py_[sel].float() + 0.5) / h * 2 - 1], 1)
    Q = f[:, py_[sel], px_[sel]].T
    M = torch.linalg.lstsq(torch.cat([P, torch.ones(len(P), 1)], 1), Q).solution      # photo -> page
    A, b = M[:2].T, M[2]
    p = (c - b) @ torch.linalg.inv(A).T
    for _ in range(iters):
        s_ = F.grid_sample(stack, p[None], align_corners=False, padding_mode="border")[0].permute(1, 2, 0)
        r = c - s_[..., :2]
        det = s_[..., 2] * s_[..., 5] - s_[..., 4] * s_[..., 3]
        det = torch.where(det.abs() < 1e-6, torch.full_like(det, 1e-6), det)
        dx = (s_[..., 5] * r[..., 0] - s_[..., 4] * r[..., 1]) / det
        dy = (-s_[..., 3] * r[..., 0] + s_[..., 2] * r[..., 1]) / det
        p = p + torch.stack([dx, dy], -1).clamp(-0.2, 0.2)
    s_ = F.grid_sample(stack, p[None], align_corners=False, padding_mode="border")[0].permute(1, 2, 0)
    err = ((c - s_[..., :2]) * torch.tensor([Wk / 2, Hk / 2])).norm(dim=-1)
    seen = (err < 1.0) & (p.abs() <= 1).all(-1)
    # sample the full-resolution photo on the output page grid
    Wo, Ho = out_wh
    pu = F.interpolate(p.permute(2, 0, 1)[None], size=(Ho, Wo), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    su = F.interpolate(seen.float()[None, None], size=(Ho, Wo), mode="bilinear", align_corners=False)[0, 0] > 0.5
    ph = torch.from_numpy(np.ascontiguousarray(img_bgr)).permute(2, 0, 1).float()[None]
    out = F.grid_sample(ph, pu[None], align_corners=False, padding_mode="border")[0]
    if su.any():
        paper = out[:, su].median(1).values
        out = torch.where(su[None], out, paper[:, None, None])
    return out.permute(1, 2, 0).clamp(0, 255).byte().numpy(), float(seen.float().mean())
