"""The two scoring conventions we report, in one place.

1. **Kaggle** ("PhysioNet - Digitization of ECG Images"): the official metric, copied verbatim
   from the competition's metric notebook (kaggle.com/code/metric/physionet-ecg-signal-extraction-metric,
   Apache-2.0). Per lead: shift of up to 0.2 s chosen by cross-correlation of the centred
   signals, samples shifted out count as missing (prediction 0), then the vertical offset that
   minimises the error is removed. Signal and noise powers are summed over the 12 leads of the
   image, the image SNR is capped at 1e6 (60 dB), and the score is 10*log10 of the MEAN of the
   LINEAR image SNRs (not the mean of dB). Only the samples shown on the paper are scored
   (lead II 10 s, the other leads 2.5 s); a lead we did not read is submitted as zeros.

2. **PMcardio** (Demolder et al., medRxiv 2024.08.31.24312876, PM-ECG-ID): per lead, alignment
   of up to +-0.5 s by Pearson correlation, vertical offset removed by the median difference,
   then PCC, RMSE (mV) and SNR (dB). A lead is a failure when it is missing or covers < 50 %
   of the reference: PCC 0, SNR 0 dB and, for RMSE, excluded ("RMSE without failures"). The
   whole-set numbers are means over leads; the failure rate is the share of failed leads.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import scipy.optimize
import scipy.signal

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ---------------------------------------------------------------------------------------------
# Kaggle: verbatim from the official metric notebook (Apache-2.0), only the DataFrame plumbing
# is replaced by arrays.
MAX_TIME_SHIFT = 0.2
PERFECT_SCORE = 1e6


def compute_power(label: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    prediction = prediction.copy()
    prediction[~np.isfinite(prediction)] = 0
    noise = label - prediction
    p_signal = np.sum(label ** 2)
    p_noise = np.sum(noise ** 2)
    return p_signal, p_noise


def compute_snr(signal: float, noise: float) -> float:
    if noise == 0:
        snr = PERFECT_SCORE
    elif signal == 0:
        snr = 0
    else:
        snr = min((signal / noise), PERFECT_SCORE)
    return snr


def align_signals(label: np.ndarray, pred: np.ndarray, max_shift: float = float("inf")) -> np.ndarray:
    label_arr = np.asarray(label, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    label_arr_centered = label_arr - np.mean(label_arr)
    pred_arr_centered = pred_arr - np.mean(pred_arr)
    correlation = scipy.signal.correlate(label_arr_centered, pred_arr_centered, mode="full")
    n_label = np.size(label_arr)
    n_pred = np.size(pred_arr)
    lags = scipy.signal.correlation_lags(n_label, n_pred, mode="full")
    valid_lags_mask = (lags >= -max_shift) & (lags <= max_shift)
    max_correlation = np.nanmax(correlation[valid_lags_mask])
    all_max_indices = np.flatnonzero(correlation == max_correlation)
    best_idx = min(all_max_indices, key=lambda i: abs(lags[i]))
    time_shift = lags[best_idx]
    start_padding_len = max(time_shift, 0)
    pred_slice_start = max(-time_shift, 0)
    pred_slice_end = min(n_label - time_shift, n_pred)
    end_padding_len = max(n_label - n_pred - time_shift, 0)
    aligned_pred = np.concatenate((np.full(start_padding_len, np.nan), pred_arr[pred_slice_start:pred_slice_end],
                                   np.full(end_padding_len, np.nan)))

    def objective_func(v_shift):
        return np.nansum((label_arr - (aligned_pred - v_shift)) ** 2)

    if np.any(np.isfinite(label_arr) & np.isfinite(aligned_pred)):
        results = scipy.optimize.minimize_scalar(objective_func, method="Brent")
        aligned_pred -= results.x
    return aligned_pred


def kaggle_image_snr(gt: np.ndarray, pred: np.ndarray, fs: float) -> Tuple[float, Dict[str, float]]:
    """Linear SNR of one image as the competition computes it.

    gt, pred: (12, n) on the ground-truth grid, LEADS order; gt is NaN where the paper does not
    show the lead. The scored samples of each lead are those shown; the prediction there is what
    a submission would hold (NaN -> 0, as a missing value must be submitted as a number).
    Returns (linear SNR, per-lead dB for diagnosis)."""
    s_sig = s_noi = 0.0
    per = {}
    for i, name in enumerate(LEADS):
        m = np.isfinite(gt[i])
        if not m.any():
            continue
        label = gt[i][m].astype(np.float64)
        p = np.nan_to_num(pred[i][m].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        aligned = align_signals(label, p, int(fs * MAX_TIME_SHIFT))
        ps, pn = compute_power(label, aligned)
        s_sig += ps
        s_noi += pn
        per[name] = float(10 * np.log10(max(ps, 1e-12) / max(pn, 1e-12)))
    return compute_snr(s_sig, s_noi), per


def kaggle_score(linear_snrs: Iterable[float]) -> float:
    """Competition score: 10*log10 of the mean linear image SNR, floored at -384 dB."""
    v = np.asarray(list(linear_snrs), dtype=np.float64)
    if not len(v):
        return float("nan")
    return max(float(10 * np.log10(max(v.mean(), 1e-300))), -384.0)


# ---------------------------------------------------------------------------------------------
# PMcardio style


def pm_lead_metrics(g: np.ndarray, p: np.ndarray) -> Optional[dict]:
    """PCC, RMSE and SNR of one aligned lead; None = failure (< 50 % of the reference covered)."""
    m = np.isfinite(p) & np.isfinite(g)
    if m.sum() < 0.5 * np.isfinite(g).sum():
        return None
    gg, pp = g[m], p[m]
    pp = pp + np.median(gg - pp)
    e = gg - pp
    snr = 10 * np.log10(max(float(gg @ gg), 1e-12) / max(float(e @ e), 1e-12))
    pcc = float(np.corrcoef(gg, pp)[0, 1]) if gg.std() > 1e-9 and pp.std() > 1e-9 else 0.0
    return dict(snr=float(snr), pcc=pcc, rmse=float(np.sqrt(np.mean(e * e))))


def pm_best_shift(g: np.ndarray, F: np.ndarray, a: int, max_shift: int, span: int = 0) -> Optional[np.ndarray]:
    """Prediction cut aligned to the reference cut g by maximum Pearson correlation within
    +-max_shift samples of position a in the whole-page prediction F (coarse step 2, refined
    by 1). span: how much longer the predicted strip is than the cut."""
    n = len(g)
    best = None

    def cut(s):
        i0 = a + s
        p = np.full(n, np.nan)
        lo, hi = max(0, i0), min(len(F), i0 + n)
        if hi - lo < n // 2:
            return None
        p[lo - i0:hi - i0] = F[lo:hi]
        return p

    for s in range(-max_shift, max_shift + max(0, span) + 1, 2):
        p = cut(s)
        if p is None:
            continue
        m = np.isfinite(p) & np.isfinite(g)
        if m.sum() < n // 2 or p[m].std() < 1e-9:
            continue
        c = np.corrcoef(g[m], p[m])[0, 1]
        if best is None or c > best[0]:
            best = (c, s, p)
    if best is None:
        return None
    s0 = best[1]
    for s in (s0 - 1, s0 + 1):
        p = cut(s)
        if p is None:
            continue
        m = np.isfinite(p) & np.isfinite(g)
        if m.sum() < n // 2 or p[m].std() < 1e-9:
            continue
        c = np.corrcoef(g[m], p[m])[0, 1]
        if c > best[0]:
            best = (c, s, p)
    return best[2]


def pm_image_metrics(gt: np.ndarray, pred: np.ndarray, fs: float) -> Dict[str, Optional[dict]]:
    """PMcardio-style metrics per lead for one image (gt, pred: (12, n) on the same grid)."""
    out = {}
    for i, name in enumerate(LEADS):
        m = np.flatnonzero(np.isfinite(gt[i]))
        if not len(m):
            continue
        a, b = int(m[0]), int(m[-1]) + 1
        g = gt[i][a:b].astype(np.float64)
        F = pred[i].astype(np.float64)
        if np.isfinite(F).sum() < 2:
            out[name] = None
            continue
        p = pm_best_shift(g, F, a, int(0.5 * fs))
        out[name] = pm_lead_metrics(g, p) if p is not None else None
    return out


def pm_summary(per_image: Iterable[Dict[str, Optional[dict]]]) -> dict:
    """Whole-set PMcardio numbers: mean SNR and PCC over leads (failure = 0), failure rate,
    and SNR / RMSE over the leads that did not fail."""
    snr, pcc, fail, snr_ok, rmse_ok = [], [], [], [], []
    for d in per_image:
        for mt in d.values():
            fail.append(mt is None)
            snr.append(mt["snr"] if mt else 0.0)
            pcc.append(mt["pcc"] if mt else 0.0)
            if mt:
                snr_ok.append(mt["snr"]); rmse_ok.append(mt["rmse"])
    f = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return dict(leads=len(fail), snr=f(snr), pcc=f(pcc), failure=f(fail), snr_ok=f(snr_ok), rmse_ok=f(rmse_ok))
