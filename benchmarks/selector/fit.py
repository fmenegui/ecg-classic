"""Fit the choice between the classical and the page network's reading on the selector pages.

    python benchmarks/selector/fit.py rows.jsonl [rows2.jsonl ...] [--out selector.json]

Model: logistic regression on the differences of the two readings' features (network minus
classical) plus a same-layout flag, each sample weighted by how much the choice matters
(|SNR difference|, capped at 10 dB). Reported with 5-fold cross-validation as the mean page SNR
of the chosen reading: current rule, fitted rule, always classical, always network, oracle.
The fitted weights go to Config.page_selector (digitize uses them instead of the score margin).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for l in Path(p).read_text().splitlines():
            r = json.loads(l)
            if r.get("features_classical") and r.get("features_network"):
                rows.append(r)
    return rows


def design(rows, names=None):
    if names is None:
        keys = set()
        for r in rows:
            keys |= set(r["features_classical"]) & set(r["features_network"])
        names = sorted(keys) + ["same_layout"]
    X = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for j, n in enumerate(names):
            if n == "same_layout":
                X[i, j] = float(r.get("layout_classical") == r.get("layout_network"))
            else:
                X[i, j] = float(r["features_network"].get(n, 0.0) or 0.0) - float(r["features_classical"].get(n, 0.0) or 0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def logistic(X, y, w, l2=1.0, iters=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Z = (X - mu) / sd
    b = np.zeros(Z.shape[1]); c = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ b + c)))
        g = (w * (p - y))
        b -= lr * (Z.T @ g / w.sum() + l2 * b / len(y))
        c -= lr * g.sum() / w.sum()
    return dict(mean=mu.tolist(), std=sd.tolist(), coef=b.tolist(), bias=float(c))


def predict(m, X):
    Z = (X - np.array(m["mean"])) / np.array(m["std"])
    return Z @ np.array(m["coef"]) + m["bias"] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()
    rows = load(a.rows)
    X, names = design(rows)
    sc = np.array([r["snr_classical"] for r in rows]); sn = np.array([r["snr_network"] for r in rows])
    cur = np.array([r["chosen"] == "network" for r in rows])
    y = (sn > sc).astype(float); w = np.minimum(np.abs(sn - sc), 10.0) + 1e-3
    rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(rows))
    pick = np.zeros(len(rows), bool)
    for k in range(5):
        tr, te = fold != k, fold == k
        m = logistic(X[tr], y[tr], w[tr], l2=a.l2)
        pick[te] = predict(m, X[te])
    choose = lambda p: float(np.mean(np.where(p, sn, sc)))  # noqa: E731
    print(f"{len(rows)} pages; differ by > 1 dB on {int((np.abs(sn - sc) > 1).sum())}")
    print(f"mean page SNR: classical {sc.mean():.2f} | network {sn.mean():.2f} | current rule {choose(cur):.2f} | "
          f"fitted (5-fold CV) {choose(pick):.2f} | oracle {np.maximum(sc, sn).mean():.2f}")
    by = {}
    for r, p_, c_ in zip(rows, pick, cur):
        d = r["degradation"].rstrip("0123456789.-")
        by.setdefault(d, []).append((r["snr_classical"], r["snr_network"], p_, c_))
    for d, v in sorted(by.items()):
        v = np.array(v, dtype=float)
        print(f"  {d:12s} n={len(v):3d} current {np.mean(np.where(v[:, 3] > 0, v[:, 1], v[:, 0])):6.2f} "
              f"fitted {np.mean(np.where(v[:, 2] > 0, v[:, 1], v[:, 0])):6.2f} oracle {np.maximum(v[:, 0], v[:, 1]).mean():6.2f}")
    m = logistic(X, y, w, l2=a.l2)
    m["features"] = names
    order = np.argsort(-np.abs(m["coef"]))
    print("largest weights:", [(names[i], round(m["coef"][i], 2)) for i in order[:8]])
    if a.out:
        a.out.write_text(json.dumps(m))
        print("selector ->", a.out)


if __name__ == "__main__":
    main()
