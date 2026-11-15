"""Nota de qualidade por derivação, sem gabarito.

O digitalizador sempre devolve um sinal; esta nota diz o quanto ele merece confiança. Só usa
o que um ECG de verdade tem de ter, e nada do conteúdo clínico:

  cobertura   fração das amostras do trecho impresso que foram lidas
  batimentos  as derivações impressas no mesmo instante são simultâneas: os QRS de uma têm de
              cair nos instantes dos QRS da página (mediana entre as outras derivações); um
              traço que seguiu grade, texto ou a linha vizinha tem a energia fora deles
  repetição   os batimentos de um mesmo trecho se parecem entre si (mediana da correlação com
              o batimento mediano); ruído e traço trocado não se repetem
  saltos      degraus de > 0,5 mV em 2 ms em mais de 5% das amostras (rastreio trocando de traço)
  amplitude   pico a pico fora de 0,05–8 mV
  membros     resíduo de Einthoven/Goldberger ANTES da fusão, onde 3 derivações de membros
              são simultâneas (a fusão zera o resíduo, por isso ele é medido antes)

A nota vai de 0 a 1: cobertura x repetição (por derivação) x confiança do layout (página,
aplicada em digitize). As outras partes ficam no relatório para diagnóstico. Não é uma
probabilidade calibrada.

Medida contra o gabarito (PM-ECG-ID, 60 ECGs, 15/11/2026): AUC 0,75 para separar derivações
com PCC < 0,5; a confiança do layout sozinha dá 0,76. É pouco para recusar com segurança
(recusar os 20% piores sobe o SNR das aceitas de 8,7 para 9,8 dB e ainda deixa passar 18%
de derivações ruins): a maior parte do lixo vem de layout trocado, em que cada traço é um
ECG bem formado com o nome errado. `Config(refuse_below=...)` existe, mas fica desligado.
"""
from __future__ import annotations

import numpy as np

LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")


def _slope_energy(v: np.ndarray, fs: int) -> np.ndarray:
    """Energia da derivada numa janela de ~80 ms (realça o QRS); NaN onde não há sinal."""
    d = np.diff(v, prepend=np.nan)
    e = d * d
    k = max(3, int(0.08 * fs) | 1)
    ok = np.isfinite(e)
    num = np.convolve(np.where(ok, e, 0.0), np.ones(k), mode="same")
    den = np.convolve(ok.astype(float), np.ones(k), mode="same")
    out = np.where(den >= 0.5 * k, num / np.maximum(den, 1.0), np.nan)
    return out


def _page_beats(sig: np.ndarray, fs: int):
    """Instantes dos QRS da página: picos da mediana, entre as derivações presentes em cada
    instante, da energia de inclinação normalizada por derivação. Devolve (índices, perfil)."""
    n = sig.shape[1]
    E = np.full(sig.shape, np.nan)
    for i, v in enumerate(sig):
        if np.isfinite(v).sum() < 0.3 * fs:
            continue
        e = _slope_energy(v.astype(np.float64), fs)
        s = np.nanpercentile(e, 98) if np.isfinite(e).any() else 0.0
        if s > 0:
            E[i] = np.clip(e / s, 0, 1)
    cnt = np.isfinite(E).sum(0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.where(cnt >= 2, np.nanmedian(E, axis=0), np.where(cnt == 1, np.nanmax(E, axis=0), 0.0))
    prof = np.nan_to_num(prof)
    # picos separados por >= 0,25 s (até 240 bpm) e acima de 30% do maior
    thr = 0.3 * prof.max() if prof.max() > 0 else np.inf
    gap = int(0.25 * fs)
    order = np.argsort(-prof)
    taken = np.zeros(n, bool)
    beats = []
    for j in order:
        if prof[j] < thr:
            break
        if not taken[j]:
            beats.append(int(j))
            taken[max(0, j - gap):j + gap + 1] = True
    return np.array(sorted(beats), int), prof


def _spans(ok: np.ndarray, min_len: int):
    d = np.diff(np.r_[0, ok.astype(np.int8), 0])
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]


def lead_quality(v: np.ndarray, fs: int, beats: np.ndarray, others_prof: np.ndarray | None = None) -> dict:
    """Partes da nota de uma derivação (v em mV, NaN fora do trecho impresso)."""
    ok = np.isfinite(v)
    out = dict(coverage=0.0, beat_sync=np.nan, repeat=np.nan, jumps=0.0, p2p=0.0, score=0.0, flags=[])
    if ok.sum() < 0.3 * fs:
        out["flags"].append("ausente")
        return out
    idx = np.nonzero(ok)[0]
    a, b = int(idx[0]), int(idx[-1]) + 1
    out["coverage"] = float(ok[a:b].mean())
    x = v.astype(np.float64)
    seg = x[a:b]
    out["p2p"] = float(np.nanpercentile(seg, 99.5) - np.nanpercentile(seg, 0.5))
    d = np.abs(np.diff(seg))
    d = d[np.isfinite(d)]
    out["jumps"] = float((d > 0.5 * (500.0 / fs)).mean()) if d.size else 0.0

    # batimentos: energia de inclinação perto dos QRS da página x longe deles
    e = _slope_energy(x, fs)
    w = int(0.07 * fs)
    near = np.zeros(len(x), bool)
    bs = [int(t) for t in beats if a - w <= t < b + w]
    for t in bs:
        near[max(0, t - w):t + w + 1] = True
    fin = np.isfinite(e)
    if bs and (fin & near).sum() > 5 and (fin & ~near).sum() > 5:
        tot = float(e[fin].sum())
        frac_t = float((fin & near).sum()) / float(fin.sum())
        frac_e = float(e[fin & near].sum()) / tot if tot > 0 else 0.0
        # 1 = toda a energia nos QRS; 0 = espalhada como o tempo (traço sem relação com eles)
        out["beat_sync"] = float(np.clip((frac_e - frac_t) / max(1e-6, 1.0 - frac_t), 0.0, 1.0))
    # repetição: correlação de cada batimento com o batimento mediano do trecho
    h = int(0.12 * fs)
    cuts = [x[t - h:t + h] for t in bs if t - h >= a and t + h <= b]
    cuts = [c - np.nanmedian(c) for c in cuts if np.isfinite(c).mean() > 0.9]
    if len(cuts) >= 2:
        M = np.vstack([np.nan_to_num(c) for c in cuts])
        med = np.median(M, axis=0)
        cs = []
        for c in M:
            if c.std() > 1e-6 and med.std() > 1e-6:
                cs.append(float(np.corrcoef(c, med)[0, 1]))
        if cs:
            out["repeat"] = float(np.median(cs))
    return out


def _combine(q: dict, limb_rel: float | None) -> float:
    """Nota da derivação. Só entram as partes que ordenaram bem as derivações contra o gabarito
    do PM-ECG-ID (60 ECGs, 660 derivações lidas, "ruim" = PCC < 0,5): cobertura e repetição
    dos batimentos (AUC 0,72), sem saturar (saturar em 1 empata quase tudo e piora a ordem).
    A energia nos QRS (AUC 0,59), os degraus (0,55) e o resíduo de membros (0,74, mas só existe
    em metade das derivações e misturado derrubou o conjunto para 0,63) ficam só no relatório."""
    s = float(np.clip((q["coverage"] - 0.5) / 0.4, 0.0, 1.0))     # < 50% coberto vale 0; >= 90% vale 1
    rep = q["repeat"]
    s *= float(np.clip(rep, 0.05, 1.0)) if np.isfinite(rep) else 0.8   # sem 2 batimentos no trecho: neutro
    if q["p2p"] < 0.05 or q["p2p"] > 8.0:
        s *= 0.3
    return float(s)


def assess(sig: np.ndarray, leads: list, fs: int, prefusion: np.ndarray | None = None) -> dict:
    """sig: (L, N) em mV depois da fusão; prefusion: o mesmo ANTES da fusão de Einthoven
    (para o resíduo de membros). Devolve {"leads": {nome: partes + score}, "global": float}."""
    sig = np.asarray(sig, float)
    beats, _ = _page_beats(sig, fs)
    limb_rel = {}
    if prefusion is not None and all(l in leads for l in LIMB):
        P = np.asarray(prefusion, float)
        ix = {l: leads.index(l) for l in LIMB}
        for trio, coef in ((("I", "II", "III"), (1, -1, 1)), (("aVR", "aVL", "aVF"), (1, 1, 1))):
            X = np.stack([P[ix[l]] for l in trio])
            m = np.isfinite(X).all(0)
            if m.sum() >= 0.2 * fs:
                r = sum(c * X[k, m] for k, c in enumerate(coef))
                den = float(np.sqrt(np.mean(X[:, m] ** 2)))
                rel = float(np.sqrt(np.mean(r ** 2)) / max(den, 1e-6))
                for l in trio:
                    limb_rel[l] = rel
    out = {}
    for i, l in enumerate(leads):
        q = lead_quality(sig[i], fs, beats)
        q["limb_residual"] = limb_rel.get(l)
        q["score"] = _combine(q, limb_rel.get(l)) if "ausente" not in q["flags"] else 0.0
        if q["score"] < 0.4 and "ausente" not in q["flags"]:
            q["flags"].append("suspeita")
        out[l] = q
    present = [q["score"] for q in out.values() if "ausente" not in q["flags"]]
    return dict(leads=out, beats=len(beats), score=float(np.mean(present)) if present else 0.0)
