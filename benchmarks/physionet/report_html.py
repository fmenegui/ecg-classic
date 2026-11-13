#!/usr/bin/env python3
"""Relatório HTML autocontido da avaliação no PhysioNet (Kaggle).

  python report_html.py <artefato> [--before <artefato_anterior>] [-o saida.html]

Lê results.json, decomposition.json (de diagnose.py), sig/*.npz e img/*.jpg do artefato.
Com --before, mostra a comparação antes/depois por tipo de captura.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TYPE_NAMES = {
    "0001": "render digital", "0003": "scan colorido", "0004": "scan P&B", "0011": "scan com sujeira",
    "0012": "scan escurecido", "0005": "foto de celular", "0006": "foto de tela", "0009": "foto amassada",
    "0010": "foto com sombra",
}
TYPE_ORDER = ["0001", "0003", "0004", "0011", "0012", "0005", "0006", "0009", "0010"]
STEPS = ["rótulos", "cobertura", "origem", "deriva", "deriva local", "ganho", "linha de base"]
STEP_LABEL = {
    "rótulos": "layout / derivação trocada", "cobertura": "trechos vazios", "origem": "origem do tempo", "deriva": "escala horizontal",
    "deriva local": "distorção horizontal local", "ganho": "ganho (mV/mm)", "linha de base": "linha de base",
}
STEP_VAR = {"rótulos": "--s0", "cobertura": "--s1", "origem": "--s2", "deriva": "--s3", "deriva local": "--s4",
            "ganho": "--s5", "linha de base": "--s6"}

CSS = """
:root{
  /* papel milimetrado de ECG: grade vermelha clara, tinta grafite, verdade em azul */
  --bg:#f6f7f8; --panel:#ffffff; --fg:#1d2429; --muted:#5d6871; --line:#d9dee2;
  --grid1:#f3d6d6; --grid5:#e7a9a9; --gt:#1f5fbf; --pred:#c2272d; --accent:#b0262c;
  --good:#1e7a46; --bad:#b0262c;
  --s0:#c2272d; --s1:#8a96a3; --s2:#d18b2c; --s3:#3f7fd0; --s4:#1f4f99; --s5:#3c9a6a; --s6:#9a6bbf;
  --f-head:"Archivo",system-ui,sans-serif; --f-text:"IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#14181b; --panel:#1b2024; --fg:#e4e8eb; --muted:#9aa5ae; --line:#2e353a;
  --grid1:#3a2426; --grid5:#5e3236; --gt:#6fa6ff; --pred:#ff6b6f; --accent:#ff6b6f;
  --good:#5fd394; --bad:#ff7a7e;
  --s0:#ff6b6f; --s1:#7d8894; --s2:#e6a54a; --s3:#5c9cf0; --s4:#9cbcf5; --s5:#57c08a; --s6:#bb93de; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#14181b; --panel:#1b2024; --fg:#e4e8eb; --muted:#9aa5ae; --line:#2e353a;
  --grid1:#3a2426; --grid5:#5e3236; --gt:#6fa6ff; --pred:#ff6b6f; --accent:#ff6b6f;
  --good:#5fd394; --bad:#ff7a7e;
  --s0:#ff6b6f; --s1:#7d8894; --s2:#e6a54a; --s3:#5c9cf0; --s4:#9cbcf5; --s5:#57c08a; --s6:#bb93de; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.55 var(--f-text);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:28px 64px}
main{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:40px}
h1,h2,h3{font-family:var(--f-head);text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:2rem;font-weight:700} h2{font-size:1.35rem;font-weight:650} h3{font-size:1rem;font-weight:650}
p{margin:0;max-width:68ch} section{display:flex;flex-direction:column;gap:14px}
.eyebrow{font:600 .72rem/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.muted{color:var(--muted)} .mono,td.n,th.n{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px;display:flex;flex-direction:column;gap:4px}
.kpi b{font:600 1.6rem/1.1 var(--f-mono);font-variant-numeric:tabular-nums}
.kpi span{font-size:.82rem;color:var(--muted)}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:.78rem} td.n,th.n{text-align:right}
tr:last-child td{border-bottom:none}
.up{color:var(--good)} .down{color:var(--bad)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.82rem;color:var(--muted)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
svg text{fill:var(--muted);font:11px var(--f-mono)}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;overflow-x:auto}
.finding{display:grid;grid-template-columns:minmax(0,1fr);gap:6px;padding:14px 16px;border-left:3px solid var(--accent);background:var(--panel);border-radius:0 6px 6px 0}
.finding h3 span{font-family:var(--f-mono);font-weight:500;color:var(--accent);margin-left:6px}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden;display:flex;flex-direction:column;min-width:0}
.card img{display:block;width:100%;height:auto;border-bottom:1px solid var(--line)}
.card .meta{padding:10px 12px;display:flex;flex-direction:column;gap:6px;font-size:.82rem}
.card .meta b{font-family:var(--f-mono)}
.strip{display:block;width:100%;height:auto}
.tabs{display:flex;flex-wrap:wrap;gap:6px}
.tabs button{font:500 .8rem var(--f-text);border:1px solid var(--line);background:var(--panel);color:var(--fg);padding:5px 10px;border-radius:4px;cursor:pointer}
.tabs button[aria-pressed="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.finding ul{margin:0;padding-left:1.2em;max-width:68ch}
ol.fix{margin:0;padding-left:1.2em;display:flex;flex-direction:column;gap:8px;max-width:72ch}
"""


def _npz_snr(art: Path):
    """SNR de cada imagem recalculado dos sinais salvos (250 Hz), com a mesma métrica para as
    duas rodadas de uma comparação (o results.json antigo usava passo de alinhamento grosso)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_snr import _snr_record
    out = {}
    for f in sorted((art / "sig").glob("*.npz")):
        z = np.load(f)
        rid, t = f.stem.split("-")
        out[(rid, t)] = _snr_record(z["gt"].astype(float), z["pred"].astype(float), float(z["fs"]))[0]
    return out


def _b64(p: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def _mean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _f(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _decomp_chart(dec_by_type):
    """Barras empilhadas horizontais: SNR atual + ganho de cada correção oráculo."""
    types = [t for t in TYPE_ORDER if t in dec_by_type]
    W, lh, left, right = 900, 30, 170, 20
    H = 30 + lh * len(types) + 26
    lo = float(np.floor(min(0.0, min(dec_by_type[t]["métrica"] for t in types)) / 5) * 5)
    hi = max(dec_by_type[t]["linha de base"] for t in types)
    hi = float(np.ceil(max(hi, 20) / 5) * 5)
    sx = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="min-width:640px" role="img" aria-label="Decomposição do erro por tipo">']
    for v in range(int(lo), int(hi) + 1, 5):
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{H-22}" stroke="var(--line)"/>'
                   f'<text x="{x:.1f}" y="{H-8}" text-anchor="middle">{v} dB</text>')
    for k, t in enumerate(types):
        d = dec_by_type[t]
        y = 24 + k * lh
        out.append(f'<text x="{left-8}" y="{y+15}" text-anchor="end" style="fill:var(--fg)">{t} {TYPE_NAMES.get(t, "")}</text>')
        x0 = sx(lo)
        x1 = sx(d["métrica"])
        out.append(f'<rect x="{x0:.1f}" y="{y+3}" width="{max(0.5, x1-x0):.1f}" height="{lh-10}" fill="var(--fg)" opacity=".85"/>')
        prev = d["métrica"]
        for s in STEPS:
            cur = max(prev, d[s])
            if cur - prev > 0.02:
                out.append(f'<rect x="{sx(prev):.1f}" y="{y+3}" width="{sx(cur)-sx(prev):.1f}" height="{lh-10}" fill="var({STEP_VAR[s]})">'
                           f'<title>{t} · {STEP_LABEL[s]}: +{cur-prev:.2f} dB</title></rect>')
            prev = cur
        out.append(f'<text x="{sx(prev)+6:.1f}" y="{y+15}">{d["métrica"]:.1f} → {prev:.1f}</text>')
    out.append("</svg>")
    leg = ['<span><i style="background:var(--fg);opacity:.85"></i>SNR atual</span>'] + [
        f'<span><i style="background:var({STEP_VAR[s]})"></i>{STEP_LABEL[s]}</span>' for s in STEPS]
    return '<div class="chart">' + "".join(out) + '</div><div class="legend">' + "".join(leg) + "</div>"


def _ecg_strip(g, p, fs, lead, win=None):
    """Uma derivação sobre papel milimetrado (25 mm/s, 10 mm/mV): verdade × ecgdig."""
    m = np.isfinite(g)
    if m.sum() < 10:
        return ""
    idx = np.nonzero(m)[0]
    a, b = idx[0], idx[-1] + 1
    if win is not None:
        a, b = max(a, win[0]), min(b, win[1])
    t0 = a / fs
    dur = (b - a) / fs
    mm_w = dur * 25
    gg = g[a:b] - np.nanmedian(g[a:b])
    pp = p[a:b] - (np.nanmedian(p[a:b]) if np.isfinite(p[a:b]).any() else 0)
    both = np.concatenate([gg[np.isfinite(gg)], pp[np.isfinite(pp)]])
    lo = min(-0.5, np.percentile(both, 1) - 0.15)
    hi = max(0.5, np.percentile(both, 99) + 0.15)
    if hi - lo > 3.5:  # mantém a faixa legível; o que sai fica cortado na borda
        c = float(np.median(gg[np.isfinite(gg)]))
        lo, hi = max(lo, c - 1.75), min(hi, c + 1.75)
    mm_h = (hi - lo) * 10
    sc = 6.0  # px por mm
    W, H = mm_w * sc, mm_h * sc
    out = [f'<svg class="strip" viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="Derivação {lead}">',
           f'<rect width="{W:.0f}" height="{H:.0f}" fill="var(--panel)"/>']
    for k in range(int(mm_w) + 1):
        out.append(f'<line x1="{k*sc}" y1="0" x2="{k*sc}" y2="{H:.0f}" stroke="var(--grid5)" stroke-width="{0.8 if k % 5 == 0 else 0}"/>'
                   if k % 5 == 0 else f'<line x1="{k*sc}" y1="0" x2="{k*sc}" y2="{H:.0f}" stroke="var(--grid1)" stroke-width=".4"/>')
    for k in range(int(mm_h) + 1):
        w_ = 0.8 if k % 5 == 0 else 0.4
        c = "var(--grid5)" if k % 5 == 0 else "var(--grid1)"
        out.append(f'<line x1="0" y1="{H-k*sc:.1f}" x2="{W:.0f}" y2="{H-k*sc:.1f}" stroke="{c}" stroke-width="{w_}"/>')

    def poly(v, color, wdt):
        pts, segs = [], []
        step = max(1, int(fs / 125))
        for i in range(0, len(v), step):
            if np.isfinite(v[i]):
                pts.append(f"{i/fs*25*sc:.1f},{H-(min(max(v[i], lo), hi)-lo)*10*sc:.1f}")
            elif pts:
                segs.append(pts)
                pts = []
        if pts:
            segs.append(pts)
        return "".join(f'<polyline points="{" ".join(s)}" fill="none" stroke="{color}" stroke-width="{wdt}" stroke-linejoin="round"/>' for s in segs)

    out.append(poly(gg, "var(--gt)", 2.6))
    out.append(poly(pp, "var(--pred)", 1.7))
    out.append(f'<text x="6" y="16" style="fill:var(--fg);font-weight:600;font-size:13px">{lead}</text>'
               f'<text x="{W-6:.0f}" y="16" text-anchor="end" style="font-size:12px">{t0:.1f}–{t0+dur:.1f} s</text></svg>')
    return "".join(out)


def _pick_examples(res, dec, k_per_type=2):
    """Por tipo: a imagem mediana e a pior (com sinal salvo)."""
    by = defaultdict(list)
    for r in res:
        if r.get("ok") and np.isfinite(r.get("snr", np.nan)):
            by[r["type"]].append(r)
    picks = []
    for t in TYPE_ORDER:
        v = sorted(by.get(t, []), key=lambda r: r["snr"])
        if not v:
            continue
        sel = [v[len(v) // 2]]
        if k_per_type > 1 and v[0] is not sel[0]:
            sel.append(v[0])
        picks += sel
    return picks


def _worst_lead(r):
    pl = r.get("per_lead") or {}
    pl = {k: v for k, v in pl.items() if k != "II" or len(pl) == 1}
    return min(pl, key=pl.get) if pl else "II"


def build(art: Path, before: Path | None, out: Path, findings_md: Path | None = None):
    R = json.loads((art / "results.json").read_text())
    res = R["results"]
    dec = json.loads((art / "decomposition.json").read_text()) if (art / "decomposition.json").exists() else []
    B = json.loads((before / "results.json").read_text())["results"] if before else None
    if B:  # mesma métrica nas duas rodadas
        na, nb = _npz_snr(art), _npz_snr(before)
        for r in res:
            if (r["id"], r["type"]) in na:
                r["snr"] = na[(r["id"], r["type"])]
        for r in B:
            if (r["id"], r["type"]) in nb:
                r["snr"] = nb[(r["id"], r["type"])]

    by = defaultdict(list)
    for r in res:
        by[r["type"]].append(r)
    ok = [r for r in res if r.get("ok")]
    allv = [r["snr"] for r in ok if np.isfinite(r["snr"])]
    n_exc = len(res) - len(ok)
    caps = [r["snr"] for r in ok if r["type"] != "0001" and np.isfinite(r["snr"])]
    rend = [r["snr"] for r in ok if r["type"] == "0001" and np.isfinite(r["snr"])]
    lay_ok = np.mean([r.get("layout") == "3x4+1" for r in ok]) if ok else np.nan
    with_exc = (sum(allv) + 0.0 * n_exc) / max(1, len(allv) + n_exc)

    H = [f"<title>ecgdig no PhysioNet</title>",
         '<meta name="claude:viewport-fit-sides" content="cover">',
         '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
         f"<style>{CSS}</style><main>"]
    H.append(f'<header style="display:flex;flex-direction:column;gap:10px"><div class="eyebrow">PhysioNet · Digitization of ECG Images · treino</div>'
             f'<h1>ecgdig no PhysioNet</h1><p class="muted">{len({r["id"] for r in res})} registros × 9 capturas = {len(res)} imagens. '
             f'SNR por imagem somando as 12 derivações (reimplementação da métrica: até ±0,2 s e média removida por derivação). '
             f'Configuração: {html.escape(str(R.get("config") or "padrão"))}.</p></header>')
    H.append('<div class="kpis">'
             f'<div class="kpi"><b>{_f(np.mean(allv))} dB</b><span>média geral ({_f(with_exc)} com as {n_exc} exceções como 0 dB)</span></div>'
             f'<div class="kpi"><b>{_f(_mean(rend))} dB</b><span>render digital (0001)</span></div>'
             f'<div class="kpi"><b>{_f(_mean(caps))} dB</b><span>scans e fotos (8 tipos)</span></div>'
             f'<div class="kpi"><b>{100*lay_ok:.0f}%</b><span>layout 3x4+1 reconhecido</span></div></div>')

    # tabela por tipo
    rows = []
    bby = defaultdict(list)
    if B:
        for r in B:
            if r.get("ok"):
                bby[r["type"]].append(r["snr"])
    dby = defaultdict(list)
    for d in dec:
        dby[d["type"]].append(d)
    for t in TYPE_ORDER:
        v = [r for r in by.get(t, []) if r.get("ok")]
        if not v:
            continue
        s = [r["snr"] for r in v]
        lay = np.mean([r.get("layout") == "3x4+1" for r in v])
        da = np.mean([bool(r.get("duration_assumed")) for r in v])
        cells = [f"<td><b>{t}</b> <span class='muted'>{TYPE_NAMES.get(t, '')}</span></td>", f"<td class='n'>{len(v)}</td>",
                 f"<td class='n'><b>{_f(np.mean(s))}</b></td>", f"<td class='n'>{_f(np.median(s))}</td>",
                 f"<td class='n'>{_f(_mean([r['snr_raw'] for r in v]))}</td>",
                 f"<td class='n'>{100*lay:.0f}%</td>", f"<td class='n'>{100*da:.0f}%</td>"]
        if B:
            b = _mean(bby.get(t, []))
            dlt = np.mean(s) - b
            cells.insert(3, f"<td class='n'>{_f(b)}</td><td class='n {'up' if dlt > 0.05 else 'down' if dlt < -0.05 else ''}'>{dlt:+.2f}</td>")
        if dby.get(t):
            ub = _mean([d["linha de base"] for d in dby[t]])
            cells.append(f"<td class='n'>{_f(ub)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    hdr = ["Tipo", "n", "SNR médio", "mediana", "SNR bruto", "layout certo", "duração suposta"]
    hcls = ["", "n", "n", "n", "n", "n", "n"]
    if B:
        hdr[3:3] = ["antes", "Δ"]
        hcls[3:3] = ["n", "n"]
    if dec:
        hdr.append("teto (oráculo)")
        hcls.append("n")
    H.append('<section><div class="eyebrow">Resultado</div><h2>SNR por tipo de captura</h2>'
             '<div class="tw"><table><thead><tr>' + "".join(f"<th class='{c}'>{h}</th>" for h, c in zip(hdr, hcls)) +
             "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
             '<p class="muted">“Teto (oráculo)” é o SNR depois de corrigir com o gabarito todas as fontes de erro medidas abaixo; '
             "o que sobra é erro de forma (traçado, picos cortados, derivações trocadas).</p></section>")

    def decomp_block(dec_, eyebrow, title, lead):
        by_ = defaultdict(list)
        for d in dec_:
            by_[d["type"]].append(d)
        agg = {t: {k: _mean([d[k] for d in v]) for k in ["métrica"] + STEPS} for t, v in by_.items()}
        tot = defaultdict(list)
        for d in dec_:
            prev = d["métrica"]
            for s_ in STEPS:
                cur = max(prev, d[s_])
                tot[s_].append(cur - prev)
                prev = cur
        return (f'<section><div class="eyebrow">{eyebrow}</div><h2>{title}</h2><p>{lead}</p>' + _decomp_chart(agg) +
                '<div class="tw"><table><thead><tr><th>Fonte de erro</th><th class="n">perda média</th>'
                '<th class="n">imagens com perda &gt; 1 dB</th></tr></thead><tbody>' +
                "".join(f"<tr><td><i style='display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;background:var({STEP_VAR[s_]})'></i>{STEP_LABEL[s_]}</td>"
                        f"<td class='n'>{np.mean(tot[s_]):.2f} dB</td><td class='n'>{100*np.mean(np.array(tot[s_]) > 1):.0f}%</td></tr>"
                        for s_ in sorted(STEPS, key=lambda s_: -np.mean(tot[s_]))) + "</tbody></table></div></section>")

    how = ("Cada correção usa o gabarito só para o seu tipo de erro, cumulativamente e nesta ordem; "
           "o segmento colorido é quanto SNR aquele erro custa. A barra escura é o SNR medido.")
    dec_b = json.loads((before / "decomposition.json").read_text()) if before and (before / "decomposition.json").exists() else None
    if dec_b:
        H.append(decomp_block(dec_b, "Diagnóstico · antes", "Para onde ia o SNR", how))
    for fm in (findings_md if isinstance(findings_md, (list, tuple)) else [findings_md]):
        if fm and Path(fm).exists():
            H.append(Path(fm).read_text())
    if dec:
        H.append(decomp_block(dec, "Diagnóstico · agora" if dec_b else "Diagnóstico",
                              "O que ainda custa SNR" if dec_b else "Para onde vai o SNR", how))

    # por derivação
    perl = defaultdict(list)
    for r in ok:
        for l, v in (r.get("per_lead") or {}).items():
            perl[l].append(v)
    if perl:
        mx = max(abs(np.mean(perl[l])) for l in perl) or 1
        bars = []
        W = 900
        for k, l in enumerate(LEADS):
            if l not in perl:
                continue
            v = np.mean(perl[l])
            x0 = 60 + 380
            w = v / max(mx, 1) * 380
            bars.append(f'<text x="50" y="{20+k*22}" text-anchor="end" style="fill:var(--fg)">{l}</text>'
                        f'<rect x="{min(x0, x0+w):.1f}" y="{8+k*22}" width="{abs(w):.1f}" height="15" fill="var(--s3)"/>'
                        f'<text x="{x0+w+(6 if w >= 0 else -6):.1f}" y="{20+k*22}" text-anchor="{"start" if w >= 0 else "end"}">{v:.2f} dB</text>')
        H.append('<section><div class="eyebrow">Por derivação</div><h2>SNR médio de cada derivação</h2>'
                 f'<div class="chart"><svg viewBox="0 0 {W} {12*22+14}" width="100%" style="min-width:560px" role="img" aria-label="SNR por derivação">'
                 f'<line x1="440" y1="4" x2="440" y2="{12*22+6}" stroke="var(--line)"/>' + "".join(bars) + "</svg></div></section>")

    # galeria
    picks = _pick_examples(res, dec)
    cards = []
    dmap = {(d["id"], d["type"]): d for d in dec}
    for r in picks:
        rid, t = r["id"], r["type"]
        imgp = art / "img" / f"{rid}-{t}.jpg"
        sigp = art / "sig" / f"{rid}-{t}.npz"
        if not imgp.exists() or not sigp.exists():
            continue
        z = np.load(sigp)
        G, P, fs = z["gt"].astype(float), z["pred"].astype(float), float(z["fs"])
        wl = _worst_lead(r)
        gi = G[LEADS.index(wl)]
        ix = np.nonzero(np.isfinite(gi))[0]
        win = (ix[0], ix[-1] + 1) if len(ix) else None
        strips = _ecg_strip(G[LEADS.index("II")], P[LEADS.index("II")], fs, "II (ritmo)", win) + \
            _ecg_strip(gi, P[LEADS.index(wl)], fs, wl, win)
        d = dmap.get((rid, t))
        dd = ""
        if d:
            prev, parts = d["métrica"], []
            for s in STEPS:
                cur = max(prev, d[s])
                if cur - prev > 0.5:
                    parts.append(f"{STEP_LABEL[s]} +{cur-prev:.1f}")
                prev = cur
            dd = "<span class='muted'>perdas: " + (", ".join(parts) or "nenhuma acima de 0,5 dB") + "</span>"
        cards.append(f'<article class="card" data-type="{t}"><img loading="lazy" src="{_b64(imgp)}" alt="Sobreposição {rid}-{t}">'
                     f'<div class="meta"><div><b>{rid}-{t}</b> · {TYPE_NAMES.get(t, "")} · <b>{r["snr"]:.1f} dB</b></div>'
                     f'<div class="muted">layout {html.escape(str(r.get("layout")))} · {r.get("speed")} mm/s · ganhos {r.get("gains")}'
                     f'{" · duração suposta" if r.get("duration_assumed") else ""}</div>{dd}</div>{strips}</article>')
    tabs = '<button type="button" aria-pressed="true" data-f="">todos</button>' + "".join(
        f'<button type="button" aria-pressed="false" data-f="{t}">{t} {TYPE_NAMES[t]}</button>' for t in TYPE_ORDER if t in by)
    H.append('<section><div class="eyebrow">Exemplos</div><h2>Imagem mediana e pior imagem de cada tipo</h2>'
             '<p class="muted">Sobreposição do ecgdig e, embaixo, a derivação II e a pior derivação da imagem em papel '
             '25 mm/s · 10 mm/mV: <span style="color:var(--gt)">azul = sinal verdadeiro</span>, '
             '<span style="color:var(--pred)">vermelho = ecgdig</span>.</p>'
             f'<div class="tabs" role="group" aria-label="Filtrar por tipo">{tabs}</div><div class="gal" id="gal">{"".join(cards)}</div></section>')
    exc = [r for r in res if not r.get("ok")]
    if exc:
        H.append('<section><div class="eyebrow">Falhas</div><h2>Exceções</h2><div class="tw"><table><tbody>' +
                 "".join(f"<tr><td class='mono'>{r['id']}-{r['type']}</td><td class='mono'>{html.escape(str(r.get('error'))[:140])}</td></tr>" for r in exc) +
                 "</tbody></table></div></section>")
    H.append("</main><script>document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{"
             "document.querySelectorAll('.tabs button').forEach(x=>x.setAttribute('aria-pressed',x===b));"
             "const f=b.dataset.f;document.querySelectorAll('#gal .card').forEach(c=>c.hidden=!!f&&c.dataset.type!==f);}));</script>")
    out.write_text("\n".join(H))
    print("escrito", out, f"{out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("art", type=Path)
    ap.add_argument("--before", type=Path)
    ap.add_argument("--findings", type=Path, nargs="*", default=[], help="trechos HTML com o diagnóstico escrito")
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()
    build(a.art, a.before, a.out or a.art / "report.html", a.findings)
