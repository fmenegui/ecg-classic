"""HTML report of the LITFL benchmark: summary + gallery of the pages with the extracted trace.

Single file (embedded data); the images sit in img/ next to it. Opens straight from disk.
"""
from __future__ import annotations

import html
import json
from datetime import date

LABELS = {
    "layout": "Layout",
    "rhythm": "Rhythm strips",
    "speed": "Speed",
    "duration": "Duration",
    "gain": "Gain",
}

METRICS = [
    ("layout_acc", "Correct layout", "pct"),
    ("rhythm_acc", "Correct rhythm strips", "pct"),
    ("gain_acc", "Correct gain", "pct"),
    ("duration_acc", "Correct duration", "pct"),
    ("speed_acc", "Correct speed", "pct"),
    ("all_structural_ok", "Layout + speed + duration + gain", "pct"),
    ("coverage_median", "Coverage (median)", "pct"),
    ("einthoven_ratio_median", "Einthoven before fusion (median)", "num"),
    ("dup_II_nrmse_median", "Rhythm II vs grid II, NRMSE (median)", "num"),
]

CSS = """
:root {
  --bg: #f4f5f7; --surface: #ffffff; --fg: #1b2230; --muted: #5b6577; --line: #dde1e8; --ink: #1d5fbf;
  --ok: #1f7a4d; --ok-bg: #e3f3ea; --bad: #b42318; --bad-bg: #fde4e1; --warn: #9a5b00; --warn-bg: #fdf0d8;
  --font-display: "Space Grotesk", "Segoe UI", system-ui, sans-serif;
  --font-body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11151c; --surface: #19202a; --fg: #e6e9ef; --muted: #9aa4b5; --line: #2a3340; --ink: #6ea8ff;
  --ok: #5fd39a; --ok-bg: #143326; --bad: #ff8a7e; --bad-bg: #3d1a17; --warn: #f2bb5c; --warn-bg: #3a2b10; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #11151c; --surface: #19202a; --fg: #e6e9ef; --muted: #9aa4b5; --line: #2a3340; --ink: #6ea8ff;
  --ok: #5fd39a; --ok-bg: #143326; --bad: #ff8a7e; --bad-bg: #3d1a17; --warn: #f2bb5c; --warn-bg: #3a2b10; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.5; margin: 0; }
.wrap { max-width: 1320px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 28px 48px; }
h1 { font-family: var(--font-display); font-weight: 600; font-size: clamp(26px, 3.4vw, 36px); margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
h2 { font-family: var(--font-display); font-weight: 600; font-size: 19px; margin: 28px 0 10px; }
.lede { color: var(--muted); max-width: 72ch; margin: 10px 0 0; }
.lede b { color: var(--fg); font-weight: 600; }
.meta { font-family: var(--font-mono); font-size: 12px; color: var(--muted); margin-top: 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; margin-top: 18px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.tile .n { font-family: var(--font-display); font-size: 26px; font-weight: 600; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tile .l { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.tile .d { font-family: var(--font-mono); font-size: 12px; margin-top: 4px; }
.up { color: var(--ok); } .down { color: var(--bad); } .same { color: var(--muted); }
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }
td.num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg); padding-block: 12px; margin-top: 18px; border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); }
.chip { font: inherit; font-size: 14px; border: 1px solid var(--line); background: var(--surface); color: var(--fg); border-radius: 999px; padding: 6px 12px; cursor: pointer; display: inline-flex; gap: 6px; align-items: center; }
.chip[aria-pressed="true"] { border-color: var(--ink); box-shadow: inset 0 0 0 1px var(--ink); }
.chip .c { font-family: var(--font-mono); font-size: 12px; color: var(--muted); }
.chip:focus-visible, .card:focus-visible, .close:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.search { margin-left: auto; font: inherit; font-size: 14px; padding: 6px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); color: var(--fg); width: 200px; max-width: 100%; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; margin-top: 16px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; cursor: zoom-in; text-align: left; padding: 0; font: inherit; color: inherit; min-width: 0; }
.card img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #fff; border-bottom: 1px solid var(--line); }
.body { padding: 10px 12px 12px; display: grid; gap: 6px; }
.top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.id { font-family: var(--font-mono); font-weight: 500; font-size: 13px; }
.pill { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; padding: 2px 8px; border-radius: 999px; }
.pill.ok { color: var(--ok); background: var(--ok-bg); } .pill.error { color: var(--warn); background: var(--warn-bg); } .pill.failure { color: var(--bad); background: var(--bad-bg); }
.miss { font-size: 11.5px; padding: 1px 7px; border-radius: 6px; color: var(--bad); background: var(--bad-bg); }
.dx { margin-left: auto; font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 45%; }
.kv { font-family: var(--font-mono); font-size: 12px; color: var(--muted); display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-variant-numeric: tabular-nums; }
.kv b { color: var(--fg); font-weight: 500; }
.kv .x { color: var(--bad); }
.note { font-size: 13px; margin: 0; color: var(--muted); }
.empty { color: var(--muted); padding: 32px 0; }
dialog { border: 0; padding: 0; background: var(--surface); color: var(--fg); border-radius: 12px; width: min(1200px, 96vw); max-height: 92vh; overflow: auto; }
dialog::backdrop { background: rgba(10, 14, 20, 0.72); }
.dlg { display: grid; gap: 10px; padding: 14px; }
.dlg img { width: 100%; height: auto; background: #fff; border-radius: 6px; }
.dlg .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.dlg a { color: var(--ink); }
.close { margin-left: auto; font: inherit; border: 1px solid var(--line); background: var(--surface); color: var(--fg); border-radius: 8px; padding: 4px 10px; cursor: pointer; }
footer { margin-top: 28px; color: var(--muted); font-size: 13px; max-width: 85ch; }
@media (max-width: 520px) { .search { margin-left: 0; width: 100%; } .dx { max-width: 100%; margin-left: 0; } }
@media (prefers-reduced-motion: no-preference) { .card { transition: border-color .15s; } .card:hover { border-color: var(--ink); } }
"""

JS = """
const D = JSON.parse(document.getElementById("data").textContent);
const L = { ok: "Correct", error: "With errors", failure: "Exception" };
const F = D.fields;
let filter = "all", q = "";
try { filter = localStorage.getItem("litfl-bench-filter") || "all"; } catch (e) {}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = v => Array.isArray(v) ? v.join(" / ") : (v ?? "—");
function rowsKV(it) {
  if (!it.ok) return `<span>error</span><b class="x">${esc(it.error)}</b>`;
  const r = it.ref, bad = new Set(it.failed);
  const line = (k, got, want) => `<span>${F[k]}</span><b${bad.has(k) ? ' class="x"' : ""}>${esc(fmt(got))}${bad.has(k) ? " ≠ " + esc(fmt(want)) : ""}</b>`;
  return [line("layout", it.layout, r.layout), line("rhythm", it.rhythm.length ? it.rhythm : "—", r.rhythm),
          line("speed", it.speed + " mm/s", r.speed + " mm/s"), line("duration", it.duration + " s", r.duration + " s"),
          line("gain", it.gains.map(g => g + " mm/mV"), (r.gain || []).map(g => g + " mm/mV"))].join("");
}
function card(it) {
  const miss = it.failed.map(k => `<span class="miss">${F[k]}</span>`).join("");
  return `<button class="card" data-id="${it.id}" aria-label="Open ${it.id}">
    ${it.image ? `<img loading="lazy" src="${it.image}" alt="Page ${it.id} rectified with the extracted trace">` : ""}
    <div class="body"><div class="top"><span class="id">${it.id}</span><span class="pill ${it.status}">${L[it.status]}</span>${miss}<span class="dx">${esc(it.dx.join(", "))}</span></div>
    <div class="kv">${rowsKV(it)}</div>${it.note ? `<p class="note">${esc(it.note)}</p>` : ""}</div></button>`;
}
function render() {
  const s = q.trim().toLowerCase();
  const list = D.items.filter(it => (filter === "all" || it.status === filter) &&
    (!s || it.id.includes(s) || it.dx.join(" ").toLowerCase().includes(s) || it.failed.map(k => F[k].toLowerCase()).join(" ").includes(s)));
  document.getElementById("grid").innerHTML = list.map(card).join("");
  document.getElementById("empty").hidden = list.length > 0;
  document.querySelectorAll(".chip").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.f === filter)));
}
function open(id) {
  const it = D.items.find(x => x.id === id); if (!it) return;
  document.getElementById("d-id").textContent = it.id;
  const p = document.getElementById("d-pill"); p.className = "pill " + it.status; p.textContent = L[it.status];
  document.getElementById("d-kv").innerHTML = rowsKV(it);
  const extra = [];
  if (it.confidence != null) extra.push("layout confidence " + it.confidence.toFixed(2));
  if (it.einthoven_ratio != null) extra.push("Einthoven " + it.einthoven_ratio.toFixed(2));
  if (it.dup_II_nrmse != null) extra.push("rhythm vs grid " + it.dup_II_nrmse.toFixed(2));
  if (it.warning) extra.push(it.warning);
  document.getElementById("d-note").textContent = [it.note, extra.join(" · ")].filter(Boolean).join(" — ");
  const img = document.getElementById("d-img"); img.src = it.image || ""; img.hidden = !it.image;
  const a = document.getElementById("d-src"); a.href = it.url || "#"; a.hidden = !it.url;
  document.getElementById("dlg").showModal();
}
const cnt = { all: D.items.length, ok: 0, error: 0, failure: 0 };
D.items.forEach(it => cnt[it.status]++);
document.querySelectorAll("[data-c]").forEach(e => e.textContent = cnt[e.dataset.c]);
document.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => { filter = b.dataset.f; try { localStorage.setItem("litfl-bench-filter", filter); } catch (e) {} render(); }));
document.getElementById("q").addEventListener("input", e => { q = e.target.value; render(); });
document.getElementById("grid").addEventListener("click", e => { const c = e.target.closest(".card"); if (c) open(c.dataset.id); });
document.getElementById("d-close").addEventListener("click", () => document.getElementById("dlg").close());
document.getElementById("dlg").addEventListener("click", e => { if (e.target.id === "dlg") e.target.close(); });
render();
"""


def _fmt(v, kind):
    if v is None:
        return "—"
    return f"{100 * v:.1f} %" if kind == "pct" else f"{v:.2f}"


def write_html(path, S, results, ref, cmp=None, cfg_kw=None):
    items = []
    for x in results:
        rf = ref.get(x["id"], {})
        failed = [k for k in LABELS if x.get("check", {}).get(k) is False]
        status = "failure" if not x["ok"] else ("error" if failed else "ok")
        items.append(dict(
            id=x["id"], status=status, failed=failed, image=x.get("image"), url=rf.get("url"),
            dx=[d.replace("_", " ") for d in rf.get("diagnoses", [])], note=rf.get("note", "") if failed or not x["ok"] else "",
            ok=x["ok"], error=x.get("error"), layout=x.get("layout"), rhythm=x.get("rhythm", []), speed=x.get("speed"),
            duration=x.get("duration"), gains=x.get("gains", []), confidence=x.get("confidence"), warning=x.get("warning"),
            einthoven_ratio=x.get("einthoven_ratio"), dup_II_nrmse=x.get("dup_II_nrmse"),
            ref=dict(layout=rf.get("layout"), rhythm=rf.get("rhythm"), speed=rf.get("speed"),
                     duration=rf.get("duration"), gain=rf.get("gain"))))
    # tiles
    tiles = []
    for key, lab, kind in METRICS:
        v = S.get(key)
        d = ""
        if cmp and cmp.get(key) is not None and v is not None and kind == "pct":
            delta = v - cmp[key]
            cls = "up" if delta > 1e-9 else ("down" if delta < -1e-9 else "same")
            arrow = "▲" if delta > 1e-9 else ("▼" if delta < -1e-9 else "=")
            d = f'<div class="d {cls}">{arrow} before {_fmt(cmp[key], kind)}</div>'
        tiles.append(f'<div class="tile"><div class="n">{_fmt(v, kind)}</div><div class="l">{html.escape(lab)}</div>{d}</div>')
    # errors per field
    by_field = {k: [it["id"] for it in items if k in it["failed"]] for k in LABELS}
    rows = "".join(f"<tr><td>{LABELS[k]}</td><td class='num'>{len(v)}</td><td class='num'>{' '.join(v) or '—'}</td></tr>"
                   for k, v in by_field.items())
    cfg = ", ".join(f"{k}={v}" for k, v in (cfg_kw or {}).items()) or "default"
    data = dict(items=items, fields=LABELS)
    # outside the f-string: Python 3.11 (the CI one) does not accept a backslash inside the expression
    data_json = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>LITFL benchmark</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap">
<style>{CSS}</style></head>
<body><div class="wrap">
<header>
<h1>LITFL benchmark</h1>
<p class="lede">The classical digitizer <b>ecgdig</b> on <b>{S['images']} real ECGs</b> from Life in the Fast Lane (scans and photos from various sources). There is no true signal: the structure (layout, rhythm-strip names, speed, duration, gain) is compared with a hand-reviewed reference, and the signal is judged by its internal consistency. Each card shows the page rectified by the grid with the <b style="color:var(--ink)">extracted trace in blue</b>.</p>
<div class="meta">{date.today().isoformat()} · Config: {html.escape(cfg)} · {S['ran']}/{S['images']} processed · {S['seconds_mean']} s per image</div>
</header>
<div class="tiles">{''.join(tiles)}</div>
<h2>Errors per field</h2>
<div class="tablewrap"><table><thead><tr><th>Field</th><th>Images</th><th>Which</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="bar" role="toolbar" aria-label="Filters">
<button class="chip" data-f="all" id="f-all">All <span class="c" data-c="all"></span></button>
<button class="chip" data-f="ok" id="f-ok">Correct <span class="c" data-c="ok"></span></button>
<button class="chip" data-f="error" id="f-error">With errors <span class="c" data-c="error"></span></button>
<button class="chip" data-f="failure" id="f-failure">Exception <span class="c" data-c="failure"></span></button>
<input class="search" id="q" type="search" placeholder="No., diagnosis or field" aria-label="Search">
</div>
<div class="grid" id="grid"></div>
<p class="empty" id="empty" hidden>No image with this filter.</p>
<footer>Einthoven: |I − II + III| / RMS(I, II, III) on the grid measurements before fusion (0 = perfect). Rhythm vs grid: NRMSE between the rhythm strip II and the grid II over the same stretch; in several LITFL montages the strip is not simultaneous, so only the change between versions is informative. Reference annotated visually on 12/11/2026 (benchmarks/litfl/reference.json). Images: litfl.com.</footer>
</div>
<dialog id="dlg" aria-label="Enlarged image"><div class="dlg">
<div class="row"><span class="id" id="d-id"></span><span class="pill" id="d-pill"></span><button class="close" id="d-close">Close</button></div>
<div class="kv" id="d-kv"></div><p class="note" id="d-note"></p>
<img id="d-img" alt=""><div class="row"><a id="d-src" href="#" target="_blank" rel="noopener">Original image on LITFL</a></div>
</div></dialog>
<script type="application/json" id="data">{data_json}</script>
<script>{JS}</script>
</body></html>
"""
    path.write_text(page, encoding="utf-8")
