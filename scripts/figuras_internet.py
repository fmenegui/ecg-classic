# Uso: git clone https://github.com/Ahus-AIM/open-ecg-digitizer ext/oed; git clone https://github.com/Tereshchenkolab/paper-ecg ext/pecg; python scripts/figuras_internet.py
import sys, pickle, json, numpy as np, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ecgdig.digitize import digitize, Config, LEADS
from ecgdig.plot import plot_ecg
E = __import__("os").environ.get("ECG_EXT", "ext/")  # clones de open-ecg-digitizer (oed/) e paper-ecg (pecg/)
imgs = [("ge_scan", E+"pecg/fullScan.png", "GE/Marquette, scan (paper-ecg)", "3x4 + ritmo V1, II · 25 mm/s"),
        ("ge_rot", E+"pecg/rotatedFullScan.png", "Mesmo GE, scan girado (paper-ecg)", "3x4 + ritmo V1, II · 25 mm/s"),
        ("schiller11", E+"oed/test/test_data/data/ecg_data/11_1.png", "Mortara/Quinton, scan (Open-ECG-Digitizer)", "V1–V6 · 50 mm/s"),
        ("schiller12", E+"oed/test/test_data/data/ecg_data/12_1.png", "Schiller, scan (Open-ECG-Digitizer)", "V1–V6 · 50 mm/s"),
        ("photo0", E+"oed/assets/visual_abstract-img0.png", "Schiller, foto de celular (Open-ECG-Digitizer)", "V1–V6 · 50 mm/s"),
        ("photo1", E+"oed/assets/visual_abstract-img1.png", "Foto com rótulos apagados (Open-ECG-Digitizer)", "provavelmente V1–V6 (mesmo paciente da foto anterior) · 50 mm/s")]
sel = sys.argv[1:]
summ = {}
for key, f, title, truth in imgs:
    if sel and key not in sel: continue
    r = digitize(f, Config(), return_debug=True)
    rn = digitize(f, Config(einthoven=False), return_debug=True)
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    dbg = cv2.cvtColor(r["debug_image"], cv2.COLOR_BGR2RGB)
    rows = r["layout_rows"]
    names = " | ".join(" ".join(rw) for rw in rows)
    li = r["layout_info"] or {}
    fig = plt.figure(figsize=(16, 15.5), dpi=110)
    fig.text(0.02, 0.985, f"{title}", fontsize=15, weight="bold", va="top")
    fig.text(0.02, 0.962, f"Verdade (impresso na folha): {truth}", fontsize=11, va="top", color="#333")
    fig.text(0.02, 0.945, f"Detectado: layout {r['layout']}  [{names}]  ·  {r['speed_mm_per_s']:g} mm/s ({r['speed_source']})  ·  "
             f"{r['duration_s']:g} s  ·  ganho {sorted(set(r['gains'].values()))} mm/mV  ·  confiança do layout {li.get('confidence', 0):.2f}",
             fontsize=10.5, va="top", color="#0b4f9c")
    ax = fig.add_axes([0.02, 0.585, 0.47, 0.32]); ax.imshow(img); ax.axis("off")
    ax.set_title("Imagem original", fontsize=11, loc="left")
    ax = fig.add_axes([0.51, 0.585, 0.47, 0.32]); ax.imshow(dbg); ax.axis("off")
    ax.set_title("Retificada pela grade + traçado extraído (azul)", fontsize=11, loc="left")
    ax = fig.add_axes([0.02, 0.01, 0.96, 0.55])
    plot_ecg(ax, r["signals"], LEADS, layout=rows, speed=r["speed_mm_per_s"],
             title=f"Sinal digitalizado redesenhado a partir dos números (mV, 500 Hz), na mesma velocidade da folha")
    fig.savefig(f"docs/internet/{key}.png"); plt.close(fig)
    m_ = dict(layout=r["layout"], speed=r["speed_mm_per_s"], conf=float(li.get("confidence", 0)))
    ps, pw = rn["per_segment"], rn["per_segment_weight"]
    for l, lst in ps.items():
        if len(lst) >= 2:
            a, b = lst[0], lst[1]; wa, wb = pw[l][0], pw[l][1]
            mu = ~np.isnan(a) & ~np.isnan(b) & (wa > 0.5) & (wb > 0.5)
            if mu.sum() > 100:
                m_[f"dup_{l}"] = dict(r=round(float(np.corrcoef(a[mu], b[mu])[0, 1]), 3),
                                      rms_uV=round(float(1000*np.sqrt(np.mean((a[mu]-b[mu])**2))), 1))
    s = rn["signals"]; mm = ~np.isnan(s[0]) & ~np.isnan(s[1]) & ~np.isnan(s[2])
    if mm.sum() > 100:
        m_["einthoven_uV"] = round(float(1000*np.sqrt(np.mean((s[0]-s[1]+s[2])[mm]**2))), 1)
        m_["rms_II_uV"] = round(float(1000*np.sqrt(np.mean(s[1][mm]**2))), 1)
    summ[key] = m_
    t = np.arange(r["signals"].shape[1]) / r["fs"]
    np.savetxt(f"docs/internet/{key}.csv", np.column_stack([t, r["signals"].T]), delimiter=",",
               header="t," + ",".join(LEADS), comments="", fmt="%.4f")
    print(key, json.dumps(m_), flush=True)
json.dump(summ, open("docs/internet/summary.json", "w"), indent=1)
