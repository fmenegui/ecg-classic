import sys, time, pickle, traceback
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from ecgdig.digitize import digitize, Config
E = __import__("os").environ.get("ECG_EXT", "ext/")  # clones de open-ecg-digitizer (oed/) e paper-ecg (pecg/)
imgs = {"schiller10": E+"oed/test/test_data/data/ecg_data/10_1.png",
        "schiller11": E+"oed/test/test_data/data/ecg_data/11_1.png",
        "schiller12": E+"oed/test/test_data/data/ecg_data/12_1.png",
        "photo0": E+"oed/assets/visual_abstract-img0.png",
        "photo1": E+"oed/assets/visual_abstract-img1.png",
        "ge_scan": E+"pecg/fullScan.png",
        "ge_rot": E+"pecg/rotatedFullScan.png"}
sel = sys.argv[1:] or list(imgs)
for k in sel:
    t=time.time()
    try:
        r = digitize(imgs[k], Config(), return_debug=True)
        li = r.get("layout_info", {})
        print(k, f"{time.time()-t:.0f}s", r["layout"], r["layout_rows"], "speed", r["speed_mm_per_s"], r["speed_source"],
              "dur", r["duration_s"], "gains", r["gains"], "ppmm", round(r["ppmm"],2), "conf", li.get("confidence"), flush=True)
        print("   rank:", str(li.get("ranking", [])[:4])[:300], flush=True)
        pickle.dump(r, open(f"{k}.pkl","wb"))
    except Exception as e:
        print(k, "FAIL", repr(e)); traceback.print_exc()
