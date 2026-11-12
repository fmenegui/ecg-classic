import sys, glob, os, time, pickle, traceback, json, numpy as np, cv2
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from ecgdig.digitize import digitize, Config
files=sorted(glob.glob('img/*'))
k,n=int(sys.argv[1]),int(sys.argv[2])
os.makedirs('out',exist_ok=True)
for f in files[k::n]:
    name=os.path.splitext(os.path.basename(f))[0]
    t=time.time()
    try:
        r=digitize(f, Config(), return_debug=True)
        li=r.get('layout_info') or {}
        d=r['debug_image']; s=1600/max(d.shape[:2]); dbg=cv2.resize(d,None,fx=min(1,s),fy=min(1,s),interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'out/{name}_dbg.jpg', dbg, [cv2.IMWRITE_JPEG_QUALITY, 85])
        res=dict(name=name, ok=True, layout=r['layout'], rows=r['layout_rows'], conf=float(li.get('confidence',0) or 0), warning=li.get('warning'),
                 speed=r['speed_mm_per_s'], speed_src=r['speed_source'], dur=r['duration_s'], gains=sorted(set(r['gains'].values())),
                 ppmm=float(r['ppmm']), pulses=r.get('pulses_found'), resid=r.get('grid_residual_mm'), t=time.time()-t,
                 finite=float(np.isfinite(r['signals']).mean()))
        np.save(f'out/{name}_sig.npy', r['signals'].astype(np.float32))
    except Exception as e:
        res=dict(name=name, ok=False, err=repr(e)[:300], tb=traceback.format_exc(limit=3)[-600:], t=time.time()-t)
    json.dump(res, open(f'out/{name}.json','w'), default=str)
    print(name, res.get('layout', res.get('err')), flush=True)
