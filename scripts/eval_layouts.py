import sys, glob, json, time, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from ecgdig.digitize import digitize, Config
from ecgdig.evaluate import snr_db
from ecgdig.layouts import LEADS
spd = None if '--autospeed' in sys.argv else 25.0
files=[a for a in sys.argv[1:] if not a.startswith('--')]
ok=0; tot=0; snrs=[]
for d in files:
  for f in sorted(glob.glob(d+'/*.png')):
    m=json.load(open(f.replace('.png','.json'))) if glob.glob(f.replace('.png','.json')) else None
    t=time.time()
    try:
        r=digitize(f, Config(speed=spd))
    except Exception as e:
        print(f, 'ERRO', e); continue
    li=r['layout_info'] or {}
    gt_rows=m['rows'] if m else None
    good = (r['layout_rows']==gt_rows) if m else None
    spd_ok = (r['speed_mm_per_s']==m['speed']) if m else None
    tot+=1; ok+= bool(good)
    d_=np.load(f.replace('.png','.npz')); g,sh=d_['sig'],d_['shown']
    s=np.full_like(g,np.nan); n=min(s.shape[1], r['signals'].shape[1]); s[:,:n]=r['signals'][:,:n]
    v=[snr_db(np.where(sh[i],g[i],np.nan), np.where(sh[i], s[i], np.nan)) for i in range(12) if sh[i].any()]
    snrs.append(np.nanmean(v))
    print('%s %-16s %s conf %.2f speed %s dur %.1f gains %s SNR %.1f %.0fs'%(f.split('/')[-2]+'/'+f[-7:-4], (m['layout'] if m else '?'), 'OK ' if good else 'ERR '+str(r['layout']), li.get('confidence',0), r['speed_mm_per_s'], r['duration_s'], sorted(set(r['gains'].values())), np.nanmean(v), time.time()-t))
    if not good and m: print('     pred', r['layout_rows'], '\n     gt  ', gt_rows, '\n     rank', li.get('ranking'))
print('layout correto %d/%d  SNR médio %.2f'%(ok,tot,np.nanmean(snrs)))
