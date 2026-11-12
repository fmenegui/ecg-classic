# uso: python evalcfg.py <dir> key=val ...  -> SNR médio e por imagem
import sys, glob, json, numpy as np, ast
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from ecgdig.digitize import digitize, Config
from ecgdig.evaluate import snr_db
d=sys.argv[1]; kw={}
for a in sys.argv[2:]:
    k,v=a.split('='); kw[k]=ast.literal_eval(v)
snrs=[]
for f in sorted(glob.glob(d+'/*.png')):
    z=np.load(f.replace('.png','.npz')); g,sh=z['sig'],z['shown']
    try: r=digitize(f, Config(**kw))
    except Exception as e: print(f,'ERR',e); continue
    s=np.full_like(g,np.nan); n=min(s.shape[1], r['signals'].shape[1]); s[:,:n]=r['signals'][:,:n]
    v=[snr_db(np.where(sh[i],g[i],np.nan), np.where(sh[i], s[i], np.nan)) for i in range(12) if sh[i].any()]
    snrs.append(np.nanmean(v)); print(f[-7:-4], '%.2f'%snrs[-1], flush=True)
print('MEAN', kw, '%.3f'%np.mean(snrs), 'median %.2f'%np.median(snrs))
