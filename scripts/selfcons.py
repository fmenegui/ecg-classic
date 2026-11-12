# autoconsistência em fotos reais: cópias duplicadas (grade x ritmo) e Einthoven, sem fusão
import sys, ast, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from ecgdig.digitize import digitize, Config
kw={}
for a in sys.argv[1:]:
    k,v=a.split('='); kw[k]=ast.literal_eval(v)
tot=[]
for f in ['/tmp/claude-0/real/ecg1.jpg','/tmp/claude-0/real/ecg2.jpg','/tmp/claude-0/ext2/pecg/fullScan.png','/tmp/claude-0/ext2/pecg/rotatedFullScan.png']:
    r=digitize(f, Config(einthoven=False, **kw), return_debug=True)
    ps,pw=r['per_segment'],r['per_segment_weight']; out=[]
    for l,lst in ps.items():
        if len(lst)<2: continue
        a,b=lst[0],lst[1]; mu=~np.isnan(a)&~np.isnan(b)&(pw[l][0]>0.5)&(pw[l][1]>0.5)
        if mu.sum()<100: continue
        e=1000*np.sqrt(np.mean((a[mu]-b[mu])**2)); out.append(f'{l}:{e:.0f}'); tot.append(e)
    s=r['signals']; mm=~np.isnan(s[0])&~np.isnan(s[1])&~np.isnan(s[2])
    ein=1000*np.sqrt(np.mean((s[0]-s[1]+s[2])[mm]**2)); tot.append(ein)
    print(f.split('/')[-1], r['layout'], ' '.join(out), 'einth %.0f'%ein, flush=True)
print('SUM', kw, '%.1f'%np.mean(tot))
