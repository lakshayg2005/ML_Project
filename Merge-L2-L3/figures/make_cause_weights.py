"""Draws the mean CACG cause weights by context length. Needs the runs' *_test_outputs.npz files, which git does not track."""
import os

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Arial','font.size':11})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'runs', 'MELD')
M=[np.load(f'{R}/cacg_none_seed{s}_test_outputs.npz') for s in (0,1,2)]
G=np.stack([m['g'] for m in M]).mean(0); ctx=M[0]['ctx']
B=[('16+',16,999),('8–15',8,15),('4–7',4,7),('1–3',1,3),('0',0,0)]
names=['Self-contagion','Cross-emotion','Self-event','Cross-event']; col=['#1F6FB5','#7FB2E0','#8A93A0','#C9CED6']; txt=['white','#1E2530','white','#1E2530']
fig,ax=plt.subplots(figsize=(8.6,3.5)); left=np.zeros(len(B))
V=np.array([G[(ctx>=a)&(ctx<=b)].mean(0) for _,a,b in B]); N=[int(((ctx>=a)&(ctx<=b)).sum()) for _,a,b in B]
for k in range(4):
    ax.barh(range(len(B)),V[:,k],left=left,color=col[k],edgecolor='white',height=.68,label=names[k])
    for i,v in enumerate(V[:,k]):
        if v>=.08: ax.text(left[i]+v/2,i,f'{v:.2f}',ha='center',va='center',color=txt[k],fontsize=10.5)
    left+=V[:,k]
ax.set_yticks(range(len(B))); ax.set_yticklabels([f'{b[0]}  (n = {n})' for b,n in zip(B,N)])
ax.set_ylabel('Prior utterances in the dialogue'); ax.set_xlabel('Mean gate weight'); ax.set_xlim(0,1)
for s in('top','right','left'): ax.spines[s].set_visible(False)
ax.tick_params(axis='y',length=0)
ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,1.0),frameon=False,fontsize=10.5,handlelength=1.2,columnspacing=1.4)
fig.savefig(os.path.join(HERE, 'meld_cause_weights.png'),dpi=300,bbox_inches='tight',facecolor='white')
print(np.round(V,3),N)
