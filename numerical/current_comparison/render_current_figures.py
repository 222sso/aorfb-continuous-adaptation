"""Render only the existing Figures 1–5 from frozen scored-summary records. No training, generation, predictions or inferential tests."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import FancyBboxPatch
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=Path('reproduced_figures'));args=parser.parse_args()
D=Path(__file__).resolve().parent;HERE=args.output_dir.resolve()
for name in ['figures','editable_figures','qa']:(HERE/name).mkdir(parents=True,exist_ok=True)
seed=pd.read_csv(D/'seed_metrics_96.csv');tests=pd.read_csv(D/'primary_tests_3.csv').set_index('backbone')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
B=['LSTM','TCN','TCN-LSTM'];C=['hard_fixed10','continuous_targetmax10'];colors=['#0072B2','#D55E00'];labels=['Hard10','Continuous10'];spec=[];plotted=[];layout_checks=[]
def save(fig,n):
 from matplotlib.text import Text
 fig.canvas.draw();renderer=fig.canvas.get_renderer();fw,fh=fig.canvas.get_width_height()
 outside=[]
 for artist in fig.findobj(Text):
  if not artist.get_visible() or not artist.get_text().strip():continue
  bb=artist.get_window_extent(renderer)
  if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fw+0.5 or bb.y1 > fh+0.5:outside.append(artist.get_text())
 layout_checks.append({'figure':n,'outside_canvas_text':outside})
 assert not outside,(n,outside)
 for ext in ['pdf','png','svg']:
  path=(HERE/'figures' if ext=='pdf' else HERE/'editable_figures')/f'Figure_{n}.{ext}';fig.savefig(path,dpi=600,facecolor='white')
 spec.append({'figure':n,'width_mm':170,'height_mm':fig.get_figheight()*25.4,'source':'Frozen v58 independent 16-pair study; no new experiments or tests'});plt.close(fig)
def box(ax,x,y,w,h,text,color='#EAF1F5',size=8):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',linewidth=.7,edgecolor='#66717A',facecolor=color));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.4)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#52606A','lw':1})
fig,ax=plt.subplots(figsize=(170/25.4,93/25.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.02,.78,.96,.18,'ChEMBL pretraining → one fixed checkpoint per backbone\nLSTM · TCN · TCN–LSTM',size=9)
box(ax,.02,.40,.43,.27,'Hard10\nYuan Model 4 selection principle [15]\n1,529 targets; unit loss weights\nUniform subset sampling','#E5F1F9',8.5)
box(ax,.55,.40,.43,.27,'Continuous10\nAll 30,829 training molecules\nAll 1,529 targets: loss weight 10\nProperty-dependent sampling + loss','#FCEDDF',8.5)
arrow(ax,(.24,.78),(.24,.67));arrow(ax,(.76,.78),(.76,.67))
box(ax,.02,.03,.96,.26,'No ChEMBL replay · final epoch 10 · 8,000 updates\n16 new paired seeds/backbone · 9 cells × 1,000 attempts/run\n96 runs / 864,000 attempts · frozen property screen\nPrimary endpoint fixed before fitting; three tests, Bonferroni correction\nExisting Figures 6–8: separate diagnostic / DFT protocols',size=8)
arrow(ax,(.24,.40),(.24,.29));arrow(ax,(.76,.40),(.76,.29));fig.subplots_adjust(0,0,1,1);save(fig,1)
fig,ax=plt.subplots(figsize=(170/25.4,96/25.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.02,.75,.96,.21,'Same RedDB split: 30,829 training + 311 validation molecules\nShared backbone-specific pretrained checkpoint; validation monitors only',size=9)
box(ax,.02,.30,.43,.35,'Hard10\nYuan Model 4 selection principle [15]\n1,529 targets only\nAll non-padding token weights = 1\nShuffle and repeat subset','#E5F1F9',8.5)
box(ax,.55,.30,.43,.35,'Continuous10: target-max variant\n1,529 targets: loss weight = 10\n29,300 others: original property-rank weights\nFull support + sampling + weighted loss','#FCEDDF',8)
arrow(ax,(.24,.75),(.24,.65));arrow(ax,(.76,.75),(.76,.65))
box(ax,.02,.02,.96,.20,'Continuous pool rule: weight ≥ 1.2; high-pool probability = 0.90\nHigh: 2,045 (1,529 targets + 516 others); low: 28,784\n344 targets move to the high pool after the override',size=8)
arrow(ax,(.76,.30),(.76,.22));fig.subplots_adjust(0,0,1,1);save(fig,2)
fig,axs=plt.subplots(1,3,figsize=(170/25.4,108/25.4),sharey=True)
metric='yield_unique_novel_target_pct';low=np.floor((seed[metric].min()-.3)*2)/2;high=np.ceil((seed[metric].max()+.3)*2)/2
for ax,b,letter in zip(axs,B,'abc'):
 h=seed[(seed.backbone==b)&(seed.condition==C[0])].sort_values('training_seed');c=seed[(seed.backbone==b)&(seed.condition==C[1])].sort_values('training_seed')
 assert h.training_seed.tolist()==c.training_seed.tolist()==list(range(20261021,20261037))
 hs=h[metric].to_numpy();cs=c[metric].to_numpy();jitter=np.linspace(-.07,.07,16)
 for i in range(16):
  ax.plot([jitter[i],1+jitter[i]],[hs[i],cs[i]],color='#B5B5B5',lw=.65,alpha=.65,zorder=1)
  plotted.append({'backbone':b,'training_seed':int(h.iloc[i].training_seed),'Hard10':float(hs[i]),'Continuous10':float(cs[i])})
 for k,values in enumerate([hs,cs]):
  ax.scatter(k+jitter,values,s=15,color=colors[k],alpha=.85,zorder=3)
  ax.errorbar(k,values.mean(),yerr=values.std(ddof=1),fmt='D',ms=5,color='black',capsize=3,lw=1.2,zorder=4)
 r=tests.loc[b]
 s=f'Δ = {r.mean_difference_pp:+.3f} pp\n98.3333% simultaneous CI\n[{r.simultaneous_ci_lower_pp:.3f}, {r.simultaneous_ci_upper_pp:.3f}] pp\nBonferroni-adjusted p = {r.p_bonferroni3:.4f}'.replace('-0.','−0.')
 ax.text(.5,1.025,s,transform=ax.transAxes,ha='center',va='bottom',fontsize=7.5,linespacing=1.4)
 ax.set_title(f'({letter}) '+b.replace('TCN-LSTM','TCN–LSTM'),loc='left',pad=58)
 ax.set_xticks([0,1],labels,rotation=12);ax.set_xlim(-.5,1.5);ax.set_ylim(low,high);ax.yaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
axs[0].set_ylabel('Novel unique predicted target yield (%)')
fig.text(.5,.060,'Δ = Continuous10 − Hard10 · pp = percentage points',ha='center',fontsize=7)
fig.text(.5,.025,'Points: 16 seeds/arm · lines: paired seeds · diamonds and bars: mean ± sample SD',ha='center',fontsize=7)
fig.subplots_adjust(left=.10,right=.98,top=.68,bottom=.22,wspace=.25);save(fig,3)

fig,axs=plt.subplots(1,3,figsize=(170/25.4,80/25.4))
for ax,m,title,letter in zip(axs,['uniqueness_pct','novelty_pct','target_pass_pct'],['Output uniqueness','Molecular novelty','Target-pass rate'],'abc'):
 for j,c in enumerate(C):
  means=[seed[(seed.backbone==b)&(seed.condition==c)][m].mean() for b in B];sds=[seed[(seed.backbone==b)&(seed.condition==c)][m].std(ddof=1) for b in B];x=np.arange(3)+(j-.5)*.32
  ax.bar(x,means,width=.29,color=colors[j],alpha=.8,yerr=sds,capsize=2,label=labels[j])
  for k,b in enumerate(B):
   vals=seed[(seed.backbone==b)&(seed.condition==c)].sort_values('training_seed')[m].to_numpy();ax.scatter(x[k]+np.linspace(-.08,.08,len(vals)),vals,s=6,color='black',alpha=.65,zorder=3,linewidths=0)
 ax.set_xticks(range(3),['LSTM','TCN','TCN–\nLSTM']);ax.set_title(f'({letter}) {title}',loc='left',pad=7);ax.set_ylabel('Percent (%)');ax.set_ylim(bottom=0);ax.yaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.01));fig.subplots_adjust(left=.08,right=.99,top=.86,bottom=.28,wspace=.52);save(fig,4)

pool=pd.read_csv(D/'pool_values_96.csv')
fig,ax=plt.subplots(figsize=(170/25.4,80/25.4));saved=[]
for j,c in enumerate(C):
 for k,b in enumerate(B):
  q=pool[(pool.backbone==b)&(pool.condition==c)].sort_values('training_seed');vals=q.unique_predicted_targets_9000.to_numpy();assert len(vals)==16;x=k+(j-.5)*.22
  ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1),fmt='o',color=colors[j],capsize=4,ms=6,label=labels[j] if k==0 else None)
  ax.scatter(x+np.linspace(-.04,.04,16),vals,s=12,color=colors[j],alpha=.55)
  ax.text(x+(-.05 if j==0 else .05),vals.mean()+35,f'{vals.mean():.1f}',ha='right' if j==0 else 'left',fontsize=8,color=colors[j])
  saved.append({'backbone':b,'condition':c,'n':16,'mean':float(vals.mean()),'sample_sd':float(vals.std(ddof=1)),'seed_counts':vals.tolist()})
ax.set_xticks(range(3),['LSTM','TCN','TCN–LSTM']);ax.set_ylabel('Distinct predicted targets / 9,000 attempts');ax.set_ylim(np.floor((pool.unique_predicted_targets_9000.min()-10)/10)*10,np.ceil((pool.unique_predicted_targets_9000.max()+35)/10)*10);ax.set_xlim(-.5,2.5);ax.yaxis.grid(True,alpha=.2)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.55,.015));fig.subplots_adjust(left=.12,right=.98,bottom=.26,top=.97)
fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height();outside=[]
for a in fig.findobj(Text):
 if a.get_visible() and a.get_text().strip():
  bb=a.get_window_extent(renderer)
  if bb.x0<-.5 or bb.y0<-.5 or bb.x1>w+.5 or bb.y1>h+.5:outside.append(a.get_text())
assert not outside,outside
save(fig,5)

(HERE/'figure_specs.json').write_text(json.dumps(spec,indent=2))
(HERE/'qa/figure_artist_layout.json').write_text(json.dumps({'status':'PASS','figures':layout_checks},indent=2))
print('Rendered existing Figures1–5; Figures6–8 retain their separate archived sources.')
