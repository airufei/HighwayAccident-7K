from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT.parent/'04_results/recomputed/results.json').read_text(encoding='utf8'))['controls']
for lang in ['en','zh']:
    plt.rcParams.update({'font.family':'Times New Roman' if lang=='en' else 'SimSun','font.size':9,'axes.linewidth':0.7,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    fig.subplots_adjust(left=.065,right=.995,bottom=.19,top=.76,wspace=.26)
    arms=['last','mean','single_mid','gru'];labels=['Last loss','Mean loss','Middle loss','GRU'] if lang=='en' else ['末帧损失','均值损失','中帧损失','GRU']
    x=np.arange(4);width=.34
    for ax,key,title in zip(axes,['balanced_accuracy','far'],['(a) Balanced accuracy','(b) Normal-clip false-alarm rate'] if lang=='en' else ['(a) 平衡准确率','(b) 正常片段误报率']):
        for shift,field,color,label in [(-.5,'reported_frame_head_endpoint','#8c979f','Frame readout' if lang=='en' else '帧输出'),(.5,'actual_clip_head_endpoint','#007f78','Clip readout' if lang=='en' else '片段聚合输出')]:
            vals=[d[a][field][key] for a in arms]
            bars=ax.bar(x+shift*width,vals,width,color=color,edgecolor='black',linewidth=.35,label=label,zorder=3)
            for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+.25,f'{val:.1f}',ha='center',va='bottom',fontsize=7)
        ax.set_xticks(x,labels);ax.set_ylabel('%');ax.set_title(title,loc='left',fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#dddddd',linewidth=.5,zorder=0);ax.tick_params(direction='out',length=3)
        ax.set_ylim((0,89) if key=='balanced_accuracy' else (0,27))
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2)
    out=ROOT/'figures'/lang;out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'frozen_head_readouts.{ext}',dpi=500,bbox_inches='tight',facecolor='white')
    plt.close(fig)
print('Wrote bilingual readout figures.')
