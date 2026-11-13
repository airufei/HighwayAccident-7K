"""Publication figures: bilingual, vector first; original scripts are untouched."""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import font_manager
FIG_LANG = os.environ.get('FIG_LANG', 'zh')
OUT = os.environ.get('FIG_OUT', str(Path(__file__).parent / 'output' / FIG_LANG))
Path(OUT).mkdir(parents=True, exist_ok=True)
available = {f.name for f in font_manager.fontManager.ttflist}
fonts = [f for f in ['Times New Roman', 'SimSun', 'Noto Serif CJK SC', 'Microsoft YaHei', 'DejaVu Serif'] if f in available]
plt.rcParams.update({'font.family': fonts, 'mathtext.fontset': 'stix',
 'font.size':8, 'axes.titlesize':8.5, 'axes.labelsize':8,
 'xtick.labelsize':7, 'ytick.labelsize':7, 'legend.fontsize':7,
 'axes.linewidth':.65, 'axes.spines.top':False,'axes.spines.right':False,
 'lines.linewidth':1.2, 'lines.markersize':3.5, 'pdf.fonttype':42,
 'ps.fonttype':42, 'svg.fonttype':'none', 'savefig.bbox':'tight',
 'savefig.pad_inches':.08, 'axes.unicode_minus':False})
OI = dict(blue='#0072B2',orange='#E69F00',green='#009E73',red='#D55E00',purple='#CC79A7',sky='#56B4E9',yellow='#F0E442',black='#202020')
ROLE = dict(stage1=OI['blue'],stage2=OI['green'],stage3=OI['orange'],warn=OI['red'],aux=OI['purple'],neutral='#606060')
TINT = dict(stage1='#edf3f7',stage2='#eef5f2',stage3='#faf5e9',warn='#faf0ec',aux='#f5eff3',neutral='#f1f1f1')
def T(key,zh,en): return en if FIG_LANG == 'en' else zh
def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(Path(OUT)/(name+'.'+ext), dpi=600)
    plt.close(fig)
def rbox(ax,x,y,w,h,fc='white',ec='#444444',lw=.8,ls='-',r=0,z=2):
    p=Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,linewidth=lw,linestyle=ls,zorder=z)
    ax.add_patch(p)
    return p
def arrow(ax,x1,y1,x2,y2,color='#444444',lw=.85,ls='-',z=3):
    p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,color=color,lw=lw,ls=ls,zorder=z)
    ax.add_patch(p)
    return p
def fit_text(ax,fig,x,y,s,maxw,fs,minfs=7,**kw):
    return ax.text(x,y,s,fontsize=max(fs,minfs),**kw)
