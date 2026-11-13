"""Manuscript diagrams, aligned to the recorded inference path."""
from style import *

def tr(zh,en): return T('',zh,en)
def canvas(height=4):
    fig,ax=plt.subplots(figsize=(7.16,height))
    ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    return fig,ax
def box(ax,x,y,w,h,title,body='',color='#444444',dashed=False):
    rbox(ax,x,y,w,h,'white',color,ls='--' if dashed else '-')
    if body:
        ax.text(x+w/2,y+h-3,title,ha='center',va='top',fontsize=8.5,fontweight='bold')
        ax.text(x+w/2,y+h/2-3,body,ha='center',va='center',fontsize=8,linespacing=1.5)
    else:ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=8.5)

def framework():
    fig,ax=canvas(3.8)
    box(ax,2,80,96,17,tr('视频输入','Video input'),tr('车辆运动与事件上下文','Vehicle motion and incident context'))
    stages=[(2,ROLE['stage1'],tr('运动筛选','Motion screening'),tr('YOLO26-m + BoT-SORT\n轨迹 → 候选窗口','YOLO26-m + BoT-SORT\nTracks → candidate windows')),
            (36,ROLE['stage2'],tr('时序分类','Temporal classification'),tr('16 帧，步长 8\n场景 / 类型 / 因素标签','16 frames, stride 8\nScene / type / factor labels')),
            (70,ROLE['stage3'],tr('辅助描述','Auxiliary summary'),tr('关键帧 + 轨迹文本\n结构化证据描述','Keyframes + trajectory text\nStructured evidence summary'))]
    for x,c,t,b in stages:box(ax,x,38,28,30,t,b,c,dashed=(x==70))
    arrow(ax,16,80,16,68);arrow(ax,30,53,36,53);arrow(ax,64,53,70,53)
    for x,t,b in [(2,tr('候选输出','Candidates'),tr('窗口与轨迹','Windows and tracks')),
                  (36,tr('主要预测输出','Primary predictions'),tr('场景与碰撞类型','Scene and collision type')),
                  (70,tr('事后检查','Retrospective inspection'),tr('观测与预测关联','Observations linked to predictions'))]:
        box(ax,x,5,28,19,t,b);arrow(ax,x+14,38,x+14,24)
    save(fig,'fig1_framework')

def encoder():
    fig,ax=canvas(4.0)
    box(ax,1,73,23,23,tr('输入序列','Input sequence'),tr('16 × 256 特征\n局部卷积 + 位置 / CLS','16 × 256 features\nLocal CNN + position / CLS'))
    box(ax,32,73,28,23,tr('标准编码层 ×2','Standard layers ×2'),tr('8 头；前馈维度 1024','8 heads; FF dimension 1024'))
    box(ax,70,73,29,23,tr('偏置编码层 ×2','Biased layers ×2'),tr('注意力分数 + B','Attention scores + B'))
    arrow(ax,24,84,32,84);arrow(ax,60,84,70,84)
    box(ax,30,40,30,20,tr('类型输入','Type input'),tr('训练：类型 / padding\n评估：统一 padding','Training: type / padding\nEvaluation: padding only'))
    box(ax,70,40,29,20,tr('偏置生成器','Bias generator'),tr('类型嵌入 → MLP → B','Type embedding → MLP → B'))
    arrow(ax,60,50,70,50);arrow(ax,84.5,60,84.5,73)
    ax.plot([99,100,100,11,89],[83,83,28,28,28],color='#444444',lw=.85)
    for x,t,b in [(1,tr('场景','Scene'),tr('三态','Three states')),(27,tr('碰撞类型','Collision type'),tr('七类输出','Seven outputs')),(53,tr('因素标签','Factor labels'),tr('七监督位','Seven supervised bits')),(79,tr('辅助回归','Auxiliary regression'),'')]:
        box(ax,x,3,20,18,t,b);arrow(ax,x+10,28,x+10,21)
    save(fig,'fig2_temporal_encoder')

def summary():
    fig,ax=canvas(3.8)
    box(ax,1,82,98,16,tr('输入证据','Evidence input'),tr('关键帧 + 轨迹测量 + 分类先验','Keyframes + trajectory measurements + classifier priors'))
    bodies=[tr('可见事件与车辆运动','Visible events and vehicle motion'),tr('TTC、间距及运动测量','TTC, spacing, and motion measurements'),tr('潜在因素及支持观测','Possible factors with supporting observations'),tr('供事后复核的简短描述','Concise description for retrospective review')]
    arrow(ax,50,82,50,78)
    for i,(name,body) in enumerate(zip(['OBSERVE','ANALYZE','INFER','EXPLAIN'],bodies)):
        y=60-i*18
        box(ax,1,y,20,17,name,color=ROLE['stage1']);box(ax,21,y,78,17,body)
        if i<3:arrow(ax,50,y,50,y-1)
    save(fig,'fig3_cot_chain')

if __name__=='__main__':framework();encoder();summary()
