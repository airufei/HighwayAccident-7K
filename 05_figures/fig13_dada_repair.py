# -*- coding: utf-8 -*-
"""图 13（§8.4・G-358⑥新增）域外同留出集评估：DADA 混合训练前后的同留出对照。

一图一事：(a) 同一 2,769 clip 留出上，留出 FAR 91.6→48.2、正常 clip 上平均 p_crash
   打分 83.4→36.6——域外丢掉的是排序信号本身，混合训练把它找回了大半；
(b) 归因分解：人口效应 0.4pp vs 模型效应 43.4pp——腰斩几乎全是真修复，不是分母伪影；
(c) 域内性能变化：交付件 bal 77.01→72.60、FAR 11.32→14.57，找回跨域能力的价格由主表承担，
   平衡点（8.5% 配比）已收官入图（第三臂 dada10）。
数字全部走出数面 dada_repair 块（两面对撞闸 12 已核），脚本内零字面读数。
注意：本训练服务器无 CJK 字体，只产英文版（FIG_LANG=en）；中文版在论文侧主机渲染。
"""
from figdata import *  # noqa
import matplotlib.patches as mpatches  # 图例代理

_F = load()
DR = _F["dada_repair"]

BEF = DR["holdout_p3c"]
AFT = DR["holdout_dada354"]
DEC = DR["decomposition"]
MAIN = DR["main_p3c"]
MAIN2 = DR["main_dada354"]
BAL = DR.get("balance_dada10")          # 周一 dada10 收官后自动出现；None＝维持双臂图
SLOT = 3 if BAL else 2                  # 每组柱数

C_BEF, C_AFT = MUTED, ROLE["stage2"]
LBL_BEFORE = T("lbl.before", "基准模型", "Baseline")
LBL_AFTER = T("lbl.after", "22% 混合", "22% mix")


def main():
    fig, axes = figure_pair(3, W_FULL, 2.8)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.845, bottom=0.25, wspace=0.32)

    # ---- (a) 同留出前后对照 --------------------------------------------------
    ax = panel(axes[0], "(a)", T("a.title", "同留出集评估", "Same-holdout evaluation"),
               T("a.msg", "同留出 2,769 clips", "One holdout, 2,769 clips"),
               ROLE["stage2"])
    g1 = T("a.g1", "留出 FAR\n（%）", "Holdout\nFAR (%)")
    g2 = T("a.g2", "正常 clip\np_crash（%）", "p_crash on\nnormal (%)")
    vals_b = [BEF["FAR"], BEF["p_crash_on_normal"]]
    vals_a = [AFT["FAR"], AFT["p_crash_on_normal"]]
    ncol_g = [SLOT, 2]                    # FAR 组三臂（dada10 有 FAR）；p_crash 组两臂（无该读数）
    x0 = 0
    for g in range(2):
        ax.bar(x0, vals_b[g], width=0.66, color=C_BEF, zorder=3)
        ax.bar(x0 + 1, vals_a[g], width=0.66, color=C_AFT, zorder=3)
        datalabels(ax, [x0], [vals_b[g]], ["%s" % vals_b[g]], color=INK)
        datalabels(ax, [x0 + 1], [vals_a[g]], ["%s" % vals_a[g]], color=INK)
        if BAL and g == 0:
            v = BAL["dada"]["FAR"]
            ax.bar(x0 + 2, v, width=0.66, color=ROLE["stage3"], zorder=3)
            datalabels(ax, [x0 + 2], [v], ["%s" % v], color=INK)
        x0 += ncol_g[g]
    ax.set_xticks([0 + (ncol_g[0] - 1) / 2.0, ncol_g[0] + 0.5])
    ax.set_xticklabels([g1, g2])
    ax.set_xlim(-0.6, x0 - 0.4)
    ax.set_ylim(0, 118)
    axis_labels(ax, None, T("a.ylab", "读数（%）", "value (%)"))
    _h = [mpatches.Patch(color=C_BEF, label=LBL_BEFORE),
          mpatches.Patch(color=C_AFT, label=LBL_AFTER)]
    if BAL:
        _h.append(mpatches.Patch(color=ROLE["stage3"],
                                 label=T("lbl.bal", "8.5% 混合", "8.5% mix")))
    ax.legend(handles=_h, fontsize=FS["legend"], loc="upper right", frameon=False, handlelength=1.4)

    # ---- (b) 归因分解：人口 vs 模型 -----------------------------------------
    ax = panel(axes[1], "(b)", T("b.title", "FAR 差异分解", "FAR difference decomposition"),
               T("b.msg", "模型效应 43.4pp", "43.4pp model effect"),
               ROLE["aux"])
    st1 = T("b.st1", "基准全量\n(29,164)", "Baseline full\n(29,164)")
    st2 = T("b.st2", "基准留出\n(2,769)", "Baseline hold\n(2,769)")
    st3 = T("b.st3", "混合后留出\n(2,769)", "22% mix\nhold (2,769)")
    ys = [DR["full_p3c"]["FAR"], BEF["FAR"], AFT["FAR"]]
    ax.bar(range(3), ys, width=0.66, color=[C_BEF, C_BEF, C_AFT], zorder=3)
    datalabels(ax, list(range(3)), ys, ["%s" % y for y in ys], color=INK)
    ax.annotate(T("b.dpop", "人口 −%spp" % DEC["population_pp"], "pop −%spp" % DEC["population_pp"]),
                (0.5, ys[1] + 6), (0.5, 101), fontsize=FS["callout"], color=MUTED, ha="center")
    ax.annotate(T("b.dmod", "模型 −%spp" % DEC["model_pp"], "model −%spp" % DEC["model_pp"]),
                (1.9, ys[2] + 18), (1.9, 70), fontsize=FS["callout"], color=ROLE["stage2"], ha="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels([st1, st2, st3])
    ax.set_ylim(0, 118)
    axis_labels(ax, None, T("b.ylab", "FAR（%）", "FAR (%)"))
    callout(ax, (2, ys[2] + 2),
            T("b.call", "42.4 已越随机线 33.3\n残余域差仍在（caveat）", "42.4 clears chance 33.3;\nresidual gap remains"),
            (2.55, 108), color=ROLE["warn"], ha="right")

    # ---- (c) 域内性能变化 ---------------------------------------------------------
    ax = panel(axes[2], "(c)", T("c.title", "域内性能变化", "In-domain performance"),
               T("c.msg", "找回由主表付账", "Recovery is paid here"),
               ROLE["stage3"])
    c1 = T("c.g1", "bal_acc\n（%）", "bal_acc\n(%)")
    c2 = T("c.g2", "FAR\n（%）", "FAR\n(%)")
    final = _F["main_dashboard"]["headline"]
    groups=[
        (T('', '第 8 轮基准', 'Baseline, epoch 8'), [MAIN['bal_acc'],MAIN['FAR']],MUTED),
        (T('', '末轮基准', 'Baseline, final'), [final['scene_bal_acc'],final['FAR']],OI['sky']),
        (T('', '22% 混合', '22% mix'),[MAIN2['bal_acc'],MAIN2['FAR']],C_AFT),
    ]
    if BAL:groups.append((T('', '8.5% 混合', '8.5% mix'),[BAL['main']['scene_bal_acc'],BAL['main']['FAR']],ROLE['stage3']))
    count=len(groups)
    for g in range(2):
        for j,(label,values,col) in enumerate(groups):
            x=g*(count+1)+j
            ax.bar(x,values[g],width=.7,color=col,zorder=3)
            datalabels(ax,[x],[values[g]],[str(values[g])],color=INK)
    ax.set_xticks([(count-1)/2,count+1+(count-1)/2])
    ax.set_xticklabels([T('', '平衡准确率', 'Balanced accuracy'), 'FAR'])
    ax.set_xlim(-.7,2*(count+1)-1.3);ax.set_ylim(0,118)
    axis_labels(ax,None,T('', '数值（%）', 'Value (%)'))
    handles=[mpatches.Patch(color=col,label=label) for label,_,col in groups]
    ax.legend(handles=handles,fontsize=FS['legend'],loc='upper right',ncol=2,frameon=False,handlelength=1.2)

    zh_note = ("口径：%s；三面 sha：comparison %s / far_decomp %s / p3c-main %s；"
               "caveat：%s（8.5%% 平衡点已入图作第三臂）；出数脚本 sha %s"
               % (DR["caliber"][:64], DR["faces"]["comparison"][:8],
                  DR["faces"]["far_decomp"][:8], DR["faces"]["p3c_main"][:8],
                  DR["caveat"][:72], FIG["instrument_sha16"]))
    en_note = ("Caliber: clip-head (last valid-frame GT vs frame-head argmax), same 2,769-clip "
               "holdout both arms; faces: comparison %s / far_decomp %s / p3c-main %s; "
               "residual FAR 48.2 vs in-domain 11-15; the 8.5%% balance point is shown as the third arm; "
               "data script sha %s"
               % (DR["faces"]["comparison"][:8], DR["faces"]["far_decomp"][:8],
                  DR["faces"]["p3c_main"][:8], FIG["instrument_sha16"]))
    save_fig(fig, "fig13_dada_repair", source_note=T("note", zh_note, en_note))


if __name__ == "__main__":
    main()
