# -*- coding: utf-8 -*-
"""图 7（§8.3）确认提前量：均值为负＋提前率与误报同源。

一图一事：(a) 两个操作点的平均提前量都在 0 左侧（bootstrap CI 不跨 0）；
(b) 提前率与 FAR 同向随 θ 退化＝同一个分数、同一次权衡，转正不靠调 θ。
出数面只给汇总统计（均值/中位/p10/p90/CI），没有逐 clip 提前量序列 ⇒
面板 (a) 画 CI-与-汇总横条图，不凭空造直方图。
"""
from figdata import *  # noqa
import matplotlib.lines as mlines  # 图例代理（面板 a 的行式图无逐条 label）

_F = load()
LT = _F["lead_time"]
SW = LT["sweep"]
ARG, REC = LT["argmax"], LT["recommended"]
CIA, CIR = LT["ci_argmax"], LT["ci_recommended"]
TH_REC = next(r for r in SW if r["theta"] == REC["theta"])
PR = REC["p10_lead_frames"]
BS = LT["bootstrap_design"]
CONF = round((1 - BS["alpha"]) * 100)


def main():
    fig, axes = figure_pair(2, W_FULL, 2.7)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.835, bottom=0.275, wspace=0.30)

    # ---- (a) 提前量分布（面内仅汇总统计） -------------------------------
    ax = panel(axes[0], "(a)", T("a.t","提前量分布（仅有汇总统计）","Lead-time summary only"),
               T("a.m","平均提前 %+.2f 帧：均值显著为负" % ARG["mean_lead_frames"],"Mean %+.2f frames: negative" % ARG["mean_lead_frames"]), ROLE["warn"])
    rows = [(ARG["mean_lead_frames"], CIA["ci_lo"], CIA["ci_hi"], ARG["median_lead_frames"],
             None, T("a.r1","主口径 argmax","argmax (main)")),
            (REC["mean_lead_frames"], CIR["ci_lo"], CIR["ci_hi"], REC["median_lead_frames"],
             (REC["p10_lead_frames"], REC["p90_lead_frames"]),
             T("a.r2","推荐 θ=%s" % REC["theta"],"rec. theta=%s" % REC["theta"]))]
    for i, (mean, lo, hi, med, pct, tag) in enumerate(rows):
        if pct:
            ax.barh(i, pct[1] - pct[0], left=pct[0], height=0.12,
                    color=TINT["neutral"], ec=GRID, lw=0.6, zorder=2)
        ax.plot([lo, hi], [i, i], "-", color=ROLE["stage1"], lw=2.2, zorder=3,
                solid_capstyle="butt")
        ax.plot([lo, lo], [i - .07, i + .07], color=ROLE["stage1"], lw=1.0, zorder=3)
        ax.plot([hi, hi], [i - .07, i + .07], color=ROLE["stage1"], lw=1.0, zorder=3)
        ax.plot([mean], [i], "o", ms=4.6, color=ROLE["stage1"], zorder=5)
        ax.plot([med], [i], "s", ms=4.4, mfc="white", mec=INK, mew=1.0, zorder=4)
        ax.text(lo - 0.7, i, "%+.2f" % mean, fontsize=FS["value"], color=INK,
                ha="right", va="center")
    ax.axvline(0, color=MUTED, ls="--", lw=0.9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[5] for r in rows])
    ax.set_ylim(-0.55, 1.85)
    ax.set_xlim(min(PR, CIA["ci_lo"]) - 4, max(REC["p90_lead_frames"],
                                                CIA["ci_hi"]) + 3)
    ax.grid(False)
    ax.set_axisbelow(True)
    axis_labels(ax, T("确认提前量（帧，","确认提前量（帧，负＝滞后于标注起点）","Lead time (frames; neg = after annotated onset)"), None)
    handles = [mlines.Line2D([], [], color=ROLE["stage1"], lw=2.2, label=T("a.g1","均值，%d%% CI" % CONF,"mean, %d%% CI" % CONF)),
               mlines.Line2D([], [], color=INK, ls="", marker="s", ms=4.4, mfc="white",
                             label=T("中位","中位","Median")),
               mlines.Line2D([], [], color=GRID, lw=6, label=T("a.g2","p10–p90","p10-p90")),
               mlines.Line2D([], [], color=MUTED, ls="--", lw=0.9,
                             label=T("a.g3","0＝标注起点才确认","0 = annotated onset"))]
    ax.legend(handles=handles, fontsize=FS["legend"], loc="upper left", ncol=2,
              frameon=False, handlelength=1.4, columnspacing=1.0, borderpad=0)
    callout(ax, (CIA["ci_hi"], 0), T("a.call","CI 上界 %+.2f 帧，整条区间在 0 左侧" % CIA["ci_hi"],"CI hi %+.2f: all left of 0" % CIA["ci_hi"]),
            (min(PR, CIA["ci_lo"]) + 0.6, 0.42), color=ROLE["warn"], ha="left")

    # ---- (b) 提前率–误报的权衡 ------------------------------------------
    ax = panel(axes[1], "(b)", T("b.t","检出比例与 FAR 随阈值变化","Detection rate and FAR vs threshold"),
               T("b.m","提前率与误报同源，转正需要更长历史窗","Same score, same trade-off"), ROLE["warn"])
    th = [r["theta"] for r in SW]
    ax.plot(th, [r["early_rate_pct"] for r in SW], "-o", color=ROLE["stage1"],
            ms=3.2, lw=1.3, label=T("b.l1","不晚于标注起点检出","By annotated onset"))
    ax.plot(th, [r["far_pct_on_normal_clip"] for r in SW], "-s", color=ROLE["warn"],
            ms=3.0, lw=1.3, label=T("b.l2","FAR（正常 clip 上）","FAR (normal)"))
    ax.axhline(ARG["early_rate_pct"], color=MUTED, ls=":", lw=1.0,
               label=T("b.l3","argmax 检出比例","argmax by onset"))
    ax.plot([TH_REC["theta"]], [TH_REC["early_rate_pct"]], "o", ms=8.5, mfc="none",
            mec=ROLE["stage1"], mew=1.6)
    ax.plot([TH_REC["theta"]], [TH_REC["far_pct_on_normal_clip"]], "s", ms=8.0,
            mfc="none", mec=ROLE["warn"], mew=1.6)
    callout(ax, (TH_REC["theta"], TH_REC["far_pct_on_normal_clip"]),
            T("b.call","推荐 θ=%s\n提前率 %s%%\nFAR %s%%"
            % (TH_REC["theta"], TH_REC["early_rate_pct"],
               TH_REC["far_pct_on_normal_clip"]),
              "theta=%s\nearly %s%%\nFAR %s%%"
            % (TH_REC["theta"], TH_REC["early_rate_pct"],
               TH_REC["far_pct_on_normal_clip"])),
            (0.55, 62), color=MUTED, ha="left")
    ax.set_xlim(min(th) - 0.02, max(th) + 0.03)
    ax.set_ylim(0, 100)
    axis_labels(ax, T("b.x","阈值 θ","Threshold"), T("b.y","比率（%）","rate (%)"))
    legend(ax, loc="center left")

    save_fig(fig, "fig07_lead_time",
             source_note=T("note","口径：%s，帧率 %s fps，正例 %d 条，只画汇总统计（结果文件里没有逐 clip 序列⇒不画直方图），"
                         "bootstrap＝%d 次重采样（seed %s，alpha %s），出数脚本 sha %s"
                     % (LT["caliber"], LT["frame_rate_fps"], ARG["n_pos"],
                        BS["n_boot"], BS["seed"], BS["alpha"], FIG["instrument_sha16"]),
                 "Frame rate %s fps, %d positives; summary stats only (no per-clip series in the result file, hence no histogram); "
                 "bootstrap %d resamples (seed %s, alpha %s); data script sha %s"
                     % (LT["frame_rate_fps"], ARG["n_pos"],
                        BS["n_boot"], BS["seed"], BS["alpha"], FIG["instrument_sha16"])))


if __name__ == "__main__":
    main()
