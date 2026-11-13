# -*- coding: utf-8 -*-
"""图 10（§8.4）域外对照：三臂平衡准确率，操作点扫描，固定机位监控桶。

一图一事：(a) 域外三臂的平衡准确率贴着随机线，域内买到的东西没兑现；
(b) 阈值只能拿召回换 FAR，换不回域内那个操作点；(c) 固定机位监控桶（real_ 前缀）
的 FAR 根本没有校准余地（正常样本太少，只能落在少数离散档上）。
命名按 §8.5-D：写「固定机位监控（real_ 前缀）」与「其余源混合」，不写"真实域 vs 合成域"。
"""
from figdata import *  # noqa
import matplotlib.patches as mpatches  # 图例代理（面板 c 的柱子本身无 label）

_F = load()
EXT = _F["external"]
BKT = _F["bucket"]

ARMS = EXT["arms"]
REF_ARM = EXT["caliber"].split("参照臂 ")[-1].strip()          # 参照臂＝交付件
ARM_COLOR = {ARMS[0]: MUTED, ARMS[1]: OI["sky"], ARMS[2]: ROLE["stage2"]}
ARM_TAG = dict(zip(ARMS, [T("ref","前代","Prior"), T("mid","中间模型","Intermediate"), T("current","时序模型","Temporal")]))
DSET = {"DADA2000": "DADA-2000", "CCD": "CCD"}
IND = EXT["in_domain_reference"]["at_far_11_9"]

# 随机线：类别数取自在册 gt_n(normal/pre/crash)，不是手抄的 33.3
N_CLASS = len(EXT["per_dataset"]["DADA2000"]["per_arm"][REF_ARM]["gt_n(normal/pre/crash)"])
CHANCE = 100.0 / N_CLASS
TOP = 105  # percentage-axis headroom; independent of any comparison model


def main():
    fig, axes = figure_pair(3, W_FULL, 2.8)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.845, bottom=0.25, wspace=0.32)

    # ---- (a) 域外三臂平衡准确率 vs 随机线 -------------------------------
    ax = panel(axes[0], "(a)", T("a.t","DADA-2000 三分类平衡准确率","DADA-2000 three-class balanced accuracy"),
               T("a.m","域内买到的东西，域外没有兑现","In-domain gains do not transfer"), ROLE["warn"])
    # CCD has no crash-class ground truth. Its zero-filled three-class
    # average is not a complete balanced-accuracy estimate; retain CCD FAR
    # in the manuscript table but omit the invalid average here.
    dsets = [ds for ds in EXT["per_dataset"] if all(
        all(n > 0 for n in EXT["per_dataset"][ds]["per_arm"][arm]["gt_n(normal/pre/crash)"])
        for arm in ARMS)]
    xs, vals = [], []
    for i, ds in enumerate(dsets):
        ax.axvspan(i * len(ARMS) - 0.5, i * len(ARMS) + len(ARMS) - 0.5,
                   color=TINT["neutral"], zorder=0)
        for j, arm in enumerate(ARMS):
            v = EXT["per_dataset"][ds]["per_arm"][arm]["balAcc"]
            ax.bar(i * len(ARMS) + j, v, width=0.78, color=ARM_COLOR[arm], zorder=3)
            xs.append(i * len(ARMS) + j)
            vals.append(v)
        ax.text(i * len(ARMS) + (len(ARMS) - 1) / 2.0, 0.62, DSET[ds],
                transform=ax.get_xaxis_transform(), fontsize=FS["label"],
                color=MUTED, ha="center", va="bottom")
        # 随机线只在这套真值三类齐全的数据集上画（面内不给 chance 读数）
        gtn = [EXT["per_dataset"][ds]["per_arm"][a]["gt_n(normal/pre/crash)"]
               for a in ARMS]
        if all(all(n > 0 for n in g) for g in gtn):
            ax.hlines(CHANCE, i * len(ARMS) - 0.45, i * len(ARMS) + len(ARMS) - 0.55,
                      color=INK, ls="--", lw=0.9, label=T("a.ch","%s，%d 类随机" % (DSET[ds], N_CLASS),"%s, chance %d" % (DSET[ds], N_CLASS)))
    datalabels(ax, xs, vals, ["%.1f" % v for v in vals], color=INK)
    # The raw in-domain reference was obtained by rethresholding a separate
    # scoring artifact, not by the main-table argmax evaluation. Do not mix
    # that 92.02% reference into the three-state external comparison.
    ax.set_xticks(list(range(len(dsets) * len(ARMS))))
    ax.set_xticklabels([ARM_TAG[a].replace("-", "\n") for a in ARMS] * len(dsets))
    ax.set_xlim(-0.5, len(dsets) * len(ARMS) - 0.5)
    ax.set_ylim(0, TOP)
    axis_labels(ax, None, T("平衡准确率（%）","平衡准确率（%）","Balanced accuracy (%)"))
    legend(ax, loc="upper center", ncol=2)

    # ---- (b) 操作点扫描救不回来 -----------------------------------------
    ax = panel(axes[1], "(b)", T("b.t","操作点扫描（%s 三臂）" % DSET["DADA2000"],"Theta sweep, 3 arms"),
               T("b.m","没有任何阈值能把域外 FAR 拉回域内水平","No theta restores OOD FAR"), ROLE["warn"])
    fr = EXT["frontier_matched_far"]["DADA2000"]
    for arm in ARMS:
        ax.plot([r["arms"][arm]["FAR"] for r in fr],
                [r["arms"][arm]["accident_R"] for r in fr],
                "-o", color=ARM_COLOR[arm], ms=3.2, lw=1.2, label=ARM_TAG[arm])
    # Show the external score sweeps only; retain the separate in-domain
    # artifact in figure_data.json without presenting it as the main model.
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    axis_labels(ax, T("FAR（%）","FAR（%）","FAR (%)"), T("b.y","事故召回率（%）","accident recall (%)"))
    legend(ax, loc="lower right")

    # ---- (c) 固定机位监控桶 vs 其余源混合桶 ------------------------------
    ax = panel(axes[2], "(c)", T("c.t","按来源分组的性能","Performance by source group"),
               T("c.m","该桶 FAR 根本无法校准（正常样本只有 %d 条）"
               % BKT["buckets"]["cctv_real"]["universe"]["n_normal"],
                 "CCTV FAR uncalibrable (only %d normals)"
               % BKT["buckets"]["cctv_real"]["universe"]["n_normal"]), ROLE["warn"])
    bks = list(BKT["buckets"].keys())
    metrics = [("FAR", "_FAR", ROLE["warn"]), (T("c.m2","平衡准确率","Bal. acc."), "_bal_acc", MUTED)]
    handles = [mpatches.Patch(color=col, label=nm) for nm, _, col in metrics]
    for i, bk in enumerate(bks):
        for j, (nm, suf, col) in enumerate(metrics):
            x = i * len(metrics) + j
            v = BKT["at_main_caliber"][bk + suf]
            ax.bar(x, v, width=0.78, color=col, zorder=3)
            datalabels(ax, [x], [v], ["%s" % v], color=INK)
    # 白刻度＝该桶 FAR 实际能取到的那几个离散值（画满柱内＝只有这些档位）
    far_steps = sorted({p["FAR"] for p in BKT["buckets"][bks[0]]["sweep"]})
    ax.hlines(far_steps, -0.34, 0.34, color="white", lw=0.7, zorder=4)
    ax.set_xticks([i * len(metrics) + (len(metrics) - 1) / 2.0 for i in range(len(bks))])
    ax.set_xticklabels([T("c.k1","固定机位监控","Fixed CCTV"), T("c.k2","其余源混合","Other sources")])
    ax.set_xlim(-0.5, len(bks) * len(metrics) - 0.5)
    ax.set_ylim(0, TOP)
    axis_labels(ax, None, T("c.y","读数（%）","value (%)"))
    real = BKT["buckets"][bks[0]]
    callout(ax, (0.39, 70), T("c.call","FAR 只有 %d\n个离散档" % real["distinct_FAR_values"],"Only %d\ndiscrete FARs" % real["distinct_FAR_values"]),
            (1.62, 55), color=ROLE["warn"], ha="left", va="center")
    ax.legend(handles=handles, fontsize=FS["legend"], loc="upper right", ncol=2,
              frameon=False, handlelength=1.4, columnspacing=1.0, borderpad=0)

    low = min(fr, key=lambda r: r["arms"][REF_ARM]["FAR"])      # 域外 FAR 最低的那个操作点
    save_fig(fig, "fig10_external",
             source_note=T("note",
                 "口径：%s，θ 格＝%s，随机线按 %d 个类算＝%s%%（只画在 %s：%s 的 crash 档有 GT＝%s）"
                         "，出数脚本 sha %s"
                     % (EXT["caliber"], EXT["theta_grid"], N_CLASS, "%.1f" % CHANCE,
                        DSET["DADA2000"], DSET["CCD"],
                        EXT["per_dataset"]["CCD"]["per_arm"][REF_ARM]["gt_n(normal/pre/crash)"][2],
                        FIG["instrument_sha16"]),
                 "Caliber: clip p_accident = 1 - p_normal, theta at GT=0; OOD FAR floor ~1/n_normal. "
                 "Chance line over %d classes = %s%% (drawn only on %s; %s crash GT = %s). Data script sha %s"
                 % (N_CLASS, "%.1f" % CHANCE, DSET["DADA2000"], DSET["CCD"],
                    EXT["per_dataset"]["CCD"]["per_arm"][REF_ARM]["gt_n(normal/pre/crash)"][2],
                    FIG["instrument_sha16"])))


if __name__ == "__main__":
    main()
