# -*- coding: utf-8 -*-
"""图 6（§8.3 类型天花板）：课程外两类的零召回，决策层买回的代价。

一图一事（纪律 ①）：
  (a) 两个零召回类不是"难类学不动"，是课程从未激活（apply_mask 剔样本＋−∞ 槽位）；
  (b) 判前偏置能把那个类的召回买回来，代价是全类 type_acc 下滑。
(b) 两条线同为百分比 ⇒ 共用一把纵轴（纪律 ③ 只禁量纲不同的双轴，不是禁共轴）。
配色语义全图件统一：stage2＝课程内，stage1＝全类口径，warn＝结构性缺陷或代价。
面上的读数才上图（纪律 ④）；type_ceiling 面没有课程标记，逐类课程归属交叉核对
main_dashboard.type_per_class 的 in_curriculum，并核两面的召回是否同值。
"""
from figdata import *  # noqa

F = load()
D = F["type_ceiling"]
PC = D["per_class"]                                              # id(str)：name/gt_n/recall(%)…
CUR = {r["id"]: r for r in F["main_dashboard"]["type_per_class"]}  # 在册课程标记（交叉核对用）
IDS = sorted(int(k) for k in PC)
N_TYPE = len(IDS)
# 偏置扫取课程外里 GT 多的那类（面上 gt_n 大者＝rollover），不写死类名
SW = D["bias_sweep"][max(D["bias_sweep"], key=lambda k: PC[k]["gt_n"])]
CV = SW["curve"]
# 课程外机制的代码原文（面上原串，只剥掉括号说明，免把图底撑宽到超出跨栏尺寸）
MECH = "，".join(t.split("（")[0] for t in D["source_lines"].split("，"))
for i in IDS:                                                    # 两面对不上就不画
    assert abs(PC[str(i)]["recall"] - CUR[i]["recall"] * 100) < 0.05


def _tail(tag):
    """面尾：末两级路径@sha8——图自己回答"这个数从哪来"。"""
    p = FIG["prov"][tag]
    return "%s@%s" % ("/".join(p["path"].replace("\\", "/").split("/")[-2:]),
                      p["sha16"][:8])


def main():
    fig, axes = figure_pair(2, W_FULL, 2.7)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.845, bottom=0.185, wspace=0.30)

    # ---- (a) 逐类召回：课程外两类＝零，且零得整整齐齐（底色带标出那两行）
    ax = panel(axes[0], "(a)", T("a.t","逐类召回（全 %d 类）" % N_TYPE,"Per-class recall (%d)" % N_TYPE),
               T("a.m","不是“难类学不动”，是课程从未激活","Never activated, not hard"), ROLE["warn"])
    out = [i for i in IDS if not CUR[i]["in_curriculum"]]
    ax.axhspan(min(out) - 0.5, max(out) + 0.5, color=TINT["warn"], zorder=0)
    ax.barh(IDS, [PC[str(i)]["recall"] for i in IDS], height=0.6,
            color=[ROLE["warn"] if i in out else ROLE["stage2"] for i in IDS], zorder=3)
    ax.set_yticks(IDS)
    names={0:T('','追尾','Rear-end'),1:T('','侧碰','Side impact'),2:T('','翻覆','Rollover'),3:T('','撞行人','Pedestrian collision'),4:T('','正面碰撞','Head-on'),5:T('','撞固定物','Fixed object'),6:T('','其他','Other')}
    ax.set_yticklabels(["%s - n=%d" % (names[i],PC[str(i)]["gt_n"]) for i in IDS])
    datalabels(ax, [PC[str(i)]["recall"] + 1.6 for i in IDS], IDS,
               ["%.1f" % PC[str(i)]["recall"] for i in IDS], ha="left")
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    axis_labels(ax, T("a.x","该类召回（%）","Recall (%)"), None)                        # 类名与 GT n 已在刻度上
    callout(ax, (PC[str(out[0])]["recall"], sum(out) / len(out)),
            T("a.call","%.1f（课程外，零梯度＋−∞ 槽）" % PC[str(out[0])]["recall"],"%.1f (off-curriculum)" % PC[str(out[0])]["recall"]),
            (26, min(out) + 0.55), color=ROLE["warn"], ha="left")

    # ---- (b) 判前偏置扫：召回买回来，全类判定掉下去（同一把百分比尺）
    ax = panel(axes[1], "(b)", T("b.t","%s 判前偏置（决策层）" % SW["name"],"Pre-decision bias: %s" % SW["name"]),
               T("b.m","买回一个类的代价是牺牲整体判定","Buying one class costs all"), ROLE["warn"])
    bx = [c["bias"] for c in CV]
    rec = [c["class_recall"] for c in CV]
    tacc = [c["type_acc_all"] for c in CV]
    ax.plot(bx, rec, "-o", color=ROLE["stage2"], lw=1.4, ms=3.6, label=T("b.l1","%s 类召回" % SW["name"],"%s recall" % SW["name"]))
    ax.plot(bx, tacc, "-s", color=ROLE["stage1"], lw=1.4, ms=3.2,
            label=T("b.l2","类型正确产出率","Correct-type yield"))
    k = max(range(len(CV)), key=lambda i: rec[i])                # 网格最优＝召回最高那格
    ax.plot([bx[k]], [rec[k]], "o", ms=8.5, mfc="none", mec=ROLE["warn"], mew=1.6)
    axis_labels(ax, T("b.x","判前偏置（bias 格点）","Decision bias"), T("b.y","百分比（%）","%"))
    ax.set_xticks([b for b in bx if float(b) == int(b)])        # 只留整数格点，稀化免挤
    ax.set_xlim(bx[0] - 0.25, bx[-1] + 0.25)
    ax.set_ylim(0, 105)
    callout(ax, (bx[k], rec[k]), T("b.call","召回 %.1f%%\n代价 type_acc %.2f%%" % (rec[k], tacc[k]),"recall %.1f%%\ncost %.2f%%" % (rec[k], tacc[k])),
            (bx[0] + 0.05, rec[k] + 5), color=ROLE["warn"], ha="left")
    legend(ax, loc="lower right")

    save_fig(fig, "fig06_type_ceiling",
             source_note=T("note",
                 "口径：全 %d 类 clip 级判定（n＝%d，type_acc_all %.2f），课程外机制原文 %s。"
                         "出数脚本 sha %s\n出处：%s"
                         % (N_TYPE, D["base"]["n_rows"], D["base"]["type_acc_all"], MECH,
                            FIG["instrument_sha16"],
                            "  ·  ".join(_tail(t) for t in ("type_ceiling", "type_perclass"))),
                 "Caliber: all %d classes, clip-level (n=%d, type_acc_all %.2f); off-curriculum via apply_mask + -inf slots (train_config.py:368-374). Data script sha %s\nSources: %s"
                         % (N_TYPE, D["base"]["n_rows"], D["base"]["type_acc_all"],
                            FIG["instrument_sha16"],
                            "  ·  ".join(_tail(t) for t in ("type_ceiling", "type_perclass")))))


if __name__ == "__main__":
    main()
