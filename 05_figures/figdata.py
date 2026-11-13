# -*- coding: utf-8 -*-
"""
图件出数入口 —— 所有数据图只从 figure_data.json 取数，脚本内不写任何字面读数。

为什么：正文改写时替换掉的那批数字，根因全部是"代际/口径错配"而不是算术错误
（正本 §16-113）。若把数字抄进绘图脚本，图就会成为第二套无人认领的出处。
本入口把 11 张图的每个数字钉在同一个出数面上，出数面再把每个数钉在册面的 sha16 上。

面路径可用环境变量 FIG_DATA 覆盖（默认读 meta/paper/FIGURES 下最新落面）。
"""
import glob
import json
import os
import struct

from style import *  # noqa: F401,F403  （plt / OI / ROLE / TINT / save / rbox / arrow / fit_text）

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", ".."))
DEFAULT_FACE_GLOB = os.path.join(PKG, "datasets", "accident_v26", "meta", "paper",
                                 "FIGURES", "*", "figure_data.json")


def load():
    p = os.environ.get("FIG_DATA")
    if not p:
        cands = [str(Path(__file__).parent / "data" / "figure_data.json")]
        if not cands:
            raise SystemExit("没有出数面：先跑 scripts/paper_figure_data.py --land，"
                             "或用 FIG_DATA=<figure_data.json> 指定")
        p = cands[-1]
    d = json.load(open(p, encoding="utf-8"))
    if d["gate"]["n_fatal"]:
        raise SystemExit("出数面带 %d 条 FATAL，拒绝画图：%s"
                         % (d["gate"]["n_fatal"], d["gate"]["fatals"][:3]))
    FIG["_face"] = p
    FIG["data"] = d["figures"]
    FIG["prov"] = d["figures"]["_provenance"]
    FIG["instrument_sha16"] = d["instrument_sha256_16"]
    print("数据面: %s  (出数脚本 %s)" % (p, d["instrument_sha256_16"]))
    return d["figures"]


FIG = {}


def _wrap_lines(fig, text, size, frac=0.985):
    """把出处行按真实像素宽折到图宽以内。

    为什么必须在图内折行：savefig 走 bbox="tight"，一行不折的长页脚会把画布
    从 7.05in 撑到 10.7in，入稿按栏宽缩放后全图字号被压掉三分之一——"看不清"
    的根因在这里，不在字号设置。分隔符（・，；/空格）后断行，不切断词。
    """
    import re
    avail = fig.get_size_inches()[0] * fig.dpi * frac
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=size)
    lines, cur = [], ""
    for chunk in [c for c in re.split(r"(?<=[・，,;；/ ])", text) if c]:
        probe.set_text(cur + chunk)
        if cur and probe.get_window_extent(renderer).width > avail:
            lines.append(cur)
            cur = chunk
        else:
            cur += chunk
    if cur:
        lines.append(cur)
    probe.remove()
    return lines


def footer(fig, text, color="#5F6368", size=5.8):
    """图底出处行：每张数据图都带，审稿人问"这个数从哪来"时图自己答。返回行数。"""
    lines = []
    for para in text.split("\n"):
        lines.extend(_wrap_lines(fig, para, size))
    fig.text(0.004, 0.004, "\n".join(lines), fontsize=size, color=color,
             ha="left", va="bottom")
    return len(lines)


def _short(path):
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    tail = parts[-2:]
    return "/".join(tail)


def provenance(fig, tags, extra=""):
    """把出数面的 sha16 直接印在图上——图上的每个读数自带出处，不靠正文回忆。"""
    bits = []
    for t in tags:
        p = FIG["prov"].get(t)
        if not p or not p.get("sha16"):
            continue
        bits.append("%s@%s" % (_short(p["path"]), p["sha16"][:8]))
    txt = "Sources: " + "  ·  ".join(bits)
    if extra:
        txt += ("  |  " + extra) if bits else extra
    footer(fig, txt)


def panel_title(ax, s, color="#202124"):
    ax.set_title(s, fontsize=8.0, color=color, loc="left", pad=4)


# ====================================================================
# 数据图统一样式（与示意图同一套：Arial / Okabe-Ito / Material 淡色）
# --------------------------------------------------------------------
# 四条硬纪律，违反任何一条就是"图太乱"的根因，脚本层直接拦：
#   ① 每个面板只讲一件事：标题＝读什么，副标题＝要表达的意思（一句话）；
#   ② 每个面板最多一个 callout（callout() 第二次调用即报错）；
#   ③ 禁双纵轴（twinx 直接抛错）——两种量纲就拆成两个面板；
#   ④ 字号统一 6.5–7.5pt，不许在单图里自定义更小/更大字号。
# ====================================================================
FS = {"tick": 6.5, "label": 7.0, "title": 7.5, "msg": 6.5,
      "legend": 6.0, "callout": 6.0, "value": 6.0}
GRID = "#DADCE0"          # 唯一允许的网格色（Material grey 300）
INK = "#202124"
MUTED = "#5F6368"
# 汉字由 style.py 的 font.family 回退链解决，脚本层不再逐字指定字体。

W_HALF, W_FULL = 3.45, 6.30   # 单栏 / 跨栏宽（英寸）：W_FULL＝合稿版心宽（A4・2.5cm 边距）
# 为什么是 6.30 而不是 7.05：图按"最终显示宽"声明，落盘宽＝显示宽 ⇒ 缩放系数恒 1，
# 图上的 6.5–7.5pt 就是纸上的 6.5–7.5pt。按 7.05 声明会被版心缩到 0.83 倍，
# 最小字号实际只剩 5.4pt——"看不清"是这么来的。投稿两栏版式若给到 7.16in，
# 只会等比放大，不会缩小，字号线永远不被击穿。


def _no_twin(self, *a, **k):
    raise RuntimeError("统一样式禁 twinx：两种量纲请拆成两个面板（见纪律 ③）")


def style_ax(ax):
    """去上右边框、淡点网格、统一刻度字号——所有数据面板先过这个。"""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(labelsize=FS["tick"], colors=INK, length=2.5, width=0.7)
    ax.grid(True, ls=":", lw=0.6, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_size(FS["label"])
    ax.yaxis.label.set_size(FS["label"])
    ax.twinx = _no_twin.__get__(ax, type(ax))
    return ax


def panel(ax, tag, title, message, accent=None):
    """标题（粗）＋一句话意思（彩色）＋统一坐标样式。返回 ax。"""
    accent = accent or ROLE["stage1"]
    style_ax(ax)
    ax.set_title("%s  %s" % (tag, title), fontsize=FS["title"], loc="left",
                 pad=11, color=INK, fontweight="bold")
    ax.text(0, 1.018, message, transform=ax.transAxes, fontsize=FS["msg"],
            color=accent, va="bottom", ha="left")
    ax._n_callouts = 0
    return ax


def axis_labels(ax, xlabel=None, ylabel=None):
    """统一轴标字号，含汉字时自动切 CJK 字体。"""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def callout(ax, xy, text, xytext, color=None, **kw):
    """每面板唯一的一条注释。第二次调用直接报错（纪律 ②）。"""
    if getattr(ax, "_n_callouts", 0) >= 1:
        raise RuntimeError("该面板已有 callout：一图一件事，请删注释或拆面板（纪律 ②）")
    ax._n_callouts += 1
    color = color or MUTED
    kw.setdefault("arrowprops", dict(arrowstyle="-", lw=0.7, color=color))
    kw.setdefault("fontsize", FS["callout"])
    kw.setdefault("color", color)
    return ax.annotate(text, xy, xytext=xytext, **kw)


def legend(ax, loc="best", ncol=1):
    lg = ax.legend(fontsize=FS["legend"], loc=loc, frameon=False, ncol=ncol,
                   handlelength=1.4, columnspacing=1.0, borderpad=0)
    return lg


def datalabels(ax, xs, ys, labels, fs=None, color=INK, dy=0.0, ha="center"):
    """把数值直接标在图形元素上，代替"读网格＋猜"。"""
    for x, y, s in zip(xs, ys, labels):
        ax.text(x, y + dy, s, fontsize=fs or FS["value"], color=color,
                ha=ha, va="bottom", zorder=5)


def figure_pair(n=2, w=None, h=2.6):
    fig, axes = plt.subplots(1, n, figsize=(w or (W_FULL if n == 2 else W_HALF), h))
    fig.subplots_adjust(hspace=0.46, wspace=0.30, left=0.10, right=0.985,
                        top=0.855, bottom=0.175)
    return fig, axes


def _png_width_in(path, dpi=600):
    """读 PNG IHDR 的真实像素宽（不引 PIL）——tight bbox 改了多少，只有落盘件自己知道。"""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("不是 PNG：%s" % path)
    return struct.unpack(">I", head[16:20])[0] / float(dpi)


def save_fig(fig, name, source_note=None):
    """落盘前先按页脚行数把底部留白让出来——页脚永远不许把画布撑宽。"""
    w = fig.get_size_inches()[0]
    n = footer(fig, source_note) if source_note else 0
    if n:
        h = fig.get_size_inches()[1]
        need = n * 5.8 * 1.32 / 72.0 / h + 0.095      # 页脚块 + 刻度/轴标
        if fig.subplotpars.bottom < need:
            fig.subplots_adjust(bottom=min(need, 0.38))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600)
    out_w = _png_width_in(os.path.join(OUT, name + ".png"))
    if out_w > w * 1.06:
        raise RuntimeError("画布被撑宽：输出 %.2fin ＞ 声明 %.2fin —— 有文本没折行"
                           "（入稿缩放会把全图字号压掉 %.0f%%）"
                           % (out_w, w, 100 * (1 - w / out_w)))
    print("saved: %s.pdf/.png -> %s  (图宽 %.2fin・页脚 %d 行)" % (name, OUT, out_w, n))

# Publication edition: keep data calculations, redesign presentation.
from pathlib import Path
import json
FS = {"tick":7, "label":8, "title":8.5, "msg":7, "legend":7, "callout":7, "value":7}
W_FULL = 7.16
GRID, INK, MUTED = "#dddddd", "#202020", "#606060"
NOTES = []
def footer(fig, text, **kwargs):
    NOTES.append(text)
    return 0
def panel(ax, tag, title, message, accent=None):
    style_ax(ax)
    ax.set_title(tag+"  "+title, loc="left", pad=10, fontsize=8.5, color=INK)
    NOTES.append(tag+" "+message)
    return ax
def callout(ax, xy, text, xytext, color=None, **kw):
    NOTES.append(text)
def figure_pair(n=2, w=None, h=2.6):
    if n == 3:
        fig, axes = plt.subplots(3,1,figsize=(W_FULL,7.8))
        fig.subplots_adjust(hspace=.65)
    else:
        fig, axes = plt.subplots(1,n,figsize=(W_FULL,3.2))
    return fig, axes
def save_fig(fig, name, source_note=None):
    if source_note: NOTES.append(source_note)
    if len(fig.axes)==3:
        fig.subplots_adjust(left=.21,right=.96,bottom=.08,top=.96,hspace=.65)
    else:
        fig.subplots_adjust(left=.15,right=.97,bottom=.22,top=.87,wspace=.75)
    # Keep curves distinguishable in grayscale as well as in color.
    for ax in fig.axes:
        seq=0
        for line in ax.lines:
            if len(line.get_xdata())>3 and line.get_linestyle()=='-':
                line.set_linestyle(['-','--','-.',':'][seq%4]); seq+=1
        for patch in ax.patches:
            if type(patch).__name__=='Rectangle' and patch.get_zorder()==3:
                patch.set_linewidth(.45); patch.set_edgecolor('#444444')
    record = {'figure':name,'language':FIG_LANG,'data_file':FIG.get('_face'),
              'instrument_sha16':FIG.get('instrument_sha16'),'notes':list(NOTES)}
    Path(OUT,name+'.notes.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    save(fig,name)
