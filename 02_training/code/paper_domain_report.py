#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_domain_report.py — scene 头分域读数落盘仪器 (论文 §7 口径核定 D3 / Table 主表分域列)

零 GPU、只读现库、只写 meta/domain/。不动训练、不动标签、不动 split、不动配方。

三件事:
  1. 把训练日志里逐 epoch 的 `[Val] Scene cls by domain` + `[Val] Scene cls` 读数
     落成机器可读面 (jsonl + csv), 主表今后必须分域报, 不再只报总 Acc。
  2. 口径同源闸: 盘上按 detection.py :968-973 (tile 排除) 与 :990-994 (real/syn)
     的同一判据独立重算 real/syn 分母, 与日志打印的 n 对账; 相等即同源,
     有差额则必须逐帧归因到 #90 重导清单且归因恰好等于差额, 归不上就 FATAL。
     ("同判据"是被数字证明的, 不是注释承诺的 —— 判例 148)
  3. 窗口形态共线面: 从 clip 分母账本算 分域 × scene 类 的满窗率/均帧数,
     证明 clip 级时序输入形态与域共线 (§7 D3 主动交代项)。

子命令:
  --build   重算并落盘 (幂等)
  --check   重算并与落盘面比对, 漂移即 FATAL (易变字段除外)
  --report  打印人话摘要
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
PKG_ROOT = os.environ.get('V26_PKG_ROOT', '.')

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / 'datasets' / 'accident_v26'
OUT = DATA / 'meta' / 'domain'
LEDGER_DIR = DATA / 'meta' / 'crc'
IMAGES = DATA / 'images'
SCENE_LABELS = DATA / 'scene_labels'
DETECTION_PY = PKG / 'scripts' / 'core' / 'detection.py'
DATASET_PY = PKG / 'scripts' / 'core' / 'dataset.py'
IMPORT_MANIFEST = DATA / 'meta' / 'synthgt' / 'parent_reimport_manifest.jsonl'
MEASURED_ON = time.strftime('%Y-%m-%d')   # 面重算一次就印一次当天，不许冻在旧日期 (判例 148)

# (日志, 该日志对应的 run 目录) —— 主线的四个阶段日志, 按时间顺序
LOGS = [
    (PKG / 'runs' / 'v26_dw_phase1.log', PKG / 'runs' / 'v26_dw_p1', 'phase1'),
    (PKG / 'runs' / 'v26_dw_phase2.log', PKG / 'runs' / 'v26_dw_p2', 'phase2'),
    (PKG / 'runs' / 'v26_dw_phase2_resume.log', PKG / 'runs' / 'v26_dw_p2', 'phase2'),
    (Path(PKG_ROOT) / 'accident3d' / 'dryrun' / 'pass19_resume_p2.log',
     PKG / 'runs' / 'v26_dw_p2', 'phase2'),
]

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
# 训练侧 epoch 锚: "  13/60      21.4G ..." —— G 前缀排除 Instances/Size 列的假匹配
ITER_RE = re.compile(r'^\s*(\d+)/(\d+)\s+[\d.]+G\s')
# detection.py:968-973 的 tile 判据, 逐字复刻
TILE_RE = re.compile(r'_t\d{2}\.[A-Za-z]+$')

ALL_RE = re.compile(
    r'^\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
SCENE_RE = re.compile(
    r'^\s*\[Val\] Scene cls: Acc=([\d.]+)% \| CE=([\d.]+) \| '
    r'normal: R=([\d.]+)% P=([\d.]+)% \| pre_crash: R=([\d.]+)% P=([\d.]+)% \| '
    r'crash: R=([\d.]+)% P=([\d.]+)% \| PDR=([\d.]+)% FAR=([\d.]+)%')
DOM_RE = re.compile(
    r'^\s*\[Val\] Scene cls by domain: '
    r'(?:real Acc=([\d.]+)% \(n=(\d+)\) \| '
    r'real R\(n/p/c\)=([\d.]+)/([\d.]+)/([\d.]+) \| )?'
    r'syn Acc=([\d.]+)% \(n=(\d+)\)')

CLIP_LEN = 30       # paper_crc_calib.py 造 clip_ledger_*.jsonl 时传给 ClipDataset 的值
SCENE_NAMES = {0: 'ignore', 1: 'normal', 2: 'pre_crash', 3: 'crash'}

# 本仪器 clip 面的几何归属 —— 实测证据, 不是注释承诺 (判例 148)
CLIP_GEOMETRY_ATTRIBUTION = {
    'ledger_geometry': 'clip_len=30 / stride=10 (paper_crc_calib.py:25, 由 base_config.py '
                       'unified-v21 等 V21 数据源档案的设计值带入)',
    'live_training_geometry': 'clip_len=16 / stride=8',
    'evidence': [
        'run_train.py:65 `--clip-len` 默认 16',
        'run_train.py:82 无条件把 `--clip-len <值>` 拼进 sys.argv 交给 train.py '
        '⇒ train.py:741 `args.clip_len is not None` 成立, 走不进 :744 那条 '
        'ds_config.get("clip_len", 30) 分支',
        'base_config.py DATASET_CONFIGS 里没有 accident_v26 / accident-both 档案 '
        '(训练日志头打印 "默认数据源: accident-both")',
        'eval_acceptance.py:37 同口径声明 "run_train.py 的 --clip-len 默认是 16, ClipDataset 默认也是 16"',
    ],
    'consequence': 'clip_supervision_denominators.json / window_shape_collinearity.json 的 clip '
                   '总体 (train 59,299 / val 15,597) 不是 Phase 3 实跑的 16/8 总体 '
                   '(qc_effective_clip_census 同宇宙实测账面 train 67,003 / val 17,684) '
                   '⇒ 引用这两个面必须带本条; §1.1 完整性要出生产数, '
                   '得在 16/8 + T110 规则下重算 (待办: census 侧补 scene==0 退出计数)',
    'census_side_gap': 'qc_effective_clip_census.py:212 仍是 T110 之前的回退 '
                       '`[s for s in sc_list if s != 0] or [1]`; 对 max==3 的 crash 判定无影响 '
                       '(全 ignore 两种写法都不算 crash), 但出 "退出监督" 计数前必须先改成 scene==0',
    'ruling_at': '2026-10-10 (#148 定性：读代码得出的几何归属，不随盘变，故不现读)',
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_text(p: Path) -> str:
    return ANSI_RE.sub('', p.read_text(encoding='utf-8', errors='replace')).replace('\r', '\n')


# ---------------------------------------------------------------- 域判据 (与产生者同源)
def frame_domain(basename: str) -> str:
    """detection.py:993 = basename.startswith('real_')。其余一律进 'syn' 桶。"""
    return 'real' if basename.startswith('real_') else 'syn'


def frame_family(basename: str) -> str:
    for pre, name in (('real_', 'cctv_real_prefix'), ('realv', 'realv_collection'),
                      ('synthetic', 'carla_synthetic'), ('tum', 'tumtraf_dashcam'),
                      ('tv', 'tv_collection')):
        if basename.startswith(pre):
            return name
    return 'other'


def is_tile(basename: str) -> bool:
    return TILE_RE.search(basename) is not None


def scene_label_of(stem: str, split: str):
    p = SCENE_LABELS / split / f'{stem}.txt'
    if not p.exists():
        return None
    toks = p.read_text().split()
    return int(toks[0]) if toks else None


def frame_denominators(split: str):
    """盘上独立重算: 与 detection.py 的 valid_mask 完全同判据的帧数分母。
    一次扫盘同时给出 (domain, family, tile, valid) 与 (family, tile, class) 两张表。
    第三张 stems 与 counts 同一循环产出 —— 闸 1 要做逐帧归因, 不能只拿一个数 (判例 146)。"""
    counts = collections.Counter()
    byclass = collections.Counter()
    stems = collections.defaultdict(set)
    for f in sorted(os.listdir(IMAGES / split)):
        stem = os.path.splitext(f)[0]
        lab = scene_label_of(stem, split)
        tile = 'tile' if is_tile(f) else 'plain'
        fam = frame_family(f)
        key = (frame_domain(f), fam, tile,
               'valid' if lab not in (0, None) else 'ignore')
        counts[key] += 1
        stems[key].add(stem)
        if lab not in (0, None):
            byclass[(fam, tile, SCENE_NAMES[lab])] += 1
    return counts, byclass, stems


def reimport_stems(split: str):
    """#90 重导清单里 target_split 的 parent 帧 stem 集合 (kind=='import')。
    闸 1 用它把"盘上比日志多"落到具体帧, 免得闸变成常红 (判例 145/147)。"""
    out = set()
    if not IMPORT_MANIFEST.exists():
        return out
    with open(IMPORT_MANIFEST, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('kind') == 'import' and r.get('target_split') == split:
                out.add(r['parent'])
    return out


def bucket_stems(stems: dict, domain: str):
    """同判据下的 (domain, *, plain, valid) 帧集合 —— 与日志 n 可比的量。"""
    s = set()
    for k, v in stems.items():
        if k[0] == domain and k[2] == 'plain' and k[3] == 'valid':
            s |= v
    return s


# ---------------------------------------------------------------- 日志解析
def parse_log(log: Path, phase: str, rundir: Path):
    """逐 epoch 取 (scene 总指标, 分域读数)。epoch 归属: 训练进度行 N/total,
    再用同 run 的 results.csv 交叉核对 mAP50-95。"""
    csv_rows = {}
    rc = rundir / 'results.csv'
    if rc.exists():
        with open(rc, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                csv_rows[int(r['epoch'])] = r

    lines = clean_text(log).split('\n')
    recs, cur_ep, last_all, last_scene, prev_dom = [], None, None, None, None
    for line in lines:
        m = ITER_RE.match(line)
        if m:
            cur_ep = int(m.group(1))
            continue
        ma = ALL_RE.match(line)
        if ma:
            last_all = {'images': int(ma.group(1)), 'instances': int(ma.group(2)),
                        'precision_B': float(ma.group(3)), 'recall_B': float(ma.group(4)),
                        'mAP50_B': float(ma.group(5)), 'mAP50_95_B': float(ma.group(6))}
            continue
        ms = SCENE_RE.match(line)
        if ms:
            g = [float(x) for x in ms.groups()]
            last_scene = {'acc': g[0], 'ce': g[1],
                          'normal_R': g[2], 'normal_P': g[3],
                          'pre_crash_R': g[4], 'pre_crash_P': g[5],
                          'crash_R': g[6], 'crash_P': g[7],
                          'PDR': g[8], 'FAR': g[9]}
            continue
        md = DOM_RE.match(line)
        if md:
            if line == prev_dom:
                continue
            prev_dom = line
            g = md.groups()
            rec = {'phase': phase, 'epoch': cur_ep, 'log': str(log),
                   'real_acc': None if g[0] is None else float(g[0]),
                   'real_n': None if g[1] is None else int(g[1]),
                   'real_R_normal': None if g[2] is None else float(g[2]),
                   'real_R_pre_crash': None if g[3] is None else float(g[3]),
                   'real_R_crash': None if g[4] is None else float(g[4]),
                   'syn_acc': float(g[5]), 'syn_n': int(g[6])}
            if last_scene:
                rec.update(last_scene)
            if last_all:
                rec.update(last_all)
            recs.append(rec)
            continue
        if line.strip() and not line.startswith('  [Val]'):
            prev_dom = None

    # epoch 交叉核对: 日志进度行给的 N 必须与 results.csv 第 N 行的 mAP50-95 对得上
    for rec in recs:
        e = rec['epoch']
        row = csv_rows.get(e)
        if row is None or 'mAP50_95_B' not in rec:
            rec['epoch_check'] = 'no_csv_row'
            continue
        if abs(round(float(row['metrics/mAP50-95(B)']), 3) - rec['mAP50_95_B']) <= 1e-9:
            rec['epoch_check'] = 'agree'
        else:
            rec['epoch_check'] = f"CONFLICT csv={float(row['metrics/mAP50-95(B)']):.3f} " \
                                 f"log={rec['mAP50_95_B']:.3f}"
    return recs


# ---------------------------------------------------------------- 窗口形态共线
def window_collinearity(split: str):
    """clip 分母账本 → 分域 × 族 × clip scene 的满窗率/均帧数。
    clip scene = scene_max (core/dataset.py:747 先剔 ignore 再取最高风险),
    scene_max==0 即"整条 clip 全 ignore" ⇒ T110 现行口径下 clip_scene=0,
    三路损失都跳过 0 ⇒ 该 clip 退出 scene 监督（旧口径才回退 normal(1)）。"""
    rows = []
    tab = collections.defaultdict(lambda: {'clips': 0, 'sum_n': 0, 'full': 0})
    fallback = collections.Counter()
    fam_tab = collections.defaultdict(lambda: {'clips': 0, 'sum_n': 0, 'full': 0})
    with open(LEDGER_DIR / f'clip_ledger_{split}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            stem = d['first']
            dom, fam = frame_domain(stem), frame_family(stem)
            sc = int(d['scene_max'])
            k = (dom, sc)
            tab[k]['clips'] += 1
            tab[k]['sum_n'] += d['n']
            tab[k]['full'] += (1 if d['n'] >= CLIP_LEN else 0)
            fk = (dom, fam, sc)
            fam_tab[fk]['clips'] += 1
            fam_tab[fk]['sum_n'] += d['n']
            fam_tab[fk]['full'] += (1 if d['n'] >= CLIP_LEN else 0)
            if sc == 0:
                fallback[(dom, fam)] += 1
    for (dom, sc), v in sorted(tab.items()):
        rows.append({'split': split, 'domain': dom, 'clip_scene': SCENE_NAMES[sc],
                     'clips': v['clips'], 'mean_frames': round(v['sum_n'] / v['clips'], 1),
                     'full_window_pct': round(100 * v['full'] / v['clips'], 1)})
    fam_rows = []
    for (dom, fam, sc), v in sorted(fam_tab.items()):
        fam_rows.append({'split': split, 'domain_by_prefix': dom, 'family': fam,
                         'clip_scene': SCENE_NAMES[sc], 'clips': v['clips'],
                         'mean_frames': round(v['sum_n'] / v['clips'], 1),
                         'full_window_pct': round(100 * v['full'] / v['clips'], 1)})
    fb = [{'split': split, 'domain': dom, 'family': fam,
           'clips_all_ignore_fallback_normal': n}
          for (dom, fam), n in sorted(fallback.items())]
    return rows, fam_rows, fb


def clip_label_supply(split: str):
    """旧口径账：多少 clip 的 scene 标签来自"整条零标注 → 回退 normal"（T110 已改，
    现行监督分母用 clip_supervision()）。"""
    g = collections.defaultdict(lambda: {'n': 0, 'fam': None, 'sm': 0})
    with open(LEDGER_DIR / f'clip_ledger_{split}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            e = g[d['video_id']]
            e['n'] = max(e['n'], d['n'])
            e['fam'] = frame_family(d['first'])
            e['sm'] = max(e['sm'], int(d['scene_max']))
    clips = collections.Counter()
    with open(LEDGER_DIR / f'clip_ledger_{split}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            clips[SCENE_NAMES[int(d['scene_max'])]] += 1
    n_clips = sum(clips.values())
    fb_clips = clips['ignore']
    normal_clips = clips['normal']
    return {
        'clip_scene_hist_before_fallback': dict(clips),
        'clips_total': n_clips,
        'all_ignore_clips_become_normal': fb_clips,
        'all_ignore_share_pct': round(100 * fb_clips / n_clips, 1),
        'normal_share_pct_before_fallback': round(100 * normal_clips / n_clips, 1),
        'normal_share_pct_after_fallback':
            round(100 * (normal_clips + fb_clips) / n_clips, 1),
        'video_groups': len(g),
    }


def clip_supervision(split: str):
    """T110 之后 clip 级 scene 监督的真实分母（分域）。
    现行规则 core/dataset.py:747：整条 clip 零标注 ⇒ clip_scene=0 ⇒ 三路损失
    (focal ignore_index=0 / cause scene>=2 / type scene==3) 都跳过 0
    ⇒ 该 clip 整体退出 scene 监督。旧口径（回退 normal(1)）把同一批 clip 算进
    normal 正例，两口径分母差一倍以上 ⇒ 主表必须分列，否则跨阶段不可比。"""
    tab = collections.defaultdict(
        lambda: {'clips': 0, 'supervised': 0, 'normal': 0, 'pre_crash': 0, 'crash': 0})
    with open(LEDGER_DIR / f'clip_ledger_{split}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sc = int(d['scene_max'])
            e = tab[frame_domain(d['first'])]
            e['clips'] += 1
            if sc > 0:
                e['supervised'] += 1
                e[SCENE_NAMES[sc]] += 1
    rows = []
    for dom in ('real', 'syn'):
        e = tab[dom]
        rows.append({
            'split': split, 'domain': dom, 'clips': e['clips'],
            'clips_supervised': e['supervised'],
            'supervised_pct': round(100 * e['supervised'] / e['clips'], 1),
            'clips_exit_supervision': e['clips'] - e['supervised'],
            'normal': e['normal'], 'pre_crash': e['pre_crash'], 'crash': e['crash'],
            'crash_share_of_supervised_pct':
                round(100 * e['crash'] / e['supervised'], 1) if e['supervised'] else None,
        })
    tot_clips = sum(r['clips'] for r in rows)
    tot_sup = sum(r['clips_supervised'] for r in rows)
    if tot_sup != sum(r['normal'] + r['pre_crash'] + r['crash'] for r in rows):
        sys.exit(f'[FATAL] {split}: 有标注 clip 自账不平 (监督 {tot_sup} ≠ 三类之和)')
    # 域桶只有 real_/其余两档，realv 与 tum/tv 全被塞进 'syn' ⇒ 单看 domain 列会误读，
    # 因此同一份账再按族拆一遍（族判据与 frame_family 同源）。
    ftab = collections.defaultdict(
        lambda: {'clips': 0, 'supervised': 0, 'normal': 0, 'pre_crash': 0, 'crash': 0})
    with open(LEDGER_DIR / f'clip_ledger_{split}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sc = int(d['scene_max'])
            e = ftab[frame_family(d['first'])]
            e['clips'] += 1
            if sc > 0:
                e['supervised'] += 1
                e[SCENE_NAMES[sc]] += 1
    fam_rows = [{
        'split': split, 'family': fam,
        'domain_by_prefix': 'real' if fam == 'cctv_real_prefix' else 'syn',
        'clips': e['clips'], 'clips_supervised': e['supervised'],
        'supervised_pct': round(100 * e['supervised'] / e['clips'], 1),
        'clips_exit_supervision': e['clips'] - e['supervised'],
        'normal': e['normal'], 'pre_crash': e['pre_crash'], 'crash': e['crash'],
    } for fam, e in sorted(ftab.items())]
    if sum(r['clips'] for r in fam_rows) != tot_clips:
        sys.exit(f'[FATAL] {split}: 族面与域面总数不等 (族 {sum(r["clips"] for r in fam_rows)} vs 域 {tot_clips})')
    return {'clips_total': tot_clips,
            'clips_supervised': tot_sup,
            'supervised_pct': round(100 * tot_sup / tot_clips, 1),
            'clips_exit_supervision': tot_clips - tot_sup,
            'crash_supervised': sum(r['crash'] for r in rows),
            'normal_supervised': sum(r['normal'] for r in rows),
            'by_domain': rows,
            'by_family': fam_rows}


def full_window_counterfactual(split: str):
    """"把短窗组做成 30 帧满窗"能补多少 clip，代价是多少帧（采集侧口径）。
    注意：新增帧本身没有 scene 标注 ⇒ 收益的上限是"沿用组内标签外推"，
    而外推等于把未标注帧当已标注，与 dataset.py:748 的回退是同一类错误。"""
    g = collections.defaultdict(lambda: {'n': 0, 'fam': None, 'sm': 0})
    with open(LEDGER_DIR / f'clip_ledger_{split}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            e = g[d['video_id']]
            e['n'] = max(e['n'], d['n'])
            e['fam'] = frame_family(d['first'])
            e['sm'] = max(e['sm'], int(d['scene_max']))
    rows = []
    for key in sorted({(e['fam'], SCENE_NAMES[e['sm']]) for e in g.values()}):
        fam, sc = key
        grp = [e for e in g.values() if e['fam'] == fam and SCENE_NAMES[e['sm']] == sc
               and e['n'] < CLIP_LEN]
        if not grp:
            continue
        have = sum(e['n'] for e in grp)
        rows.append({'split': split, 'family': fam, 'clip_scene': sc,
                     'short_groups': len(grp), 'frames_present': have,
                     'frames_needed_to_full_window': CLIP_LEN * len(grp) - have,
                     'full_window_clips_gained': len(grp)})
    return rows


def domain_predicate_conflict():
    """两条并存的域判据 —— 互相矛盾，必须写进 §7。"""
    return {
        'A_eval_side': {
            'file': str(DETECTION_PY.relative_to(PKG)), 'md5': md5(DETECTION_PY),
            'rule': "basename.startswith('real_') → real；其余全部 → 'syn' 桶",
            'refs': ':990-994 (is_real), :968-973 (is_tile), :964 (valid), :1144 (print)',
            'effect': 'TUMTraf 真实行车记录仪 / tv 合集 / realv 合集 被算进 "syn"',
        },
        'B_phase3_training_side': {
            'file': 'scripts/train_data.py',
            'md5': md5(PKG / 'scripts' / 'train_data.py'),
            'rule': "video_id.startswith('synthetic'/'synth') → 1(synthetic)；其余 → 0(real)",
            'refs': 'V24ClipDataset._infer_domain (train_data.py:54-64)，供 CDCR 使用',
            'effect': 'TUMTraf / tv / realv 被算进 real —— 与 A 恰好相反',
        },
        'verdict': '同一个"域"字在评估面与 Phase 3 正则面里指向不同的帧集合；'
                   '论文任何"跨域/域差"结论都必须先固定用哪一条，且不能叫 real vs synthetic。',
    }


# ---------------------------------------------------------------- 落盘
_WRITE = True          # --check 只算不写

def write_text(path: Path, text: str):
    if not _WRITE:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)
    with open(path, encoding='utf-8') as f:
        if f.read() != text:
            sys.exit(f'[FATAL] 回读不一致: {path}')


def write_json(path: Path, obj):
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + '\n')


def write_csv(path: Path, rows, cols):
    lines = [','.join(cols)]
    for r in rows:
        lines.append(','.join('' if r.get(c) is None else str(r[c]) for c in cols))
    write_text(path, '\n'.join(lines) + '\n')


def build():
    recs = []
    for log, rundir, phase in LOGS:
        if not log.exists():
            print(f'[warn] 日志缺失, 跳过: {log}')
            continue
        recs.extend(parse_log(log, phase, rundir))
    if not recs:
        sys.exit('[FATAL] 日志里一条分域读数都没有 —— 训练侧分域闸没打开')

    # 同一 (phase, epoch) 若来自两个日志, 读数必须一致 (resume 重叠面)
    by_pe = collections.defaultdict(list)
    for r in recs:
        by_pe[(r['phase'], r['epoch'])].append(r)
    conflicts = []
    for (ph, ep), group in sorted(by_pe.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        sig = {(g['real_acc'], g['real_n'], g['syn_acc'], g['syn_n']) for g in group}
        if len(sig) > 1:
            conflicts.append({'phase': ph, 'epoch': ep, 'readings': sorted(map(str, sig))})

    # 闸 1: 分母同源 —— 日志 n 必须等于盘上按同判据重算的 valid 帧数。
    #   差额不再直接判死, 也不无声放过: 必须逐帧归因到 #90 重导清单里、且归因数恰好等于差额,
    #   否则 FATAL (判例 145/147 —— 常红的闸等于没有闸, 常绿的闸也等于没有闸)。
    scan = {sp: frame_denominators(sp) for sp in ('train', 'val')}
    denom = {sp: scan[sp][0] for sp in scan}
    byclass = {sp: scan[sp][1] for sp in scan}
    stems = {sp: scan[sp][2] for sp in scan}
    for sp in ('train', 'val'):
        bad = [(k, len(stems[sp].get(k, ())), v)
               for k, v in denom[sp].items() if len(stems[sp].get(k, ())) != v]
        if bad:
            sys.exit(f'[FATAL] [{sp}] 帧集合与计数器不同源: {bad[:3]}')
    gates = {}
    for sp in ('train', 'val'):
        d = denom[sp]
        gates[sp] = {
            'real_valid_plain': sum(v for k, v in d.items()
                                    if k[0] == 'real' and k[2] == 'plain' and k[3] == 'valid'),
            'syn_valid_plain': sum(v for k, v in d.items()
                                   if k[0] == 'syn' and k[2] == 'plain' and k[3] == 'valid'),
            'frames_total': sum(d.values()),
        }
    val_n = {r['real_n'] for r in recs if r['real_n'] is not None}
    val_s = {r['syn_n'] for r in recs}
    imp = {sp: reimport_stems(sp) for sp in ('train', 'val')}

    def attribute(domain, log_ns):
        disk = gates['val'][f'{domain}_valid_plain']
        s = bucket_stems(stems['val'], domain)
        if len(log_ns) != 1:
            return {'status': 'FATAL', 'reason': f'日志里 {domain} 的 n 不止一个值 {sorted(log_ns)} '
                    f'⇒ 训练中途换过数据集, 分域账不可比'}
        n = next(iter(log_ns))
        gap = disk - n
        base = {'log_n': n, 'disk': disk, 'gap': gap}
        if gap == 0:
            return {**base, 'status': 'agree', 'how': '逐字相等'}
        if gap < 0:
            return {**base, 'status': 'FATAL',
                    'reason': '日志比盘上还多 ⇒ 日志对应的帧已被删, 面不可复算'}
        hit = s & imp['val']
        if not imp['val']:
            return {**base, 'status': 'FATAL', 'reason': f'差额 {gap} 无清单可归因 '
                    f'(缺 {IMPORT_MANIFEST})'}
        if gap == len(hit):
            return {**base, 'status': 'agree', 'how': '差额逐帧归因到 #90 重导清单',
                    'attributed_reimport_stems': len(hit),
                    'reimport_manifest_val': len(imp['val']),
                    'reimport_not_in_this_bucket': len(imp['val'] - s),
                    'log_vintage': 'pre-#90'}
        return {**base, 'status': 'FATAL', 'attributed_reimport_stems': len(hit),
                'unexplained': gap - len(hit),
                'reason': f'差额 {gap} 里只有 {len(hit)} 帧能归因到重导清单, '
                          f'剩 {gap - len(hit)} 帧说不清来源 ⇒ 判据或账本有问题'}

    g1 = {dom: attribute(dom, ns) for dom, ns in (('real', val_n), ('syn', val_s))}
    gate1 = {
        'log_real_n': sorted(val_n), 'disk_real_valid_plain': gates['val']['real_valid_plain'],
        'log_syn_n': sorted(val_s), 'disk_syn_valid_plain': gates['val']['syn_valid_plain'],
        'by_domain': g1,
        'agree': all(v['status'] == 'agree' for v in g1.values()),
    }
    if not gate1['agree']:
        sys.exit('[FATAL] 口径不同源: ' + '; '.join(
            f"{dom} 日志 n={v.get('log_n')} vs 盘上 {v.get('disk')} ⇒ {v.get('reason')}"
            for dom, v in g1.items() if v['status'] == 'FATAL'))
    print('[闸1] ' + ' | '.join(f"{dom}: n={v['log_n']} 盘上={v['disk']} {v['how']}"
                                + (f" (归因 {v['attributed_reimport_stems']} 帧)"
                                   if 'attributed_reimport_stems' in v else '')
                                for dom, v in g1.items()))

    # "syn" 桶的语义: 逐族拆开, 让"真实域 vs 合成域"这句话可被证伪或证实
    syn_family_mix = {sp: {} for sp in ('train', 'val')}
    for sp in ('train', 'val'):
        tot = 0
        for k, v in denom[sp].items():
            if k[0] == 'syn' and k[2] == 'plain' and k[3] == 'valid':
                syn_family_mix[sp][k[1]] = v
                tot += v
        syn_family_mix[sp]['total'] = tot
        syn_family_mix[sp]['pct'] = {
            f: round(100 * v / tot, 1) for f, v in syn_family_mix[sp].items() if f != 'total'}

    epoch_checks = collections.Counter(r.get('epoch_check', 'n/a') for r in recs)
    collinearity, fam_collinearity, fallback = {}, {}, {}
    for sp in ('train', 'val'):
        rows, frows, fb = window_collinearity(sp)
        collinearity[sp] = rows
        fam_collinearity[sp] = frows
        fallback[sp] = fb

    trend = {}
    for ph in ('phase1', 'phase2'):
        sub = [r for r in recs if r['phase'] == ph and r['epoch'] and r['real_acc'] is not None]
        if not sub:
            continue
        first, last = sub[0], sub[-1]
        trend[ph] = {
            'epochs': [first['epoch'], last['epoch']],
            'real_acc_first_last': [first['real_acc'], last['real_acc']],
            'syn_acc_first_last': [first['syn_acc'], last['syn_acc']],
            'real_normal_R_first_last': [first.get('real_R_normal'), last.get('real_R_normal')],
            'real_pre_crash_R_first_last': [first.get('real_R_pre_crash'), last.get('real_R_pre_crash')],
            'real_crash_R_first_last': [first.get('real_R_crash'), last.get('real_R_crash')],
            'gap_acc_last': round(last['syn_acc'] - last['real_acc'], 1),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ['phase', 'epoch', 'acc', 'ce', 'normal_R', 'normal_P', 'pre_crash_R', 'pre_crash_P',
            'crash_R', 'crash_P', 'PDR', 'FAR', 'real_acc', 'real_n', 'real_R_normal',
            'real_R_pre_crash', 'real_R_crash', 'syn_acc', 'syn_n', 'mAP50_B', 'mAP50_95_B',
            'precision_B', 'recall_B', 'epoch_check', 'log']
    write_csv(OUT / 'scene_by_domain_epochs.csv', recs, cols)
    write_json(OUT / 'scene_by_domain_epochs.json', {
        'measured_on': MEASURED_ON,
        'what': 'scene 头逐 epoch 总指标 + 分域 (real/syn) 读数, 全部来自训练日志, 非人工填写',
        'producer': {
            'file': str(DETECTION_PY.relative_to(PKG)), 'md5': md5(DETECTION_PY),
            'domain_predicate': "os.path.basename(im_file).startswith('real_')  [:990-994]",
            'tile_predicate': r"re.search(r'_t\d{2}\.[A-Za-z]+$')  [:968-973]",
            'valid_predicate': 'scene_targets != 0  [:964]',
            'printed_by': 'dom_msg  [:1144]',
        },
        'epoch_anchor': '训练进度行 N/total (剥 ANSI、\\r→\\n 后取最后一个匹配), '
                        '再用同 run results.csv 的 metrics/mAP50-95(B) 交叉核对',
        'epoch_check_counts': dict(epoch_checks),
        'gate1_denominator_agreement': gate1,
        'gate2_resume_overlap_conflicts': conflicts,
        'CRITICAL_domain_label_semantics': {
            'finding': "日志里的 'syn' 桶 = 文件名非 real_ 前缀的一切, 其中 TUMTraf 真实行车记录仪 "
                       "占多数; 它不是 '合成域'。'real' 桶 = 固定机位 CCTV。",
            'val_syn_bucket_family_mix': syn_family_mix['val'],
            'train_syn_bucket_family_mix': syn_family_mix['train'],
            'consequence': '禁止在论文里把这两个域写成 real vs synthetic; '
                           '应写成 fixed-camera CCTV (n 小) vs 其余源 (含 TUMTraf 自车视角), '
                           '或直接按族报。',
        },
        'trend': trend,
        'rows': recs,
    })
    faces = {
        'clip_label_supply': {sp: clip_label_supply(sp) for sp in ('train', 'val')},
        'clip_supervision': {sp: clip_supervision(sp) for sp in ('train', 'val')},
        'full_window_counterfactual': {sp: full_window_counterfactual(sp)
                                       for sp in ('train', 'val')},
        'domain_predicate_conflict': domain_predicate_conflict(),
    }
    write_json(OUT / 'frame_denominators.json', {
        'measured_on': MEASURED_ON,
        'method': '盘上独立重算 images/<split> 帧名 + scene_labels/<split>/*.txt 首字段, '
                  '判据逐字复刻 detection.py 的 valid_mask/is_tile/is_real',
        'splits': {sp: dict(
                       {'by_domain_family_tile_valid':
                        {f'{k[0]}|{k[1]}|{k[2]}|{k[3]}': v for k, v in sorted(denom[sp].items())},
                        'labeled_frames_by_family_tile_class':
                        {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in sorted(byclass[sp].items())},
                        **gates[sp]}) for sp in ('train', 'val')},
        'gate1': gate1,
    })
    write_json(OUT / 'window_shape_collinearity.json', {
        'measured_on': MEASURED_ON,
        'clip_geometry_attribution': CLIP_GEOMETRY_ATTRIBUTION,
        'clip_len': CLIP_LEN,
        'clip_scene_rule': 'core/dataset.py:747 先剔 scene==0 再取 max; '
                           '整条 clip 全 ignore ⇒ clip_scene=0（T110 现行口径, 不再回退 normal）'
                           '⇒ 该 clip 退出 scene 监督; 旧口径的回退面见 fallback 列',
        'fallback_key_note': 'all_ignore_clips_fallback_to_normal 统计的是"整条 clip 零标注"的面; '
                             'T110(2026-10-08) 后这批 clip 不再变成 normal, 而是退出监督 '
                             '—— 键名保留以兼容既有引用, 语义以此注为准',
        'domain_rule': 'clip 首帧名按 detection.py 前缀判据 (real / syn); family 列给出真实构成',
        'by_split_domain_scene': collinearity,
        'by_split_domain_family_scene': fam_collinearity,
        'all_ignore_clips_fallback_to_normal': fallback,
    })
    write_json(OUT / 'clip_label_supply.json', {
        'measured_on': MEASURED_ON,
        'clip_geometry_attribution': CLIP_GEOMETRY_ATTRIBUTION,
        'what': '旧口径下的成色账 —— 有多少 clip 的"normal"其实是零标注回退（T110 已废该回退）',
        'rule_under_test': '旧口径 core/dataset.py  `valid_scene = [...] or [1]` '
                           '⇒ 整条零标注 clip 记为 normal(1)。'
                           'T110(2026-10-08) 已改为 `clip_scene = max(valid_scene) if valid_scene else 0` '
                           '⇒ 这批 clip 退出 scene 监督，现行分母见 clip_supervision_denominators.json',
        'only_affects': 'Phase 3（ClipDataset/V24ClipDataset, clip 级 scene 标签）。'
                        'Phase 1/2 的 scene 评估是帧级判据, 不受影响。',
        'splits': faces['clip_label_supply'],
    })
    write_json(OUT / 'clip_supervision_denominators.json', {
        'measured_on': MEASURED_ON,
        'clip_geometry_attribution': CLIP_GEOMETRY_ATTRIBUTION,
        'what': 'T110 现行口径下 clip 级 scene 监督的真实分母（分域）—— 主表跨阶段对比必须用它',
        'rule': 'core/dataset.py:747  clip_scene = max(非 0 scene) 若存在, 否则 0;'
                ' focal(ignore_index=0) / cause(scene>=2) / type(scene==3) ⇒ scene==0 的 clip 不产生监督',
        'superseded_rule': '旧口径把同一批 clip 记为 normal(1) ⇒ 分母 59,299/15,597; '
                           '现行口径分母见各 split 的 clips_supervised',
        'domain_caveat': 'by_domain 用评估侧前缀判据 (detection.py:993, 仅 real_ 归 real), '
                         'realv/tum/tv 都落在 syn 桶 ⇒ 要读真实构成用 by_family',
        'splits': faces['clip_supervision'],
    })
    write_json(OUT / 'full_window_counterfactual.json', {
        'measured_on': MEASURED_ON,
        'clip_geometry_attribution': CLIP_GEOMETRY_ATTRIBUTION,
        'question': '把短窗组补成 30 帧满窗，能新增多少 clip 级监督，代价多少帧',
        'caveat': '新增帧本身没有 scene 标注 ⇒ 若沿用组内标签外推 = 把未标注帧当已标注, '
                  '与 dataset.py:748 的回退是同一类错误; 真要收益必须重标 (采集+标注成本)',
        'splits': faces['full_window_counterfactual'],
    })
    write_readme(recs, gate1, syn_family_mix, trend, collinearity, fallback, faces, gates, byclass)
    return recs, gate1, trend


def _verdict(v):
    """闸 1 的判定文字由实算生成 —— 不在 README 里写死"相等 ⇒ 同源" (判例 148)。"""
    if v['status'] != 'agree':
        return 'FATAL ' + v.get('reason', '')
    if 'attributed_reimport_stems' in v:
        return (f"同源（日志是 #90 前的老面；差额 {v['gap']} 帧逐帧归因到重导清单："
                f"命中 {v['attributed_reimport_stems']} = 差额，"
                f"清单 val 共 {v['reimport_manifest_val']} 帧，另有 "
                f"{v['reimport_not_in_this_bucket']} 帧不在本桶）")
    return '同源（逐字相等）'


def write_readme(recs, gate1, syn_mix, trend, collinearity, fallback, faces, gates, byclass):
    lines = [
        '# scene 头分域读数（论文主表的分域列来源，2026-10-08 起零 GPU 实测）',
        '',
        '仪器：`scripts/paper_domain_report.py`（`--build` / `--check` / `--report`）',
        '红线遵守：只读日志 + 只读现库，只写本目录；**未动训练、未动标签、未动 split、未动配方**。',
        '',
        '## 1. 口径与产生者同源（可复算的闸，不是引用）',
        '',
        '分域读数由训练侧 `scripts/core/detection.py` 打印（`:1144`），判据是：',
        '',
        '- `real` = **帧文件名以 `real_` 开头**（`:990-994`）',
        '- `tile` = 文件名匹配 `_t\\d{2}\\.<ext>` 的 2×2 瓦片，**从场景指标整体排除**（`:968-973`）',
        '- `valid` = `scene_cls != 0`（0 = ignore，`:964`）',
        '',
        '本仪器在盘上独立重算这三个判据的帧数，与日志打印的 `n=` 对账（判定由闸 1 实算，不写死）：',
        '',
        '| | 日志 n | 盘上重算 | 差额 | 判定 |',
        '|---|---|---|---|---|',
        f"| val real | {gate1['by_domain']['real'].get('log_n')} | "
        f"{gates['val']['real_valid_plain']} | {gate1['by_domain']['real'].get('gap')} | "
        f"{_verdict(gate1['by_domain']['real'])} |",
        f"| val syn  | {gate1['by_domain']['syn'].get('log_n')} | "
        f"{gates['val']['syn_valid_plain']} | {gate1['by_domain']['syn'].get('gap')} | "
        f"{_verdict(gate1['by_domain']['syn'])} |",
        '',
        '## 2. ⚠ 域标签的语义是错的（本仪器实测推翻，写论文前必读）',
        '',
        '日志把两个域写成 `real` / `syn`，读起来像"真实 CCTV vs 合成 CARLA"。**按族拆开帧名后**，',
        '`syn` 桶里的实际构成（val，帧级参与场景评估的 valid 非 tile 帧）：',
        '',
        '| 族 | 帧数 | 占 syn 桶 |',
        '|---|---|---|',
    ]
    tot = syn_mix['val']['total']
    for fam, pct in sorted(syn_mix['val']['pct'].items(), key=lambda x: -x[1]):
        if fam == 'total':
            continue
        lines.append(f'| {fam} | {syn_mix["val"][fam]:,} | {pct}% |')
    lines += [
        '',
        '**⇒ `syn` 桶的大头是 TUMTraf 真实行车记录仪（自车视角），不是合成画面。**',
        '真正的 CARLA 合成帧在 val 只占其中一小部分。',
        '所以：',
        '',
        '1. 论文里**禁止**把这两个读数写成 "real domain vs synthetic domain"；',
        '2. 能写的是 **"固定机位 CCTV（`real_` 前缀，n=1,641）" vs "其余源混合（含自车视角行车记录仪，n=17,463）"**；',
        '3. 两者帧数比 1:10.6 ⇒ 总 Acc 由后者主导，主表必须分域报，否则小域崩坏会被大域掩盖。',
        '',
        '## 3. 逐 epoch 读数（节选最近 5 行，全量见 `scene_by_domain_epochs.csv`）',
        '',
        '| phase | epoch | 总 Acc | real Acc | real R(n/p/c) | syn Acc | mAP50-95 |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in recs[-5:]:
        rr = (f"{r['real_R_normal']:.0f}/{r['real_R_pre_crash']:.0f}/{r['real_R_crash']:.0f}"
              if r.get('real_R_normal') is not None else '—')
        lines.append(f"| {r['phase']} | {r['epoch']} | {r.get('acc', '—')}% | "
                     f"{r['real_acc']}% | {rr} | {r['syn_acc']}% | {r.get('mAP50_95_B', '—')} |")
    lines += ['', '## 4. 趋势（首→末 epoch，全部来自日志）', '']
    for ph, t in trend.items():
        def pair(k):
            a, b = t[k]
            return '—' if a is None else f'{a}→{b}'
        lines += [
            f'- **{ph}** ep{t["epochs"][0]}→{t["epochs"][1]}：'
            f'real Acc {pair("real_acc_first_last")}%，'
            f'syn Acc {pair("syn_acc_first_last")}%，末轮差 {t["gap_acc_last"]} pp；'
            f'real 召回 n {pair("real_normal_R_first_last")} / '
            f'p {pair("real_pre_crash_R_first_last")} / '
            f'c {pair("real_crash_R_first_last")}',
        ]
    lines += [
        '',
        '## 5. clip 级时序形态 × 域（§7 口径核定 D3 的主动交代项）',
        '',
        'clip 级（Phase 3 的实际输入形态：30 帧窗 / 步长 10 + 末窗，`core/dataset.py`）实测：',
        '',
        '| split | domain | clip scene | clips | 均帧 | 满窗率 |',
        '|---|---|---|---|---|---|',
    ]
    for sp in ('train', 'val'):
        for r in collinearity[sp]:
            lines.append(f"| {sp} | {r['domain']} | {r['clip_scene']} | {r['clips']:,} | "
                         f"{r['mean_frames']} | {r['full_window_pct']}% |")
    lines += [
        '',
        '**读法**：满窗率 0% 的行意味着该类**从不以 30 帧连续序列出现过**，',
        '所谓"时序建模"在这些类上只是短窗（多为 3–5 帧）上的统计。任何"模型学会了时序因果"',
        '的说法都会被这条 confound 污染 ⇒ §7 主动交代，并把分域报数 + 形态表一起给出。',
        '',
        '> 口径变更说明：上一轮口头给的共线表用的是自定义域判据（把 `tum`/`tv`/`realv` 也算 real），',
        '> 与训练侧判据（`real_` 前缀）不一致，结论已被本表取代。',
        '',
        '## 6. clip 级 scene 监督的成色：`normal` 有 3/4 是零标注回退（结构性缺陷）',
        '',
        '**旧口径（已废）**：`core/dataset.py` 里 clip 帧标签先剔 `scene==0`，**全为 0 时回退 normal(1)** ——',
        '即"整条 clip 一帧都没标"会被当成"正常"。旧口径的分母账：',
        '',
        '| split | clips | 零标注 clip（回退 normal） | 占比 | normal 占比（回退前→后） |',
        '|---|---|---|---|---|',
    ]
    for sp in ('train', 'val'):
        s = faces['clip_label_supply'][sp]
        lines.append(
            f"| {sp} | {s['clips_total']:,} | {s['all_ignore_clips_become_normal']:,} | "
            f"{s['all_ignore_share_pct']}% | {s['normal_share_pct_before_fallback']}% → "
            f"**{s['normal_share_pct_after_fallback']}%** |")
    lines += [
        '',
        '**影响面**：只影响 clip 级 scene 标签，即 **Phase 3（ClipDataset / V24ClipDataset）**',
        '的训练与 CDCR/UASC 那条路；Phase 1/2 的 scene 评估是帧级判据（§1），不受影响。',
        '',
        '**处置（2026-10-08 已执行，T110）**：回退已改成 ignore —— '
        '`core/dataset.py` 整条 clip 零标注 ⇒ `clip_scene=0`、`cause_valid=0`，',
        '三路损失（focal `ignore_index=0` / cause `scene>=2` / type `scene==3`）本就跳过 0，',
        '**未改配方、未改标签、未动 split**。分级依据见 `meta/clipnorm/`：',
        '零标注 clip 的 93.9%(train)/96.0%(val) 来自"该视频一帧都没标"的面，其中',
        'synthetic 侧 27,679 条全是 1–4 帧碎片且 video_id 就是 rear-end/head-on/t-bone 事故剧情',
        '⇒ 按 coverage 规则也只有 138/29,597 条能被正名为真 normal。',
        'clip 级 normal：train 45,211→15,752，val 13,283→2,278；crash/normal 0.215→0.617(train)。',
        '验证：`scripts/test_clip_ignore_fallback.py` 三层闸全绿（真库面抽样 ignore 71.3% vs 账本 71.2%）。',
        '',
        '### 6.1 现行口径下 clip 级 scene 监督的真实分母（主表跨阶段对比用它）',
        '',
        '零标注 clip 改成 ignore 之后，clip 级 scene 头的分母从"全部 clip"缩成"有标注 clip"，',
        '**Phase 1/2（帧级）与 Phase 3（clip 级）的读数不再同分母，报任何跨阶段差值必须并列这两个数**：',
        '',
        '| split | domain | clip 总数 | 进监督 | 保留率 | 退出监督 | normal | pre_crash | crash | crash/进监督 |',
        '|---|---|---|---|---|---|---|---|---|---|',
        *[f"| {r['split']} | {r['domain']} | {r['clips']:,} | {r['clips_supervised']:,} | "
          f"{r['supervised_pct']}% | {r['clips_exit_supervision']:,} | {r['normal']:,} | "
          f"{r['pre_crash']:,} | {r['crash']:,} | {r['crash_share_of_supervised_pct']}% |"
          for s in faces['clip_supervision'].values() for r in s['by_domain']],
        '',
        '> 注意 `domain` 列只有两档（`real_` 前缀 vs 其余），**realv / tum / tv 全被塞进 `syn` 桶**',
        '> —— 这是 §7 那两条矛盾域判据的评估侧口径，别把这里的 `syn` 读成"纯合成"。按族拆同一份账：',
        '',
        '| split | family | 域前缀 | clip 总数 | 进监督 | 保留率 | normal | pre_crash | crash |',
        '|---|---|---|---|---|---|---|---|---|',
        *[f"| {r['split']} | {r['family']} | {r['domain_by_prefix']} | {r['clips']:,} | "
          f"{r['clips_supervised']:,} | {r['supervised_pct']}% | {r['normal']:,} | "
          f"{r['pre_crash']:,} | {r['crash']:,} |"
          for s in faces['clip_supervision'].values() for r in s['by_family']],
        '',
        f"**合计**：train {faces['clip_supervision']['train']['clips_total']:,} → "
        f"{faces['clip_supervision']['train']['clips_supervised']:,} 条"
        f"（{faces['clip_supervision']['train']['supervised_pct']}%），",
        f"val {faces['clip_supervision']['val']['clips_total']:,} → "
        f"{faces['clip_supervision']['val']['clips_supervised']:,} 条"
        f"（{faces['clip_supervision']['val']['supervised_pct']}%）；",
        f"val 侧 crash 监督只剩 {faces['clip_supervision']['val']['crash_supervised']:,} 条，"
        f"real 域仅 {faces['clip_supervision']['val']['by_domain'][0]['crash']} 条",
        '⇒ **val 的 clip 级 recall 抖动下限很低，Phase 3 主表必须报 CI，不能只报点估计**（见 §统计面）。',
        '',
        '## 7. 两条互相矛盾的"域"判据并存（论文必须先固定一条）',
        '',
        '| 处 | 判据 | 后果 |',
        '|---|---|---|',
        f"| 评估面 `{faces['domain_predicate_conflict']['A_eval_side']['file']}"
        f"{faces['domain_predicate_conflict']['A_eval_side']['refs']}` | "
        f"{faces['domain_predicate_conflict']['A_eval_side']['rule']} | "
        f"{faces['domain_predicate_conflict']['A_eval_side']['effect']} |",
        f"| Phase 3 训练面 `{faces['domain_predicate_conflict']['B_phase3_training_side']['file']}` | "
        f"{faces['domain_predicate_conflict']['B_phase3_training_side']['rule']} | "
        f"{faces['domain_predicate_conflict']['B_phase3_training_side']['effect']} |",
        '',
        f"**判定**：{faces['domain_predicate_conflict']['verdict']}",
        '',
        '## 8. "把短窗组补成满窗"要多少帧、值多少 clip（零 GPU 盘算，不建议投）',
        '',
        '| split | family | clip scene | 短窗组 | 现有帧 | 补满窗需新增帧 | 新增 clip |',
        '|---|---|---|---|---|---|---|',
    ]
    for sp in ('train', 'val'):
        for r in faces['full_window_counterfactual'][sp]:
            if r['clip_scene'] not in ('normal', 'pre_crash'):
                continue
            lines.append(f"| {sp} | {r['family']} | {r['clip_scene']} | {r['short_groups']:,} | "
                         f"{r['frames_present']:,} | {r['frames_needed_to_full_window']:,} | "
                         f"+{r['full_window_clips_gained']:,} |")
    carla_norm = next((r for r in faces['full_window_counterfactual']['train']
                       if r['family'] == 'carla_synthetic' and r['clip_scene'] == 'normal'), None)
    lines += [
        '',
        f"**结论（算术，不是偏好）**：CARLA normal 侧最多值 "
        f"+{carla_norm['full_window_clips_gained']:,} 条 train clip，",
        f"代价是补 {carla_norm['frames_needed_to_full_window']:,} 帧"
        '—— 而这些新帧**本身没有 scene 标注**。若沿用组内标签外推到 30 帧，',
        '等于把未标注帧当已标注，与 §6 的回退是同一类错误，只是放大。',
        '⇒ 真 normal 监督的瓶颈是**标注**（帧级），不是窗口形态；投采集补窗换不来真监督。',
        '帧级真标注面（scene≠0）按族：',
        '',
        '| split | family | tile? | normal | pre_crash | crash |',
        '|---|---|---|---|---|---|',
    ]
    for sp in ('train', 'val'):
        agg = {}
        for (fam, tt, cls), v in byclass[sp].items():
            agg.setdefault((fam, tt), {})[cls] = v
        for (fam, tt) in sorted(agg):
            d = agg[(fam, tt)]
            lines.append(f"| {sp} | {fam} | {tt} | {d.get('normal', 0):,} | "
                         f"{d.get('pre_crash', 0):,} | {d.get('crash', 0):,} |")
    lines += [
        '',
        '## 9. 没测的（别在这里找）',
        '',
        '- 分域 **混淆矩阵**：训练侧只落总矩阵（`runs/*/scene_confusion.csv`），分域仅打印 Acc 与召回。',
        '- 分域 mAP / 分域 cause-type 指标：验证循环里没有域轴。',
        '- 小域（`real_` 前缀，n=1,641）虚警的归因：视觉域差 vs 时序形态 shortcut，',
        '  需要 GPU 上的四臂对照，另案（Phase 3c 后）。',
        '',
        '## 10. 漂移',
        '',
        '训练在跑，日志在长 ⇒ `--check` 把"已落盘 epoch 的读数必须逐字不变"作为闸，',
        '新增 epoch 不算漂移。分母同源闸（§1）每次 `--check` 都重算，日志与盘上任一侧改动都会被抓。',
        '',
    ]
    write_text(OUT / 'README.md', '\n'.join(lines))


def build_snapshot():
    recs, gates, trend = build()
    return recs, gates, trend


def check():
    """重算并与落盘 CSV 比对：已落盘 epoch 的读数不许变（漂移闸）。"""
    global _WRITE
    _WRITE = False
    persisted = []
    cp = OUT / 'scene_by_domain_epochs.csv'
    if not cp.exists():
        sys.exit('[FATAL] 尚未落盘，先跑 --build')
    with open(cp, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            persisted.append(r)
    recs, gate1, _ = build_snapshot()
    key = lambda r: (str(r['phase']), str(r['epoch']))
    pmap = {key(r): r for r in persisted}
    nmap = {key(r): r for r in recs}
    drift = []
    for k, pr in pmap.items():
        nr = nmap.get(k)
        if nr is None:
            drift.append(f'丢失历史读数 {k}')
            continue
        for col in pr:
            if col in ('log', 'epoch_check'):
                continue
            if str(pr[col]) != str(nr.get(col, '')):
                drift.append(f'{k} {col}: 落盘={pr[col]} 现算={nr.get(col)}')
    new = [k for k in nmap if k not in pmap]
    if drift:
        print('[FAIL] 漂移：')
        for d in drift[:20]:
            print('  -', d)
        return 1
    print(f'[PASS] 已落盘 {len(pmap)} 个 epoch 读数逐字复现；'
          f'分母同源闸 agree={gate1["agree"]}')
    if new:
        print(f'  (新增 {len(new)} 个 epoch 校验点：{new[-1]} —— 训练在跑，属预期增长)')
    return 0


def report():
    with open(OUT / 'scene_by_domain_epochs.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f'分域读数：{len(rows)} 个 epoch 校验点（phase1+phase2）')
    for r in rows[-6:]:
        print(f"  {r['phase']:>6} ep{r['epoch']:>3}  总 Acc={r['acc']:>5}%  "
              f"real={r['real_acc']:>5}% (n={r['real_n']}) "
              f"R n/p/c={r['real_R_normal']}/{r['real_R_pre_crash']}/{r['real_R_crash']}  "
              f"syn={r['syn_acc']:>5}% (n={r['syn_n']})  mAP50-95={r['mAP50_95_B']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if not (a.build or a.check or a.report):
        ap.print_help()
        return 2
    if a.build:
        build_snapshot()
        print(f'[OK] 已落盘 {OUT}')
    if a.check:
        return check()
    if a.report:
        report()
    return 0


if __name__ == '__main__':
    sys.exit(main())
