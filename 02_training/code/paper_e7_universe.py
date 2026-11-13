#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E7 共同考卷尺 (G-314, #273)。

E7 = V25↔V26 在**同一评估脚本、同一份考卷**上的对照 (§1 表格: 跨脚本对比禁止)。
「同一份考卷」不是当然成立的: V26 的 val 宇宙比 V25 大, 而多出来的帧里有一部分
落在 **V25 的训练集** ⇒ 拿 V26 全量 val 评 V25 臂 = 让 V25 答它见过的题, 虚高。
本尺把污染量测出来, 只放出「两臂都真正 held-out」的冻结名单, 并做不变量硬闸。

放出的考卷 (逐帧文件名, 排序冻结):
  common_raw = V26val ∩ V25val              两臂都 held-out (仅帧级)
  common     = common_raw − 段级脏帧        ← E7 唯一可用考卷 (帧级 + 段级双 held-out)
  (对照量, 不入考卷) V26val ∩ V25train      = 拿全量 val 考 V25 臂的记忆污染量
「段级」= 该帧所属 clip 分组键出现在任一臂 train ⇒ 整段都不进考卷; 键函数用管线唯一真相源。

用法:
  python scripts/paper_e7_universe.py --selftest          # 造合成目录树验不变量有牙
  python scripts/paper_e7_universe.py --emit-face         # 出宇宙名单面 (CPU, 秒级)
  python scripts/paper_e7_universe.py --arch-census       # 键集对账 (CPU, 需 torch, 分钟级)
产物一律落 datasets/accident_v26/meta/paper/E7_common_split/ (§7 纪律)。
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
PKG_ROOT = os.environ.get('V26_PKG_ROOT', '.')

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(HERE, '..'))
for _p in (HERE, os.path.join(HERE, 'core')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# 视频级分组键的唯一真相源 (dataset.py:429 同源)。绝不回退到自写 rsplit: 那是第二把尺,
# 且实测更粗 (把同视频的 00/01 两段并成一个键 ⇒ clip 级污染被高估)。兜不到就出声死。
try:
    from base_config import extract_video_id as _extract_video_id
except ModuleNotFoundError:
    try:
        from core.base_config import extract_video_id as _extract_video_id
    except ModuleNotFoundError as e:
        raise SystemExit(f"[红] 取不到管线真相源 base_config.extract_video_id: {e}")
# 成因摘除位同样只认那一处真相源 (`core/base_config.py` 的集合)。本件早先是自己抄一份
# `[4, 7]` 写进宇宙面文案 ⇒ 真相源一改, 面还在说旧话, 且没有任何尺会红。
try:
    from base_config import CAUSE_DISABLED_BITS as _CAUSE_DISABLED
except ModuleNotFoundError:
    try:
        from core.base_config import CAUSE_DISABLED_BITS as _CAUSE_DISABLED
    except ModuleNotFoundError as e:
        raise SystemExit(f"[红] 取不到管线真相源 base_config.CAUSE_DISABLED_BITS: {e}")
DATASET = 'datasets/accident_v26'
FACE_DIR_REL = DATASET + '/meta/paper/E7_common_split'


def _resolve(p):
    """一律折成绝对路径: 训练侧 core.base_config 在 import 时会 os.chdir, 不能靠相对路径。"""
    return p if os.path.isabs(p) else os.path.join(PKG_ROOT, p)


PAPER_FACE_ROOT = _resolve(DATASET + '/meta/paper')
INSTRUMENT = 'paper_e7_universe.py'

# V25 侧的树在包外 ⇒ 绝对路径写死, 换版本时必须现读目录是否存在
V25_ROOT = os.environ.get('V25_DATA_ROOT', 'datasets/accident_v25')
DIRS = {
    'v26_val': _resolve(f'{DATASET}/images/val'),
    'v26_train': _resolve(f'{DATASET}/images/train'),
    'v25_val': f'{V25_ROOT}/images/val',
    'v25_train': f'{V25_ROOT}/images/train',
}
MODEL_YAML_DEFAULT = 'config/yolo26m_detect_cls_fixed.yaml'   # 仪器 argparse 默认 (:232)
# lever yaml 的**文件名**必须能被 ultralytics `guess_model_scale` 认出 m:
# 规则 = stem 里匹配 `yolo(e-)?[v]?\d+([nslmx])`。今天实测: 文件名不带尺度码 ⇒ 退成 n ⇒
# 深度不同 ⇒ 键数 886→861、V25 权重"丢弃"从 35 虚涨到 60, 而自检 A (只评原配臂) 全绿看不见。
LEVER_NAME = 'yolo26m_e7_v25arm_has_contrastive.yaml'
_SCALE_RE = r'yolo(e-)?[v]?\d+([nslmx])'
CLIP_LEN = 16                                                 # run_train.py 与验收同口径
CAUSE_DISABLED_BITS = sorted(_CAUSE_DISABLED)   # 现读真相源, 不再抄字面量


def _scale_code(path_or_name):
    r"""复刻 ultralytics `guess_model_scale` 的规则 (在 stem 里搜 `yolo(e-)?[v]?\d+([nslmx])`)。
    lever yaml 文件名解析不出尺度码 ⇒ 构建会静默退成 'n', 键数/宽度全变, 而读数看着正常。"""
    m = re.search(_SCALE_RE, os.path.splitext(os.path.basename(str(path_or_name)))[0])
    return m.group(2) if m else ''


def list_frames(d):
    if not os.path.isdir(d):
        raise SystemExit(f"[红] 目录不存在, 无法出宇宙: {d}")
    return {e.name for e in os.scandir(d) if e.name.lower().endswith(('.jpg', '.jpeg', '.png'))}


def video_key(name):
    """real_0jlyeLcFLVc_00_000012.jpg → real_0jlyeLcFLVc_00 (仪器同款: 去扩展名后走
    `base_config.extract_video_id`, 含摄像头/段标识 ⇒ 与 ClipDataset 分组同粒度)。"""
    return _extract_video_id(os.path.splitext(os.path.basename(name))[0])


def build_universes(d=None):
    dirs = d or DIRS
    v26val, v26tr = list_frames(dirs['v26_val']), list_frames(dirs['v26_train'])
    v25val, v25tr = list_frames(dirs['v25_val']), list_frames(dirs['v25_train'])
    common_raw = v26val & v25val
    # 段级剔除: 该帧所属 clip 分组键只要出现在任一臂的 train, 整段都不进共同考卷
    tr_vids = {video_key(n) for n in v26tr} | {video_key(n) for n in v25tr}
    seg_excluded = {n for n in common_raw if video_key(n) in tr_vids}
    common = common_raw - seg_excluded
    # 只在 V26 val、却在 V25 train 的帧 = "全量 val 考 V25 臂" 的记忆污染面
    full_val_leak_into_v25_train = v26val & v25tr
    v25_only = v25val - v26val
    return {
        'common_raw': common_raw, 'common': common, 'seg_excluded': seg_excluded,
        'all_train': v26tr | v25tr, 'tr_vids': tr_vids,
        'full_val_leak_into_v25_train': full_val_leak_into_v25_train,
        'v25_only_frames': v25_only, 'v25_only_in_v26_train': v25_only & v26tr,
        'counts': {'v26_val': len(v26val), 'v26_train': len(v26tr),
                   'v25_val': len(v25val), 'v25_train': len(v25tr)},
    }


def invariants(u):
    """硬闸: 任一条不过 ⇒ 本尺不出面 (不是警告, 是拒绝)。返回 [(名, 过?, 说明)]。

    后三条是**后置条件**: 剔除逻辑被人改掉 (不再剔) 时它们必须变红; 自测用
    「跳过剔除的变异 u」证明它们有牙, 不是永真的安慰剂。
    """
    c, raw = u['common'], u['common_raw']
    fr_leak = c & u['all_train']
    seg_leak = {n for n in c if video_key(n) in u['tr_vids']}
    ratio = len(c) / max(len(raw), 1)
    return [
        ('common 非空', len(c) > 0, f"common = {len(c)} 帧"),
        ('帧级零泄漏 (common 不含任何 train 帧)', len(fr_leak) == 0,
         f"命中 {len(fr_leak)} 帧 (必须 0)"),
        ('段级零重叠 (common 每段都不在任一臂 train)', len(seg_leak) == 0,
         f"命中 {len(seg_leak)} 帧 (必须 0)"),
        ('共同宇宙非平凡 (剔完 ≥ raw 的一半)', ratio >= 0.5,
         f"common/raw = {len(c)}/{len(raw)} = {ratio:.1%} (折到一半以下＝键函数坏了)"),
    ]


def sha16(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _instrument_sha():
    """把**本仪器自己**的 sha 钉进面: 论文读者拿面就能问"哪版代码产的", 不必回台账翻。"""
    try:
        return sha16(open(os.path.abspath(__file__), encoding='utf-8').read())
    except Exception as e:
        return f'unreadable_{type(e).__name__}'


def filter_clips_by_frame_list(clips, labels, want):
    """E7 考卷过滤器 (纯函数, 自测锚在 --selftest 里;评估侧 `eval_acceptance.py --frame-list` 用它)。

    clips   = ClipDataset.clips, 每项 (帧索引列表, video_id, …) —— 与 tile-clip 排除同款形态
    labels  = clip_dataset.base.labels (取 'im_file' 的 basename 当帧名)
    want    = 共同考卷名单 (common_frames.txt 读出来的 basename 集合)
    律: **每一帧都在名单内**的 clip 才留, 跨界 clip 整条剔 (半条混 ⇒ 两臂看到的窗口不一样,
        对照就不干净了)。返回 (kept, n_partial, n_missing_in_tree);
        n_missing_in_tree 必须为 0 —— 不为 0 说明这臂的 val 树里没有名单中的某些帧,
        即两臂考卷并非同一张, E7 当场不可结线 (§1「同 split」被违反)。
    """
    name = lambda fi: os.path.basename(str(labels[fi]['im_file']))
    kept, n_partial = [], 0
    for c in clips:
        if all(name(fi) in want for fi in c[0]):
            kept.append(c)
        else:
            n_partial += 1
    have = {name(fi) for fi in range(len(labels))}
    return kept, n_partial, len(set(want) - have)


def _current_round():
    """轮号一律现读 (多会话共享命名空间), 不许把 G-### 写死进默认值当缓存。"""
    p = _resolve(DATASET + '/meta/autogov/current_round.json')
    try:
        with open(p) as f:
            return json.load(f).get('round', 'ADHOC')
    except Exception as e:
        return f'unreadable_{type(e).__name__}'


def write_if_changed(path, content, cur_sha):
    """幂等落盘: 同字节不重写 (mtime 会顶到当下, 但 sha 不变 ⇒ 凭据面仍对得上)。"""
    if os.path.exists(path) and sha16(open(path, 'rb').read().decode()) == cur_sha:
        return 'unchanged'
    with open(path, 'w') as f:
        f.write(content)
    return 'written'


def emit_face(args):
    t0 = time.time()
    u = build_universes()
    checks = invariants(u)
    ok = all(c[1] for c in checks)
    out_dir = args.out if os.path.isabs(args.out) else _resolve(args.out)
    durable = (out_dir == PAPER_FACE_ROOT or out_dir.startswith(PAPER_FACE_ROOT + os.sep))
    print('[不变量闸] ' + ('全过' if ok else '有不过 ⇒ 拒绝出面'))
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'} {name}: {detail}")
    if not ok:
        print('[红] 不变量不过 ⇒ 不落面 (rc=2)'); sys.exit(2)
    if not durable:
        print(f'[红] --out 不在 {PAPER_FACE_ROOT} 之下 ⇒ 只允许草稿路径, 本尺不出正式面 (rc=2)')
        sys.exit(2)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        'face': 'E7_common_split_universe', 'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'round': args.round, 'task': '#273', 'instrument': INSTRUMENT,
        'instrument_sha256_16': _instrument_sha(),
        'purpose': ('E7 两臂的共同考卷: 只在 common 宇宙上比分, 全量 V26 val 上比 = V25 臂作弊'),
        'source_dirs': {k: v for k, v in DIRS.items()},
        'counts': u['counts'],
        'universes': {
            'common_raw': {'n': len(u['common_raw']),
                           'note': 'V26val ∩ V25val, 未做段级剔除'},
            'common_frames': {'n': len(u['common']), 'file': 'common_frames.txt',
                              'sha256_16': sha16('\n'.join(sorted(u['common'])) + '\n'),
                              'families': dict(Counter(video_key(n).split('_')[0]
                                                      for n in u['common']).most_common(8)),
                              'definition': '两臂 val 的交集 ∩ 段级干净 (帧级+段级双 held-out)'},
        },
        'held_out_evidence': {
            'segment_level_excluded_frames': len(u['seg_excluded']),
            'segment_level_excluded_groups': len({video_key(n) for n in u['seg_excluded']}),
            'common_over_raw_pct': round(100.0 * len(u['common']) / max(len(u['common_raw']), 1), 2),
            'caliber': '分组键 = 管线唯一真相源 extract_video_id (ClipDataset/eval 建 clip 用的同一个键)',
        },
        'contamination_measured': {
            'full_v26_val_frames_also_in_v25_train': len(u['full_val_leak_into_v25_train']),
            'pct_of_v26_val': round(100.0 * len(u['full_val_leak_into_v25_train'])
                                    / u['counts']['v26_val'], 2),
            'v25_val_frames_not_in_v26_val': len(u['v25_only_frames']),
            'of_which_in_v26_train': len(u['v25_only_in_v26_train']),
            'coarse_source_video_caliber_note': (
                '更粗的"同一源视频"口径**不在本面发布**: 各族键尾不同 (实测尾值 top: _50mm / _16mm / '
                '_00 / _000136), 一条通用规则折不出源视频 ⇒ 要发这一口径必须先按族定义规则再重测; '
                '自写 rsplit 会把同视频多段并起来, 虚报成 clip 级泄漏 (本轮实测: 粗键自写规则给出 '
                '55 个碰撞组/4,592 帧, 换成真相源键后为 0)'),
        },
        'cause_axis_clause': (
            f'成因轴只在 7 个在册位上比: V26 结构性摘除 bit{CAUSE_DISABLED_BITS} '
            f'(头里恒置 -30, `detect_classify_head.py:220-222`), V25 这两位是活的 ⇒ '
            f'9 位口径直接比会把 V26 判死在两位上, 分母差必须写进正文'),
        'gate': {'invariants_all_pass': ok, 'out_is_durable_paper_face': durable,
                 'rule': '不变量全过 ⇒ 才允许出面; 任何一帧泄漏进任一臂 train 都拒绝出共同考卷'},
        'face_status': 'verified' if ok else 'draft_unverified',
        'elapsed_s': round(time.time() - t0, 2),
    }
    body = '\n'.join(sorted(u['common'])) + '\n'
    fn = 'common_frames.txt'
    st = write_if_changed(os.path.join(out_dir, fn), body, sha16(body))
    manifest['universes']['common_frames']['write'] = st
    print(f"  {fn}: {len(u['common'])} 行 [{st}]")
    mpath = os.path.join(out_dir, 'universe_manifest.json')
    with open(mpath, 'w') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[面] {mpath}")
    print(f"[结论] common_raw={len(u['common_raw'])} → 段级剔 {len(u['seg_excluded'])} ⇒ "
          f"common={len(u['common'])}; 全量 val 考 V25 臂的污染={len(u['full_val_leak_into_v25_train'])}帧")
    return manifest


# ---------------------------------------------------------------- 架构对等性
def arch_census(args):
    """V25 权重能不能装进 V26 仪器? 缺口是哪个模块、由哪个键门控、评估读不读它。

    两个自己踩过的坑写死在这里, 防止下一次拿假数结案:
      ① model yaml 必须现读仪器 argparse 默认 (`eval_acceptance.py:232`), 凭文件名像就选
         `yolo26m_detect_cls.yaml` ⇒ DetectClassify 落在索引 21 而权重在 23 ⇒ 349/886 的假读数;
      ② yaml 必须**按路径**传给 DetectClassifyModelV18: 读成 dict 直接构造会让 ultralytics
         的 scale 推断退化成 n ⇒ 整棵骨干宽度对不上;
      ③ 生成的 lever yaml **文件名**也得带尺度码 (guess_model_scale 只认 stem 里的 yolo<d><码>):
         今天实测裸名 ⇒ 对比臂静默退成 n ⇒ 键数 886→861、"丢弃" 35→60 的假账, 而自检 A 全绿
         看不见 (A 只评原配臂) ⇒ 加自检 B (对比臂须为原配臂严格超集＋共享键形状全等) 与前置拒跑。
    """
    import torch
    import yaml as _yaml
    for p in (os.path.join(HERE), os.path.join(HERE, 'core')):
        if p not in sys.path:
            sys.path.insert(0, p)
    from detection import DetectClassifyModelV18
    from temporal_model import build_temporal_model
    import train_config

    dev = 'cpu'   # 本 Census 只构建+对键, 不吃 GPU, 不与在训任务抢卡

    def build(yaml_path):
        det = DetectClassifyModelV18(cfg=yaml_path, verbose=False)
        return build_temporal_model(
            detection_model=det, clip_len=CLIP_LEN,
            ctta_config=train_config.CTTA_CONFIG, uasc_config=train_config.UASC_CONFIG,
            cdcr_config=train_config.CDCR_CONFIG, ptcl_config=train_config.PTCL_CONFIG)

    def keys(ckpt):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        ms = ck.get('model_state_dict', ck.get('model', ck))
        meta = {k: v for k, v in ck.items()
                if k not in ('model_state_dict', 'optimizer_state_dict', 'model', 'optimizer')}
        return ms, {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))}

    def diff(own_sd, ms):
        """own_sd 必须是**张量**字典 (state_dict), 不是形状字典 —— 形状比较要真 shape。"""
        common = [k for k in ms if k in own_sd]
        return {
            'ckpt_keys': len(ms), 'instrument_keys': len(own_sd),
            'hit': len(common), 'hit_pct': round(100.0 * len(common) / max(len(own_sd), 1), 2),
            'shape_mismatch': [k for k in common if tuple(own_sd[k].shape) != tuple(ms[k].shape)],
            'random_in_instrument': sorted(set(own_sd) - set(ms)),
            'dropped_from_ckpt': sorted(set(ms) - set(own_sd)),
        }

    def fam(ks):
        return dict(sorted(Counter('.'.join(k.split('.')[:3]) for k in ks).items(),
                           key=lambda x: -x[1]))

    v26_ckpt = args.v26_ckpt if os.path.isabs(args.v26_ckpt) else _resolve(args.v26_ckpt)
    v25_ckpt = args.v25_ckpt
    ms26, meta26 = keys(v26_ckpt)
    ms25, meta25 = keys(v25_ckpt)

    tm = build(_resolve(MODEL_YAML_DEFAULT))
    sd = tm.state_dict()
    own = {k: tuple(v.shape) for k, v in sd.items()}
    selftest_a = {'hit': sum(1 for k in ms26 if k in own), 'instrument_keys': len(own),
                  'shape_mismatch': len([k for k in ms26
                                         if k in own and own[k] != tuple(ms26[k].shape)])}
    trust_a = (selftest_a['hit'] == selftest_a['instrument_keys']
               and selftest_a['shape_mismatch'] == 0)
    arm_default = diff(sd, ms25)
    del tm, sd

    # 纯配置臂: 只往 model yaml 顶层加 has_contrastive: true (门控在 detection.py:88 pop 它)
    # 生成的 lever yaml 落**正式面目录**而不是 /tmp (G-312 教训: /tmp 输入重启即失, 出处不可久留),
    # 这样 E7 真跑时用的就是这份钉了 sha 的配置, 不是"当时临时造的某一份"。
    out_dir = args.out if os.path.isabs(args.out) else _resolve(args.out)
    durable_out = _is_durable(out_dir)
    if durable_out:
        os.makedirs(out_dir, exist_ok=True)
        lever_dir = out_dir
    else:
        lever_dir = tempfile.mkdtemp(prefix='e7lever_')
    lever_name = LEVER_NAME
    tmp_yaml = os.path.join(lever_dir, lever_name)
    if _scale_code(tmp_yaml) != _scale_code(MODEL_YAML_DEFAULT):
        raise SystemExit(f"[红] lever yaml 文件名尺度码 {_scale_code(tmp_yaml)!r} ≠ 原配 "
                         f"{_scale_code(MODEL_YAML_DEFAULT)!r} ⇒ 构建会退成另一档宽度, 拒跑 "
                         f"(LEVER_NAME 里必须带 yolo<d><码>)")
    d = _yaml.safe_load(open(_resolve(MODEL_YAML_DEFAULT)))
    d.pop('has_contrastive', None)
    d['has_contrastive'] = True
    with open(tmp_yaml, 'w') as f:
        _yaml.safe_dump(d, f, sort_keys=False)
    lever_sha = sha16(open(tmp_yaml).read())
    tm2 = build(tmp_yaml)
    sd2 = tm2.state_dict()
    own2 = {k: tuple(v.shape) for k, v in sd2.items()}
    # 自检 B: has_contrastive 只准**加**键, 不许改宽度/深度 —— 原配臂每只键须在对比臂里同形状出现。
    # 只有自检 A 拦不住尺度码退化 (A 只评原配臂), 今天就是这么读出一张 861 键的假账的。
    shared = [k for k in own if k in own2]
    check_b = {'default_keys': len(own), 'contrastive_keys': len(own2), 'shared': len(shared),
               'shared_shape_drift': [k for k in shared if own[k] != own2[k]],
               'lost_from_contrastive': sorted(set(own) - set(own2)),
               'n_added': len(set(own2) - set(own)),
               'note': '对比臂键集必须是原配臂的严格超集且共享键形状全等; 否则 lever yaml 的 '
                       '尺度码/结构被文件名或改写弄坏, 本面对比臂数字全部作废'}
    check_b['pass'] = bool(not check_b['shared_shape_drift'] and not check_b['lost_from_contrastive']
                           and len(own2) >= len(own))
    arm_contrastive = diff(sd2, ms25)
    v26_in_contrastive = diff(sd2, ms26)
    del tm2, sd2

    # 静态侧: 缺口模块的输出评估读不读? (只 grep 定义与消费, 不执行)
    out = {
        'face': 'E7_arch_equivalence', 'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'round': args.round, 'task': '#273', 'instrument': INSTRUMENT,
        'instrument_sha256_16': _instrument_sha(), 'device': dev,
        'clip_len': CLIP_LEN, 'model_yaml_instrument_default': MODEL_YAML_DEFAULT,
        'harness_self_check_A': {
            **selftest_a, 'pass': trust_a,
            'note': '仪器原配必须复现在册 886/886; 不过则下面所有对账作废 (yaml 选错/scale 退化都靠它拦)'},
        'harness_self_check_B': check_b,
        'weights': {'v25': {'path': v25_ckpt, **meta25}, 'v26': {'path': v26_ckpt, **meta26}},
        'arm_default_yaml': {
            'vs_v25': {**arm_default, 'random_in_instrument_families': fam(arm_default['random_in_instrument']),
                       'dropped_from_ckpt_families': fam(arm_default['dropped_from_ckpt'])},
            'note': '这就是计划 §8.4-4 说的「shape 硬闸」场景 —— 实测 shape_mismatch=0, 拦路的不是形状闸'},
        'arm_has_contrastive_true': {
            'vs_v25': {**arm_contrastive,
                       'random_in_instrument_families': fam(arm_contrastive['random_in_instrument'])},
            'vs_v26': {**v26_in_contrastive,
                       'dropped_from_ckpt_families': fam(v26_in_contrastive['dropped_from_ckpt'])},
            'lever': ('纯配置: model yaml 顶层加 has_contrastive: true; 读者 = '
                      '`detection.py:88` pop → :149 建 text_encoder/text_proj_head, '
                      '→ detect_classify_head.py:197 建 vis_proj_head。零改码。'),
            'lever_yaml': {'file': lever_name, 'path': tmp_yaml, 'sha256_16': lever_sha,
                           'durable_in_face_dir': durable_out,
                           'note': 'E7 真跑 V25 臂时 --model-yaml 就指这份; 落盘后 sha 即出处 '
                                   '(G-312 教训: /tmp 临时造的配置重启即失)'}},
        'lane_head': {
            'gate': '无配置开关: `detect_classify_head.py:182` 无条件构造 ⇒ V25 权重永远给不出这 10 只张量',
            'consumers': '只有损失侧读它 (`losses.py:792-801` 取 preds[7]); 评估读数不碰索引 7/8/9',
            'must_prove': '随机 lane_head 不改三目标读数 ⇒ 两次不同初始化对撞, 三目标读数须逐位全等'},
        'exam_universe': 'E7 两臂同考卷 = E7_common_split/common_frames.txt (本尺 --emit-face 出)',
        'gate': {'harness_self_check_A_pass': trust_a, 'harness_self_check_B_pass': check_b['pass'],
                 'lever_yaml_durable': durable_out,
                 'rule': 'A (原配复现 886/886) 与 B (对比臂是原配臂严格超集且共享键形状全等) 任一不过 '
                         '⇒ 本面键集数字作废, 先修探针'},
        'face_status': 'verified' if (trust_a and check_b['pass']) else 'draft_unverified',
        'verdict': None,
    }
    if not trust_a:
        out['verdict'] = 'UNTRUSTED_HARNESS: 自检 A 没过, 本面所有键集数字作废, 先修探针再谈 E7'
    elif not check_b['pass']:
        out['verdict'] = (f"UNTRUSTED_HARNESS: 自检 B 没过 (原配 {check_b['default_keys']} 键 vs "
                          f"对比臂 {check_b['contrastive_keys']} 键, 丢 {len(check_b['lost_from_contrastive'])}, "
                          f"形状漂 {len(check_b['shared_shape_drift'])}) ⇒ lever yaml 把架构改坏了 "
                          f"(头号嫌疑 = 文件名尺度码/结构改写), 对比臂数字作废")
    elif (arm_contrastive['hit'] == len(ms25) and not arm_contrastive['shape_mismatch']):
        out['verdict'] = (
            f"E7 可做, 但不是「解形状闸」: 形状冲突实测 0; 真正要做的是给 V25 臂一套 "
            f"has_contrastive: true 的 model yaml ⇒ V25 全部 {len(ms25)} 只张量 100% 装载, "
            f"唯一缺口是 V26 的车道线分支 {len(arm_contrastive['random_in_instrument'])} 只随机张量 "
            f"(评估不消费, 须以双初始化对撞自证)。两臂各按自己的形状装配 + 同一仪器/同一共同考卷。")
    else:
        out['verdict'] = ('E7 需判停或退化装配: has_contrastive 臂仍未收口 V25 全部张量 '
                          f"(hit={arm_contrastive['hit']}/{arm_contrastive['instrument_keys']}, "
                          f"随机 {len(arm_contrastive['random_in_instrument'])}, "
                          f"丢弃 {len(arm_contrastive['dropped_from_ckpt'])})")

    if not durable_out:
        print(f'[红] --out 不在正式面根下 ⇒ 只打印不落盘: {out_dir} (lever yaml 落在 {lever_dir})')
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    p = os.path.join(out_dir, 'arch_equivalence.json')
    with open(p, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[面] {p}")
    print(f"  自检A pass={trust_a} | 自检B pass={check_b['pass']} "
          f"({check_b['default_keys']}→{check_b['contrastive_keys']} 键) | "
          f"原配vs V25: {arm_default['hit']}/{arm_default['instrument_keys']}"
          f" shape {len(arm_default['shape_mismatch'])} | has_contrastive vs V25: "
          f"{arm_contrastive['hit']}/{arm_contrastive['instrument_keys']} "
          f"随机{len(arm_contrastive['random_in_instrument'])} 丢弃{len(arm_contrastive['dropped_from_ckpt'])}")
    print(f"  verdict: {out['verdict']}")
    return out


# ---------------------------------------------------------------- lane_head 双初始化对撞闸
def lanehead_gate(args):
    """V25 臂那 10 只随机 lane_head 张量, 会不会改三目标读数? 用双初始化对撞证它不改。

    arch_equivalence.json 里 `lane_head.must_prove` 欠的就是这一条。判据三形缺一不可:
      正对照 = 两次装配之间**只有** lane_head 那 10 只张量不同 (撞确实发生了);
      被测   = 评估读的 out[3]/out[4]/out[5] (成因/类型/逐帧) 逐位全等;
      变异   = 把每只**非** lane_head 的浮点张量都加噪声 ⇒ 读数必须变 (否则"全等"
               只是比较器坏了)。没有第三条, 前两条可以被一个假仪器同时"满足"。
    另加结构条: 同一份 V25 权重分别装进 lever yaml 与默认 yaml, 三目标读数的**索引契约**
    (out 长度与 out[3..5] 形状) 必须一致 —— 评估是按下标取数的, lever 若在前插输出就会串位。

    一律走**消费通道** (模型 = `eval_acceptance.build_models`, 数据 = 它的 val_loader),
    不自建第二套装配, 否则证的是探针不是论文里那次数。
    """
    import torch
    import eval_acceptance as ea
    import train_config

    train_config.ensure_cwd_v22()
    out_dir = args.out if os.path.isabs(args.out) else _resolve(args.out)
    lever_yaml = os.path.join(out_dir, LEVER_NAME)
    arch_p = os.path.join(out_dir, 'arch_equivalence.json')
    for p, why in ((lever_yaml, 'E7 V25 臂的 model 配置'),
                   (arch_p, '键集对账面 (本闸的期望随机键名单出自它)')):
        if not os.path.isfile(p):
            raise SystemExit(f"[红] {why} 不在盘上: {p} ⇒ 先跑 `--arch-census` 再来对撞")
    arch = json.load(open(arch_p, encoding='utf-8'))
    expected_lane = sorted(
        arch['arm_has_contrastive_true']['vs_v25']['random_in_instrument'])
    if arch.get('face_status') != 'verified':
        raise SystemExit(f"[红] arch 面 face_status={arch.get('face_status')} ⇒ 键集本身没背书, 先修它")

    def ns(yaml_path):
        return argparse.Namespace(
            checkpoint=args.v25_ckpt, model_yaml=yaml_path,
            data_root=_resolve(DATASET), data_yaml='config/accident_detect.yaml',
            device=str(args.gate_device), clip_len=CLIP_LEN,
            batch_size=args.gate_batch, num_workers=args.gate_workers,
            frame_list=None, output_dir=out_dir, dump_scores=None, dump_limit=0)

    print(f"[对撞] 权重={args.v25_ckpt}\n       lever={lever_yaml}\n"
          f"       期望随机键 {len(expected_lane)} 只 (现读 arch 面) | 卡=cuda:{str(args.gate_device).split(',')[0]}")
    torch.manual_seed(args.gate_seed_a)
    tm1, val_loader, meta1, dev = ea.build_model_and_data(ns(lever_yaml))
    torch.manual_seed(args.gate_seed_b)
    _, tm2, dev2, _ = ea.build_models(ns(lever_yaml))
    torch.manual_seed(args.gate_seed_a)
    _, tm_def, _, _ = ea.build_models(ns(_resolve(MODEL_YAML_DEFAULT)))
    assert str(dev) == str(dev2), f'两次构建落在不同设备 {dev} vs {dev2}'

    sd1, sd2 = tm1.state_dict(), tm2.state_dict()
    if set(sd1) != set(sd2):
        raise SystemExit(f"[红] 两次装配键集不同 (缺 {sorted(set(sd1)-set(sd2))[:3]} "
                         f"多 {sorted(set(sd2)-set(sd1))[:3]}) ⇒ 对撞不成立, 先修装配")
    diff_keys = sorted(k for k in sd1 if not torch.equal(sd1[k], sd2[k]))
    lane = [k for k in diff_keys if '.lane_head.' in k]
    nonlane = [k for k in diff_keys if '.lane_head.' not in k]
    seed_lane_only = not nonlane

    batches = []
    for i, b in enumerate(val_loader):
        if i >= args.gate_batches:
            break
        batches.append(b)
    if not batches:
        raise SystemExit('[红] val_loader 一个批次都没给出 ⇒ 无读数字, 本闸不出结论')

    def readout(model, batch):
        """逐字复刻评估侧的取数与喂法 (eval_acceptance.py:372-406)。"""
        imgs = batch['img'].to(dev)
        imgs = imgs.float() / 255.0 if imgs.dtype == torch.uint8 else imgs.float()
        type_ids = batch.get('accident_type', None)
        if type_ids is not None:
            type_ids = torch.full_like(type_ids.to(dev, dtype=torch.long), 7)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            out = model(imgs, temporal_mode=True, type_ids=type_ids)
        return out, [out[3], out[4], out[5]]

    def cmp_pair():
        rows = []
        for bi, b in enumerate(batches):
            o1, r1 = readout(tm1, b)
            o2, r2 = readout(tm2, b)
            per = [{'slot': nm, 'bit_equal': bool(torch.equal(a, c)),
                    'max_abs_diff': (a.float() - c.float()).abs().max().item()}
                   for nm, a, c in zip(('cause', 'type', 'frame'), r1, r2)]
            rows.append({'batch': bi, 'n_out_a': len(o1), 'n_out_b': len(o2),
                         'shape_a': [list(t.shape) for t in r1], 'read': per})
        return rows, all(x['bit_equal'] for r in rows for x in r['read'])

    # 形 A = 换 seed 的两次装配 (字面上那条「双初始化对撞」)
    rows_seed, equal_seed = cmp_pair()

    # 形 A 撞不出全部 10 只: lane_head 的 1/4 是 GroupNorm (weight=1/bias=0 常数初始化,
    # detect_classify_head.py:183-186) ⇒ 与种子无关, 换 seed 永远撞不动那 4 只 (实测 6/10)。
    # 所以形 B 把期望的 10 只**逐只重掷** (含归一化的常数), 断言不变: 差异必须恰好这 10 只。
    for k in expected_lane:
        v = sd2[k]
        scale = float(v.float().std()) if v.numel() > 1 else 1.0
        v.copy_(torch.randn_like(v) * max(scale, 1e-3))
    forced_diff = sorted(k for k in sd2 if not torch.equal(sd1[k], sd2[k]))
    forced_collision = forced_diff == expected_lane
    rows_forced, equal_forced = cmp_pair()

    # 变异对照: 动**每一只**非 lane_head 的浮点张量 (参数 + 缓冲都算, BN 统计量也在读路径上)
    n_pert = 0
    for k, v in tm2.state_dict().items():
        if '.lane_head.' in k or not v.is_floating_point():
            continue
        v.add_(torch.randn_like(v) * 1e-3)
        n_pert += 1
    mut_diff = 0
    for b in batches:
        _, r1 = readout(tm1, b)
        _, r2 = readout(tm2, b)
        mut_diff += sum(0 if torch.equal(a, c) else 1 for a, c in zip(r1, r2))
    teeth = n_pert > 0 and mut_diff > 0

    # 结构条: lever yaml 不许把三目标读数的下标挪位
    o_def, r_def = readout(tm_def, batches[0])
    _, r_lev = readout(tm1, batches[0])
    contract = (len(o_def) == rows_seed[0]['n_out_a']
                and [list(t.shape) for t in r_def] == [list(t.shape) for t in r_lev])

    gate = {
        'seed_form_collision_keys': len(lane),
        'seed_form_lane_only': seed_lane_only,
        'forced_form_collision_all_expected': forced_collision,
        'readouts_bitwise_equal_seed_form': equal_seed,
        'readouts_bitwise_equal_forced_form': equal_forced,
        'mutation_control_has_teeth': teeth,
        'index_contract_stable': contract,
        'n_random_keys_seed_form': len(diff_keys),
        'n_random_keys_expected_from_arch': len(expected_lane),
        'non_lane_diff_keys': nonlane[:10],
        'forced_diff_keys': forced_diff,
        'n_perturbed_tensors': n_pert,
        'n_readouts_changed_by_mutation': mut_diff,
        'rule': '五形全真才准写「随机 lane_head 不改 V25 臂三目标读数」(seed 形有碰撞且只差 lane・'
                '强制形恰好 10 只・两形读数都逐位全等・变异对照有牙・下标契约稳); 任一红 ⇒ E7 数值不得进论文',
    }
    passed = all(gate[k] for k in (
        'seed_form_lane_only', 'forced_form_collision_all_expected',
        'readouts_bitwise_equal_seed_form', 'readouts_bitwise_equal_forced_form',
        'mutation_control_has_teeth', 'index_contract_stable')) and gate['seed_form_collision_keys'] > 0
    rep = {
        'face': 'E7_lanehead_dualinit_gate',
        'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'round': args.round, 'task': '#273', 'instrument': INSTRUMENT,
        'instrument_sha256_16': _instrument_sha(),
        'arm': 'v25 + lever yaml (has_contrastive: true)',
        'checkpoint': {'path': args.v25_ckpt, 'meta': {k: v for k, v in meta1.items()
                       if isinstance(v, (int, float, str, bool))}},
        'seeds': {'build_a': args.gate_seed_a, 'build_b': args.gate_seed_b},
        'batch_clips': args.gate_batch, 'n_batches': len(batches),
        'device': str(dev), 'clip_len': CLIP_LEN,
        'lever_yaml': {'path': lever_yaml, 'sha256_16': sha16(open(lever_yaml).read())},
        'arch_face': {'path': arch_p, 'face_status': arch.get('face_status')},
        'random_keys_seed_form': diff_keys, 'random_keys_forced_form': forced_diff,
        'random_keys_expected': expected_lane,
        'per_batch_seed_form': rows_seed, 'per_batch_forced_form': rows_forced,
        'gate': gate,
        'face_status': 'verified' if passed else 'draft_unverified',
        'verdict': None,
    }
    rep['verdict'] = (
        f'随机 lane_head 对三目标读数零影响 (换 seed 只差 {len(diff_keys)} 只 conv・强制重掷满 '
        f'{len(expected_lane)} 只・两形读数均逐位全等・变异对照有牙・下标契约稳) '
        f'⇒ E7 V25 臂可用 lever yaml 上正式考卷'
        if passed else
        f'闸未过: {{"seed形只差lane": {gate["seed_form_lane_only"]}, '
        f'"强制形恰好10只": {gate["forced_form_collision_all_expected"]}, '
        f'"seed形读数全等": {gate["readouts_bitwise_equal_seed_form"]}, '
        f'"强制形读数全等": {gate["readouts_bitwise_equal_forced_form"]}, '
        f'"变异有牙": {gate["mutation_control_has_teeth"]}, '
        f'"下标契约": {gate["index_contract_stable"]}}} '
        f'⇒ E7 数值禁入论文, 先归因 (非 lane 差异键 {nonlane[:3]})')
    print(f"  批次 {len(batches)} (每批 {args.gate_batch} clips・设备 {dev}) | seed 形差异 "
          f"{len(diff_keys)}/{len(expected_lane)} 只 (非 lane {len(nonlane)}) | "
          f"强制形差异 {len(forced_diff)} 恰好={forced_collision} | "
          f"读数全等 seed={equal_seed} forced={equal_forced} | "
          f"变异动 {n_pert} 只⇒变 {mut_diff} 处 | 下标契约={contract}")
    print(f"  verdict: {rep['verdict']}")
    if not _is_durable(out_dir):
        print(f'[红] --out 不在正式面根下 ⇒ 只打印不落盘: {out_dir}')
    else:
        p = os.path.join(out_dir, 'lanehead_dualinit_gate.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[面] {p}")
    if not passed:
        raise SystemExit(2)
    return rep


# ---------------------------------------------------------------- GT 改口量 (E7 的诚实边界)
DRIFT_AXES = (('scene_labels', '0'),      # ignore 的语义出处 = eval_acceptance.py:411 `frame_valid = scene_seq != 0`
              ('type_labels', '-1'))      # 同文件 type 分桶里 GT=-1 记"无槽"


def gt_drift(args):
    """「同一张考卷」保证的是**帧**相同, 不保证**标准答案**相同 ⇒ 把这个差量测出来。

    E7 两臂都拿今天这份 V26 正本判卷, 而 V25 那套权重当年是对着 V25 树的标签训的。
    考卷里每一帧 v25/v26 两份 txt 逐字比一遍, 按四档归桶 (相同 / 补标 / 撤标 / 真改口):
    补标与撤标只是覆盖面变化, **真改口**才是"同一帧的类被换了说法" ⇒ 引 E7 读数时
    必须把这两个数一起报出去, 否则会被读成纯粹模型对模型。
    """
    out_dir = args.out if os.path.isabs(args.out) else _resolve(args.out)
    exam = os.path.join(out_dir, 'common_frames.txt')
    man_p = os.path.join(out_dir, 'universe_manifest.json')
    for p in (exam, man_p):
        if not os.path.isfile(p):
            raise SystemExit(f"[红] 缺件: {p} ⇒ 先跑 --emit-face")
    names = [ln.strip() for ln in open(exam, encoding='utf-8') if ln.strip()]
    man = json.load(open(man_p, encoding='utf-8'))
    n_man = man['universes']['common_frames']['n']
    if len(names) != n_man:
        raise SystemExit(f"[红] 考卷 {len(names)} 行 ≠ 面里登记 {n_man} 帧 ⇒ 名单与账不同一张, 拒出数")

    def first(p):
        try:
            with open(p, encoding='utf-8') as f:
                return f.readline().strip()
        except FileNotFoundError:
            return None

    roots = {'v25': V25_ROOT, 'v26': _resolve(DATASET)}
    axes, missing_total = {}, 0
    for axis, ign in DRIFT_AXES:
        c = Counter()
        pairs = Counter()
        cls = {'v25': 0, 'v26': 0}
        for n in names:
            stem = os.path.splitext(n)[0] + '.txt'
            a = first(os.path.join(roots['v25'], axis, 'val', stem))
            b = first(os.path.join(roots['v26'], axis, 'val', stem))
            if a is None or b is None:
                c['缺文件'] += 1
                missing_total += 1
                continue
            if a != ign:
                cls['v25'] += 1
            if b != ign:
                cls['v26'] += 1
            if a == b:
                c['逐字相同'] += 1
            elif a == ign:
                c['补标 (v25=ignore→v26 有类)'] += 1
                pairs[f'{a}->{b}'] += 1
            elif b == ign:
                c['撤标 (v25 有类→v26=ignore)'] += 1
                pairs[f'{a}->{b}'] += 1
            else:
                c['真改口 (两侧都有类且不同)'] += 1
                pairs[f'{a}->{b}'] += 1
        tot = sum(c.values())
        axes[axis] = {
            'ignore_value': ign, 'n_frames': tot,
            'classed': {**cls, 'ratio_v26_over_v25': round(cls['v26'] / max(cls['v25'], 1), 2)},
            'counts': dict(sorted(c.items())),
            'pct': {k: round(100.0 * v / max(tot, 1), 3)
                    for k, v in sorted(c.items()) if k != '缺文件'},
            'top_transitions': pairs.most_common(8),
        }
        print(f"  [{axis}] 帧 {tot} | 相同 {c['逐字相同']} ({100*c['逐字相同']/max(tot,1):.2f}%) | "
              f"补标 {c['补标 (v25=ignore→v26 有类)']} | 撤标 {c['撤标 (v25 有类→v26=ignore)']} | "
              f"真改口 {c['真改口 (两侧都有类且不同)']} ({100*c['真改口 (两侧都有类且不同)']/max(tot,1):.2f}%) | "
              f"缺文件 {c['缺文件']} | 有类帧 v25={cls['v25']} v26={cls['v26']} "
              f"({cls['v26']/max(cls['v25'],1):.1f}x)")
    gate = {'n_frames_match_manifest': len(names) == n_man, 'zero_missing_label_files': missing_total == 0,
            'rule': '两条形同真才出面: 考卷行数须等于宇宙面登记的帧数・两份标签树都逐帧读得到'}
    rep = {
        'face': 'E7_gt_drift_v25_vs_v26', 'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'round': args.round, 'task': '#273', 'instrument': INSTRUMENT,
        'instrument_sha256_16': _instrument_sha(),
        'exam': {'path': exam, 'n': len(names), 'sha256_16': sha16('\n'.join(names)),
                 'manifest_n': n_man},
        'roots': roots, 'axes': axes, 'gate': gate,
        'face_status': 'verified' if all(gate.values()) else 'draft_unverified',
        'verdict': (f"E7 两臂共用今天这份 GT: scene 真改口 "
                    f"{axes['scene_labels']['pct'].get('真改口 (两侧都有类且不同)')}%・"
                    f"补标 {axes['scene_labels']['pct'].get('补标 (v25=ignore→v26 有类)')}%; "
                    f"type 真改口 {axes['type_labels']['pct'].get('真改口 (两侧都有类且不同)')}%・"
                    f"撤标 {axes['type_labels']['pct'].get('撤标 (v25 有类→v26=ignore)')}% "
                    f"; scene 有类帧 v25={axes['scene_labels']['classed']['v25']} → "
                    f"v26={axes['scene_labels']['classed']['v26']} "
                    f"({axes['scene_labels']['classed']['ratio_v26_over_v25']}x) "
                    f"⇒ 帧相同而答案不完全相同, 引 E7 对照必须同框带这两个口径"),
    }
    print(f"  verdict: {rep['verdict']}")
    if not _is_durable(out_dir):
        print(f'[红] --out 不在正式面根下 ⇒ 只打印不落盘: {out_dir}')
    else:
        p = os.path.join(out_dir, 'gt_drift_v25_vs_v26.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[面] {p}")
    if not all(gate.values()):
        raise SystemExit(2)
    return rep


# ---------------------------------------------------------------- 两臂读数面
def _argmax(v):
    return max(range(len(v)), key=lambda i: v[i])


def _recalls(cm):
    """混淆矩阵 [gt][pred] 逐类召回 (并列报, 不只报总准确率 —— 总准确率会被多数类买走)。"""
    out = {}
    for i, lab in enumerate(('normal', 'pre_crash', 'crash')):
        n = sum(cm[i])
        out[lab] = round(100.0 * cm[i][i] / n, 2) if n else None
    return out


def arm_readcounts(rows):
    """从一臂的逐 clip 原始概率面重算场景读数 (纯函数, 自测锚 26–27)。

    帧头口径与 eval_acceptance 在册那把尺逐位同形: 有效帧=scene_gt≠0, clip 判卷=最后一个
    有效帧的**帧头**预测。clip 头 (out[1]) 另立一张表: GT=clip_scene_gt (1/2/3→0/1/2),
    0=无 clip 级标注 ⇒ 不进分母 (拿 ignore 当 normal 会把召回买高)。
    """
    corr = tot = pred_crash_frames = 0
    cm = [[0] * 3 for _ in range(3)]
    fcm = [[0] * 3 for _ in range(3)]
    cmh = [[0] * 3 for _ in range(3)]
    crash_probs = []
    for x in rows:
        n = x['n_real_frames']
        g, p = x['scene_gt_seq'][:n], x['frame_scene_prob'][:n]
        valid = [i for i, gv in enumerate(g) if gv != 0]
        for i in valid:
            gi, k = g[i] - 1, _argmax(p[i])
            tot += 1
            pred_crash_frames += int(k == 2)
            corr += int(k == gi)
            if 0 <= gi < 3:
                fcm[gi][k] += 1
        if valid:
            li = valid[-1]
            cg, cp = g[li] - 1, _argmax(p[li])
            if 0 <= cg < 3 and 0 <= cp < 3:
                cm[cg][cp] += 1
        cgt, h = x.get('clip_scene_gt'), x.get('clip_scene_prob')
        if cgt in (1, 2, 3) and h is not None:
            cmh[cgt - 1][_argmax(h)] += 1
            if cgt == 3:
                crash_probs.append(h[2])
    return {
        'frame_acc': round(100.0 * corr / max(tot, 1), 2),
        'frame_n': tot, 'frame_pred_crash': pred_crash_frames,
        'frame_confusion': cm, 'frame_recalls': _recalls(cm),
        'per_frame_confusion': fcm, 'per_frame_recalls': _recalls(fcm),
        'clip_head_confusion': cmh, 'clip_head_n': sum(map(sum, cmh)),
        'clip_head_recalls': _recalls(cmh),
        'clip_head_pred_crash_total': sum(r[2] for r in cmh),
        'clip_head_crash_prob_mean_pct_on_crash_clips':
            round(sum(crash_probs) * 100.0 / max(len(crash_probs), 1), 2),
    }


def _clip_frame_trees(clip_len, batch_size):
    """重出现场建 clip 的那棵树 ⇒ 每 clip 的帧名 + 两棵树的场景标签 (CPU, 分钟级)。

    动因: 面上的行只有 video_id 与 GT 序列, 没有帧文件名 ⇒ 想把 V25 那套**当年的答案**
    贴到同一批概率上, 只能按评估侧同一套构建 (tile 排除 + 考卷过滤) 重建, 重建完必须拿
    "逐 clip 重出的 v26 GT 序列 == 面上 scene_gt_seq" 当对齐闸, 对不齐就出声作废。
    """
    import torch
    import eval_acceptance as ea
    from dataset import ClipDataset
    from ultralytics.utils import DEFAULT_CFG  # noqa: F401  (build_base_dataset 内部要)

    class _StrideStub:
        stride = torch.tensor([32])

    exam = os.path.join(_resolve(FACE_DIR_REL), 'common_frames.txt')
    ns = argparse.Namespace(data_root=_resolve(DATASET), data_yaml='config/accident_detect.yaml',
                            batch_size=batch_size, num_workers=4, clip_len=clip_len,
                            frame_list=exam)
    base = ea.build_base_dataset(ns, _StrideStub(), 'val')
    cd = ClipDataset(base, clip_len=clip_len, stride=8, max_objs=8)
    labels = cd.base.labels
    tile = {i for i in range(len(labels))
            if re.search(r'_t\d{2}\.[A-Za-z]+$', os.path.basename(str(labels[i]['im_file'])))}
    cd.clips = [c for c in cd.clips if not any(fi in tile for fi in c[0])]
    want = {ln.strip() for ln in open(exam, encoding='utf-8') if ln.strip()}
    kept, partial, missing = filter_clips_by_frame_list(cd.clips, labels, want)
    if missing:
        raise SystemExit(f"[红] 重建树上缺考卷帧 {missing} ⇒ 两臂考卷不是同一张, 读数作废")
    name = lambda fi: os.path.basename(str(labels[fi]['im_file']))  # noqa: E731

    def first(p):
        try:
            with open(p, encoding='utf-8') as f:
                return f.readline().strip()
        except FileNotFoundError:
            return None

    out = []
    for c in kept:
        fs = [name(fi) for fi in c[0]]
        out.append({
            'video_id': c[1], 'frames': fs,
            'gt26': [int(first(os.path.join(_resolve(DATASET), 'scene_labels', 'val',
                                            os.path.splitext(f)[0] + '.txt')) or 0) for f in fs],
            'gt25': [int(first(os.path.join(V25_ROOT, 'scene_labels', 'val',
                                           os.path.splitext(f)[0] + '.txt')) or 0) for f in fs],
        })
    return out


def two_arm_readout(args):
    """E7 两臂读数面 (CPU): 三目标并排 + 帧头/clip 头双口径 + 两棵树对判。

    动因 (G-319・#273): V25 臂的**帧头 crash 列整列为 0**, 而 clip 级场景头从没进过任何读数面
    ⇒ 补出这张面才能回答"V25 的 crash 信号到底在不在这儿", 并顺手把"对着 V25 自己那套答案判"
    也算出来 (否则会被读成"只是 V26 换了说法把 V25 判死了")。
    新尺先对上在册尺: 两臂的 frame_acc 与 scene_confusion_matrix 必须由本尺从原始概率重算到
    与 eval_acceptance.json 逐位相同, 对不上 ⇒ 作废 (SystemExit 2), 不出新数。
    """
    t0 = time.time()
    arms = {}
    for tag, d in (('v26', args.arm26), ('v25', args.arm25)):
        dd = d if os.path.isabs(d) else _resolve(d)
        for p in (os.path.join(dd, 'clip_scores.jsonl'), os.path.join(dd, 'eval_acceptance.json')):
            if not os.path.isfile(p):
                raise SystemExit(f"[红] 缺件: {p} ⇒ 先跑该臂 eval_acceptance --dump-scores")
        rows = [json.loads(l) for l in open(os.path.join(dd, 'clip_scores.jsonl'),
                                            encoding='utf-8')]
        summ = json.load(open(os.path.join(dd, 'eval_acceptance.json'), encoding='utf-8'))
        arms[tag] = {'dir': dd, 'rows': rows, 'summ': summ}
        print(f"  [{tag}] {len(rows)} 行 ← {dd}")

    tree = _clip_frame_trees(16, 4)
    print(f"  [重建树] {len(tree)} clip (与面上行数须相等)")

    gate = {}
    a26, a25 = arms['v26'], arms['v25']
    gate['arms_rows_equal'] = len(a26['rows']) == len(a25['rows'])
    gate['rows_equal_summary_dump_rows'] = all(
        len(v['rows']) == v['summ']['dump_scores']['rows_written'] for v in arms.values())
    gate['tree_rebuild_row_count'] = len(tree) == len(a26['rows'])
    gate['alignment_video_and_gt'] = all(
        x['video_id'] == y['video_id'] and x['scene_gt_seq'] == y['scene_gt_seq']
        and x['clip_scene_gt'] == y['clip_scene_gt']
        for x, y in zip(a26['rows'], a25['rows']))
    gate['clip_scene_prob_present_all_rows'] = all(
        r.get('clip_scene_prob') is not None for v in arms.values() for r in v['rows'])
    # 重建树对齐闸: 逐 clip 重出的 v26 序列必须与面上记的一字不差
    # 重建树对齐闸: 逐 clip 重出的 v26 序列必须与面上记的一字不差。
    # G-323: 评估侧对短 clip 把 scene_gt_seq 补 0 到 clip_len (core/dataset.py:887-888),
    # 重建树只有真帧 ⇒ 旧式"头截断相等"把 435/2906 个短 clip 误判不齐。改按 n_real_frames 对齐并加验补尾形态。
    al = sum(1 for x, c in zip(a26['rows'], tree)
             if x['n_real_frames'] != len(c['gt26'])
             or x['scene_gt_seq'][:x['n_real_frames']] != c['gt26']
             or any(v != 0 for v in x['scene_gt_seq'][x['n_real_frames']:]))
    gate['tree_gt_matches_dump'] = (al == 0)
    print(f"  [对齐] 重建树 vs 面: 不齐 {al} clip / {len(tree)}")

    readouts, core = {}, {}
    for tag, v in arms.items():
        rc26 = arm_readcounts(v['rows'])
        gate[f"reproduces_registered_frame_acc_{tag}"] = (
            rc26['frame_acc'] == round(float(v['summ']['frame_acc']), 2))
        gate[f"reproduces_registered_scene_confusion_{tag}"] = (
            rc26['frame_confusion'] == v['summ']['scene_confusion_matrix'])
        n_crash_clips = sum(1 for r in v['rows'] if r['clip_scene_gt'] == 3)
        gate[f"clip_gt_crash_equal_type_denom_{tag}"] = (n_crash_clips == v['summ']['type_total_crash'])
        rows25 = [dict(r, scene_gt_seq=c['gt25'][:len(r['scene_gt_seq'])], clip_scene_gt=None)
                  for r, c in zip(v['rows'], tree)]
        readouts[tag] = {'vs_v26_gt': rc26, 'vs_v25_gt': arm_readcounts(rows25)}
        s = v['summ']
        core[tag] = {k: s.get(k) for k in (
            'scene_acc', 'scene_bal_acc', 'scene_macro_f1', 'frame_acc', 'type_acc',
            'type_acc_cv', 'type_cv_total', 'type_cv_correct', 'type_n_labeled_out',
            'type_n_unslotable', 'cause_recall', 'cause_precision', 'cause_f1',
            'cause_active_clips', 'detection_latency_s_nominal_fps', 'detection_latency_s_eff',
            'detection_latency_n', 'early_detection_rate', 'PDR', 'FAR', 'total_clips',
            'fps_nominal', 'fps_eff')}
        core[tag]['checkpoint'] = s['checkpoint']
        core[tag]['ckpt_sha256_16'] = s['checkpoint_identity']['sha256_16']
        core[tag]['ckpt_bytes'] = s['checkpoint_identity']['bytes']
        print(f"  [{tag}] 帧头 crash 召回 {rc26['frame_recalls']['crash']}%・"
              f"预测 crash 帧 {rc26['frame_pred_crash']}・clip 头 crash 召回 "
              f"{rc26['clip_head_recalls']['crash']}%・clip 头共判 crash "
              f"{rc26['clip_head_pred_crash_total']} 条")

    drift_p = os.path.join(_resolve(FACE_DIR_REL), 'gt_drift_v25_vs_v26.json')
    if not os.path.isfile(drift_p):
        raise SystemExit(f"[红] 缺 GT 漂移面: {drift_p} ⇒ 先跑 --gt-drift")
    drift = json.load(open(drift_p, encoding='utf-8'))
    gate['gt_drift_face_verified'] = (drift.get('face_status') == 'verified')
    cov = {ax: drift['axes'][ax]['classed'] for ax in ('scene_labels', 'type_labels')}

    gate['rule'] = ('六道同真才出面: 两臂行数等・行内 video_id/GT 逐位齐・重建树 GT 与面一字不差・'
                    'clip 头列全非空・本尺重算的 frame_acc/混淆矩阵与在册面逐位相同・crash clip 数=类型分母')
    ok = all(bool(v) for k, v in gate.items() if k != 'rule')
    rep = {
        'face': 'E7_two_arm_readout', 'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'round': args.round, 'task': '#273', 'instrument': INSTRUMENT,
        'instrument_sha256_16': _instrument_sha(), 'exam_n_frames': len(
            [ln for ln in open(os.path.join(_resolve(FACE_DIR_REL), 'common_frames.txt'),
                               encoding='utf-8') if ln.strip()]),
        'arms': {t: {'dir': v['dir'], 'rows': len(v['rows'])} for t, v in arms.items()},
        'calibers': {
            'frame_head': '逐帧口径 (eval_acceptance 在册那把尺): 有效帧=scene_gt≠0・clip 判卷=最后一个有效帧',
            'clip_head': 'clip 级场景头 out[1] 三档判卷・GT=clip_scene_gt (1/2/3), 0=无标注不进分母',
            'vs_v25_gt': '同一批概率对着 V25 当年那套标签判 (剥掉尺子错位的可能)',
            'cross_phase_ban': '跨相位禁比 type_cv/metric/total 绝对线 (PTCL 课程分母随相变)',
        },
        'core_objectives': core,
        'scene_readouts': readouts,
        'gt_label_coverage': cov,
        'gt_drift_face': {'path': drift_p, 'instrument_sha256_16': drift['instrument_sha256_16'],
                          'written_at': drift['written_at']},
        'lanehead_gate_face': os.path.join(_resolve(FACE_DIR_REL), 'lanehead_dualinit_gate.json'),
        'gate': gate,
        'face_status': 'verified' if ok else 'rejected',
        'verdict': (
            f"同一张考卷 (${len(a26['rows'])} clip): V26 帧头 crash 召回 "
            f"{readouts['v26']['vs_v26_gt']['frame_recalls']['crash']}% / clip 头 "
            f"{readouts['v26']['vs_v26_gt']['clip_head_recalls']['crash']}%; V25 帧头 "
            f"{readouts['v25']['vs_v26_gt']['frame_recalls']['crash']}% (预测 crash 帧 "
            f"{readouts['v25']['vs_v26_gt']['frame_pred_crash']}) / clip 头 "
            f"{readouts['v25']['vs_v26_gt']['clip_head_recalls']['crash']}% (共判 crash "
            f"{readouts['v25']['vs_v26_gt']['clip_head_pred_crash_total']} 条)。V25 **对着自己那套答案**判仍是帧头 crash "
            f"{readouts['v25']['vs_v25_gt']['frame_recalls']['crash']}% (有类帧仅 "
            f"{readouts['v25']['vs_v25_gt']['frame_n']}) ⇒ 不是 V26 换说法判死的。"
            f"同时必须同框: 考卷上场景有类帧 v25={cov['scene_labels']['v25']} → v26="
            f"{cov['scene_labels']['v26']} ({cov['scene_labels']['ratio_v26_over_v25']}x) ⇒ "
            f"E7 读的是「同考卷・各自数据」的两版端到端差, **不是控制变量的架构消融"),
        'elapsed_s': round(time.time() - t0, 2),
    }
    print(f"  verdict: {rep['verdict']}")
    if not ok:
        print(f"[红] 闸未过 {[k for k, v in gate.items() if k != 'rule' and not v]}")
    face_root = os.path.dirname(a26['dir'])
    if not _is_durable(face_root):
        print(f"[红] 面根不在正式 paper 根下 ⇒ 只打印不落盘: {face_root}")
    else:
        p = os.path.join(face_root, 'two_arm_readout.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[面] {p}")
    if not ok:
        raise SystemExit(2)
    return rep


# ---------------------------------------------------------------- 自测
GATE_COMMON = 'common 非空'
GATE_FRAME = '帧级零泄漏'
GATE_SEG = '段级零重叠'
GATE_RATIO = '共同宇宙非平凡'


def _gate(checks, prefix):
    """按前缀取闸读数: 闸名 wording 被改仍能对上; 闸被删掉则自测直接炸 (不许静默少测)。"""
    for name, passed, _ in checks:
        if name.startswith(prefix):
            return passed
    raise AssertionError(f"[红] 闸不见了: {prefix}")


def selftest(args):
    """合成目录树验不变量闸「有牙」: 注入脏帧必须变红, 撤掉必须转绿。

    底本 (六位帧号 ⇒ extract_video_id 走回退分支, 段级键 = 去掉最后一段):
      v26_val   = a1 a2 b1 b2 d1 c1 readme.txt   v26_train = z1 a3 w1
      v25_val   = a1 a2 b1 b2 d1 y1 w1            v25_train = x1 c1
      a*=real_a_00_*  b*=syn_b_00_*  d*=real_d_00_*  c*=tum_c_00_*
    ⇒ common_raw = {a1,a2,b1,b2,d1}; 段级剔 {a1,a2} (a3 在 V26train 露过同一段)
      ⇒ common = {b1,b2,d1}; 剔完 3/5 = 60% ⇒ 四闸在底本上全绿 (不是恒红死闸)。
    四条变异各打一条闸: 跳过段级剔除⇒只红段级; 塞 train 帧进考卷⇒红帧级;
    键函数过粗把宇宙剔光⇒红非平凡; 剔到空⇒红「common 非空」。帧级闸与段级闸是
    嵌套的 (帧级红必带段级红), 所以「段级闸有独立牙」只能靠变异①证明。
    """
    import shutil
    root = tempfile.mkdtemp(prefix='e7univ_selftest_')
    res = []

    def touch(rel, names):
        p = os.path.join(root, rel)
        os.makedirs(p, exist_ok=True)
        for n in names:
            open(os.path.join(p, n), 'w').close()
        return p

    A1, A2, A3 = ('real_a_00_000001.jpg', 'real_a_00_000002.jpg', 'real_a_00_000003.jpg')
    B1, B2, D1 = 'syn_b_00_000001.jpg', 'syn_b_00_000002.jpg', 'real_d_00_000001.jpg'
    C1, X1, Y1, Z1, W1 = ('tum_c_00_000001.jpg', 'real_x_00_000001.jpg',
                          'real_y_00_000001.jpg', 'real_z_00_000001.jpg',
                          'real_w_00_000001.jpg')
    dirs = {
        'v26_val': touch('v26/val', [A1, A2, B1, B2, D1, C1, 'readme.txt']),
        'v26_train': touch('v26/train', [Z1, A3, W1]),
        'v25_val': touch('v25/val', [A1, A2, B1, B2, D1, Y1, W1]),
        'v25_train': touch('v25/train', [X1, C1]),
    }
    u = build_universes(dirs)
    chk = invariants(u)
    res.append(('1 common_raw = 两臂 val 的交集 (只在一侧的 c1/y1/w1 不进)',
                u['common_raw'] == {A1, A2, B1, B2, D1}))
    res.append(('2 段级剔 {a1,a2}: 同段在 V26train 露过 (a3) 就整段出局',
                u['seg_excluded'] == {A1, A2}))
    res.append(('3 考卷 = 剔完的 common = {b1,b2,d1}', u['common'] == {B1, B2, D1}))
    res.append(('4 底本自身四闸全绿 (证明不是恒红的死闸)', all(p for _, p, _ in chk)))
    res.append(('5 全量 val 考 V25 臂的记忆污染被抓到 (c1 ∈ V25train)',
                u['full_val_leak_into_v25_train'] == {C1}))
    res.append(('6 反向污染也被抓到 (w1: V25val 独有却落在 V26train)',
                u['v25_only_in_v26_train'] == {W1}))
    res.append(('7 只在一侧 val 的帧被单列 (y1,w1)', u['v25_only_frames'] == {Y1, W1}))
    # 变异①: 有人删掉段级剔除那行 ⇒ common 退化成 common_raw。a1/a2 本身不在任何 train
    # 目录 ⇒ 帧级闸查不出来, 只有段级闸能红 —— 这条就是段级闸的独立牙。
    cb = invariants(dict(u, common=u['common_raw']))
    res.append(('8 变异①: 跳过段级剔除 ⇒ 只红段级闸 (帧级/非平凡/非空仍绿)',
                _gate(cb, GATE_SEG) is False and _gate(cb, GATE_FRAME) is True
                and _gate(cb, GATE_RATIO) is True and _gate(cb, GATE_COMMON) is True))
    # 变异②: 直接把一只 train 里的帧写进考卷 ⇒ 最严的帧级闸必须红
    cf = invariants(dict(u, common=u['common'] | {C1}))
    res.append(('9 变异②: 注入 train 帧 ⇒ 帧级闸变红', _gate(cf, GATE_FRAME) is False))
    # 变异③: 键函数坏掉/过粗 ⇒ 考卷被悄悄剔光到 20%, 前置条件全绿也要被非平凡闸拦住
    cg = invariants(dict(u, common={B1}))
    res.append(('10 变异③: 剔到 20% ⇒ 只红非平凡闸 (防"悄悄剔光"出近空考卷)',
                _gate(cg, GATE_RATIO) is False and _gate(cg, GATE_FRAME) is True
                and _gate(cg, GATE_SEG) is True and _gate(cg, GATE_COMMON) is True))
    # 变异④: 剔到空 ⇒ 「common 非空」红 (空考卷不得出面)
    res.append(('11 变异④: 空宇宙 ⇒ 「common 非空」红',
                _gate(invariants(dict(u, common=set())), GATE_COMMON) is False))
    # 可逆性: 从盘上撤掉 a3 ⇒ 段级不再剔, 考卷变大且四闸仍全绿 (证明闸不是硬编码红)
    os.remove(os.path.join(dirs['v26_train'], A3))
    u3 = build_universes(dirs)
    res.append(('12 撤掉 a3 ⇒ 段级不剔 ({a1,a2} 回考卷) 且四闸仍全绿 (可逆, 非硬编码)',
                u3['seg_excluded'] == set() and u3['common'] == {A1, A2, B1, B2, D1}
                and all(p for _, p, _ in invariants(u3))))
    touch('v26/train', [A3])
    res.append(('13 幂等: 同一集合两次 build 的 sha 相同',
                sha16('\n'.join(sorted(u['common'])) + '\n')
                == sha16('\n'.join(sorted(build_universes(dirs)['common'])) + '\n')))
    # 键函数必须真是真相源的两条分支, 不是我自写的 rsplit (自写更粗 ⇒ 虚报泄漏)
    res.append(('14 video_key 回退分支: 含段标识, 不把 00/01 两段并起来',
                video_key('real_0jlyeLcFLVc_00_000012.jpg') == 'real_0jlyeLcFLVc_00'
                and video_key('real_0jlyeLcFLVc_01_000012.jpg') == 'real_0jlyeLcFLVc_01'))
    tum = ('tum_r09_s01_s040_camera_basler_north_16mm_1617874261_256000000'
           '_s040_camera_basler_north_16mm.jpg')
    res.append(('15 video_key 正则分支 (10 位时间戳) ≠ 朴素 rsplit ⇒ 键真是真相源',
                video_key(tum) == 'tum_r09_s01_s040_camera_basler_north_16mm'
                and video_key(tum) != os.path.splitext(tum)[0].rsplit('_', 1)[0]))
    res.append(('16 video_key 对无下划线短名不炸 (去扩展名)', video_key('abc.jpg') == 'abc'))
    res.append(('17 只收图不收说明文件 (readme.txt 不进宇宙)',
                'readme.txt' not in list_frames(dirs['v26_val'])))
    res.append(('18 正式面根判定 (草稿路径不得出面)',
                _is_durable('/tmp/whatever') is False
                and _is_durable(_resolve(DATASET + '/meta/paper/E7_common_split')) is True))
    res.append(('19 相邻目录前缀不许冒充正式面根 (meta/paperX ≠ meta/paper)',
                _is_durable(_resolve(DATASET + '/meta/paperX/E7')) is False))
    res.append(('20 段级剔的组数也被出面 (可复核到具体段)',
                {video_key(n) for n in u['seg_excluded']} == {'real_a_00'}))
    res.append(('21 lever yaml 文件名带尺度码且与原配同档 (防静默退成 n)',
                _scale_code(LEVER_NAME) == _scale_code(MODEL_YAML_DEFAULT) == 'm'))
    res.append(('22 _scale_code 对无尺度码的裸名判空 (锚 21 有牙, 不是恒真)',
                _scale_code('model_yaml_v25_arm_has_contrastive_true.yaml') == ''
                and _scale_code('e7lever/x.yaml') == ''))
    # 考卷过滤器 (评估侧 --frame-list 用的就是它): 跨界 clip 必须整条剔, 且缺帧要出声
    labs = [{'im_file': '/x/' + n} for n in (A1, A2, B1, C1, D1)]
    clips = [([0, 1], 'real_a_00'), ([2, 3], 'syn_b_00'), ([4], 'real_d_00')]
    kp, part, miss = filter_clips_by_frame_list(clips, labs, {A1, B1, D1})
    res.append(('23 过滤器＝整条都在名单才留 (跨界 clip 不半条混: 含 A2 的 clip0 与含 C1 的 clip1 都剔)',
                kp == [clips[2]] and part == 2 and miss == 0))
    res.append(('24 过滤器: 名单里有本树缺席的帧 ⇒ miss>0 出声 (两臂考卷不同一张, E7 不得结线)',
                filter_clips_by_frame_list(clips, labs, {A1, 'zz_absent.jpg'})[2] == 1))
    res.append(('25 过滤器 fail-closed: 空名单 ⇒ 一卷不考 (不是"等于全考")',
                filter_clips_by_frame_list(clips, labs, set()) == ([], 3, 0)))
    # G-319: 两臂读数尺那把纯函数 (arm_readcounts) 的锚 —— 手工可算的小例 + 两条后置断言。
    _mini = [
        {'n_real_frames': 4, 'scene_gt_seq': [1, 2, 3, 3],
         'frame_scene_prob': [[0.1, 0.6, 0.3], [0.2, 0.7, 0.1],
                              [0.1, 0.2, 0.7], [0.8, 0.1, 0.1]],
         'clip_scene_gt': 3, 'clip_scene_prob': [0.2, 0.3, 0.5]},
        {'n_real_frames': 2, 'scene_gt_seq': [0, 0],
         'frame_scene_prob': [[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]],
         'clip_scene_gt': 0, 'clip_scene_prob': [0.1, 0.1, 0.8]},
        {'n_real_frames': 2, 'scene_gt_seq': [1, 1],
         'frame_scene_prob': [[0.9, 0.05, 0.05], [0.85, 0.1, 0.05]],
         'clip_scene_gt': 1, 'clip_scene_prob': [0.9, 0.05, 0.05]},
        {'n_real_frames': 2, 'scene_gt_seq': [0, 2],
         'frame_scene_prob': [[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]],
         'clip_scene_gt': 2, 'clip_scene_prob': [0.1, 0.8, 0.1]},
    ]
    _r = arm_readcounts(_mini)
    res.append(('26 读数尺: 手工小例逐位对上 (帧 acc 71.43・confusion 只落最后一个有效帧・'
                'ignore 帧不进分母但所在 clip 仍判卷・clip_scene_gt=0 的 clip 不进 clip 头表)',
                _r['frame_acc'] == 71.43 and _r['frame_n'] == 7 and _r['frame_pred_crash'] == 1
                and _r['frame_confusion'] == [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
                and _r['frame_recalls'] == {'normal': 100.0, 'pre_crash': 100.0, 'crash': 0.0}
                and _r['clip_head_confusion'] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                and _r['clip_head_n'] == 3 and _r['clip_head_pred_crash_total'] == 1
                and _r['clip_head_recalls'] == {'normal': 100.0, 'pre_crash': 100.0, 'crash': 100.0}
                and _r['clip_head_crash_prob_mean_pct_on_crash_clips'] == 50.0))
    _r2 = arm_readcounts([dict(_mini[0], clip_scene_prob=None)] + _mini[1:])
    res.append(('27 读数尺变异对照: clip 头列缺失 (老面没这列) ⇒ clip 头表必须只剩非缺失那几条',
                _r2['clip_head_n'] == 2 and _r2['clip_head_pred_crash_total'] == 0
                and _r2['clip_head_crash_prob_mean_pct_on_crash_clips'] == 0.0
                and _r2['frame_n'] == 7))
    shutil.rmtree(root, ignore_errors=True)
    n_pass = sum(1 for _, p in res if p)
    for name, p in res:
        print(f"  {'PASS' if p else 'FAIL'} {name}")
    print(f"[自测] {n_pass}/{len(res)} " + ('PASS' if n_pass == len(res) else 'FAIL'))
    sys.exit(0 if n_pass == len(res) else 1)


def _is_durable(out_dir):
    return out_dir == PAPER_FACE_ROOT or out_dir.startswith(PAPER_FACE_ROOT + os.sep)


def main():
    ap = argparse.ArgumentParser(description='E7 共同考卷尺 (宇宙名单 + 架构对等性)')
    ap.add_argument('--emit-face', action='store_true', help='出宇宙名单面 (CPU, 秒级)')
    ap.add_argument('--arch-census', action='store_true', help='键集对账 (CPU, 需 torch)')
    ap.add_argument('--gt-drift', action='store_true',
                    help='量考卷上 v25/v26 标签的改口量 (CPU, 分钟级): 帧相同≠答案相同')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--out', default=FACE_DIR_REL)
    ap.add_argument('--round', default=None, help='缺省=现读 current_round.json')
    ap.add_argument('--v25-ckpt', default=os.environ.get('V25_CKPT', 'models/v25_p3c_best_temporal_v1.pt'))
    # 自检 A 的在册基线 886/886 是**逐键名**量的, 与哪一版交付件无关 ⇒ 默认跟着交付件走。
    # G-319: 3b 收官后交付件已换成 3c 轮8 代 (1e6828c8594519d1), 默认仍指 3b 会让忘传参的人
    # 把上一相的件当 V26 现值报出去 ⇒ 挪到 3c。键集若变⇒自检 A 红⇒本面作废, 不会静默混用。
    ap.add_argument('--v26-ckpt', default='runs/v26_dw_lane_g304_p3c/best_temporal.pt')
    # G-319(#273): lane_head 双初始化对撞闸 (兑现 arch 面的 must_prove)。
    ap.add_argument('--lanehead-gate', action='store_true',
                    help='随机 lane_head 是否改三目标读数: 两次装配只差那 10 只张量 ⇒ 读数须逐位全等 '
                         '(含变异对照与下标契约; 需 GPU, 分钟级)')
    ap.add_argument('--gate-device', default='1',
                    help='对撞尺用的卡 (build_models 取 cuda:<首 token>, 传 cpu 会拼成 cuda:cpu 崩)')
    ap.add_argument('--gate-batches', type=int, default=3, help='取 val_loader 前几批对撞')
    ap.add_argument('--gate-batch', type=int, default=2, help='每批 clip 数 (独立参数, 不与训练口径互撞)')
    ap.add_argument('--gate-workers', type=int, default=2)
    ap.add_argument('--gate-seed-a', type=int, default=0)
    ap.add_argument('--gate-seed-b', type=int, default=20261103)
    # G-319(#273): 两臂读数面 (CPU)。只读两臂已落的面, 不重跑 GPU。
    ap.add_argument('--two-arm-readout', action='store_true',
                    help='帧头/clip 头双口径＋两棵树对判・新尺须先复现在册 frame_acc/混淆矩阵')
    ap.add_argument('--arm26', default=None, help='V26 臂目录 (含 clip_scores.jsonl + eval_acceptance.json)')
    ap.add_argument('--arm25', default=None, help='V25 臂目录 (同上)')
    args = ap.parse_args()
    args.round = args.round or _current_round()
    print(f"[轮号] {args.round} (现读)" if args.round != 'ADHOC' else "[轮号] 未落戳")
    if args.selftest:
        return selftest(args)
    if args.arch_census:
        return arch_census(args)
    if args.gt_drift:
        return gt_drift(args)
    if args.lanehead_gate:
        return lanehead_gate(args)
    if args.two_arm_readout:
        if not args.arm26 or not args.arm25:
            raise SystemExit('[红] --two-arm-readout 必须同框给 --arm26/--arm25 ⇒ '
                             '缺一个就成了"半张考卷读数", 不许默认兜底')
        return two_arm_readout(args)
    return emit_face(args)


if __name__ == '__main__':
    main()
