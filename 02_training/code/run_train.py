#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_train.py — V26 泛化能力专项训练入口 (完全自包含)

依赖闭包已全部移植到 v26_training_package/scripts/ 与 scripts/core/,
数据根为 datasets/accident_v26/, 不再 import 任何 v22/v24/v25 目录文件。

用法:
  python run_train.py --phase 3c --name v26_s1 --epochs 20
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'core'))

# V26 环境变量覆盖 (在导入 train_config 之前设置)
os.environ['V26_MODE'] = '1'

# V26 增强覆盖
os.environ['V26_MOSAIC'] = '1.0'
os.environ['V26_MIXUP'] = '0.3'
os.environ['V26_COPY_PASTE'] = '0.1'
os.environ['V26_HSV_S'] = '0.9'
os.environ['V26_HSV_V'] = '0.6'
os.environ['V26_ERASING'] = '0.4'
os.environ['V26_CLOSE_MOSAIC'] = '10'

# V26 正则化覆盖
os.environ['V26_WEIGHT_DECAY'] = '0.2'
os.environ['V26_DROPOUT'] = '0.4'
os.environ['V26_LABEL_SMOOTHING'] = '0.1'

# V26 学习率
os.environ['V26_LR'] = '0.0003'


def apply_v26_overrides():
    """在训练循环前应用 V26 配置覆盖"""
    import train_config
    import base_config

    # 类型类别权重: 单一真相源 = base_config.TYPE_CLASS_WEIGHT。
    # losses.py 在 import 时从 base_config 绑值, 对 train_config 的重绑到不了损失 —
    # 旧版此处赋 [3.0, 3.5, ...] 是假接线, 打出的 [生效] 是谎值 (2026-10-22 核实修正)
    train_config.TYPE_CLASS_WEIGHT = base_config.TYPE_CLASS_WEIGHT

    print("=" * 60)
    print("V26 训练模式 (v26_training_package, 自包含)")
    print("=" * 60)
    # G-304(F5): 诚实接线表 —— 逐旋钮标"谁 READ 它"。
    #   原来这里只有一句"V26_* 未生效"的笼统警告, 而读者真正想知道的是
    #   "我改的这个数到底进没进训练"。逐条实测 (grep 全包 *.py, 排除 backups/):
    #   判据 = 有没有一处 READ, 不是"config 里有没有这个键" (G-300: config 有键 ≠ 有接线)。
    _WIRING = [
        # (旋钮, 状态, 依据)
        ('V26_MOSAIC/MIXUP/COPY_PASTE/HSV_S/HSV_V/ERASING/CLOSE_MOSAIC', '无读者·不生效',
         '唯一写者=本文件 :25-31, 全包无 READ'),
        ('V26_WEIGHT_DECAY/DROPOUT/LABEL_SMOOTHING/LR', '无读者·不生效',
         '唯一写者=本文件 :34-39, 全包无 READ'),
        ('TYPE_CLASS_WEIGHT', '生效',
         'base_config.TYPE_CLASS_WEIGHT (losses.py import 时绑值, 同源对象)'),
        ('scene_cls_weight (λ_scene)', '生效',
         'config/accident_detect.yaml → core/detection.py:574 读法; 相位3 由 phase3_train.py:314 同读'),
        ('--lr0', '生效', 'train.py argparse → phase3_train.py:339 base_lr'),
        ('--device 0,1 (相位3 卡数)', '只取第一张 ⇒ 横幅 GPUs=1',
         'phase3_train.py:117 world_size=int(os.environ.get("WORLD_SIZE",1)); 本文件把 --device 原样传给子命令、'
         '不 spawn torchrun ⇒ WORLD_SIZE 缺席 ⇒ :147 非 DDP 分支取 args.device.split(",")[0] ⇒ 相位3 全程单卡 '
         '(effective_batch=batch×accum×1 算术自洽)。真双卡＝regime 变更 (节拍/optim 状态/有效 batch 全变) '
         '⇒ 不在链上动, 只在重启窗口另立案, 且须先同尺复现已登记读数'),
        ('close_mosaic', '相位1/2 生效, 相位3 不生效',
         'train.py:596 写进 trainer.args; 相位3 训练通道 Mosaic 已被 phase3_train.py 的 train-aug 闸钉 p=0'),
        ('warmup_epochs', '相位3 用在体常量, 不读 config',
         'phase3_train.py:727 warmup_epochs=5 → pct_start; train.py:620/648/725 的 2.0 只作用于相位1/2'),
        ('sampler_ratios', '相位3 不生效 (均匀采样)',
         'core/detection.py:535 只在建相位1/2 sampler 时 READ; phase3_train.py:703 已自带诚实横幅'),
        ('λ_uasc (V22_INNOVATION_LOSS_WEIGHTS[0])', '无读者·不生效',
         'phase3_train.py:302 取 innov_w, 只有 :303 用了 [1]=cdcr; [0] 全包无 READ'),
        ('λ_lane / λ_supcon / λ_consist', '相位3 不入梯度 (只在检测侧 criterion 里生效)',
         '相位3 总损失=temporal_model.py:1076-1081 五项 (det/scene/cause/type/cdcr), 而 det_loss_val '
         '在 phase3_train.py:926 被显式钉 0.0 ⇒ 这三只权重进不了 P3 反传; 它们的真读者是 '
         'core/losses.py:556-557 + :749 + :963 (检测侧 total)。'
         'core/losses.py:583-586 那句 "Loss weights: … λ_lane/λ_supcon/λ_consist" 是建模型时打的, '
         '跨相串味 ⇒ 判相位3 监督份额只认 det/scene/cause/type/cdcr 五项'),
        ('λ_cdcr (V22_INNOVATION_LOSS_WEIGHTS[1])', '3b/3c 生效, 3a 恒 0',
         'phase3_train.py:303 (phase 门) + :994 is_cdcr_active'),
        ('agreement (UASC 输入)', '恒 1.0 ⇒ UASC 退化成固定平滑',
         'train_data.py:75 default_agreement; phase3_train.py:565/594 两处都传 1.0'),
        ('plan_config.py', '整个模块无 import 者',
         '全包 grep 只命中自身 ⇒ 该文件里的配方不是被执行的真相 (真相在 base_config/train_config/yaml)'),
    ]
    for _k, _s, _why in _WIRING:
        print(f"  [{_s}] {_k}\n          依据: {_why}")
    print(f"  [现值] TYPE_CLASS_WEIGHT={base_config.TYPE_CLASS_WEIGHT}")
    # G-307: 旧版这里印 "λ_scene(yaml)=<base_config.DEFAULT_SCENE_CLS_WEIGHT>", 标签说 yaml
    #   取的却是 0.8 兜底默认, 而真生效值是 yaml 的 5.243 (phase3_train.py:314 经 trainer.data 读)
    #   ⇒ 值与标签不符; 拆成两只各自署名的读数, 读不到 yaml 就不落该键, 不许拿默认顶。
    try:
        import yaml as _yaml
        _yaml_path = os.path.join(os.path.dirname(SCRIPT_DIR), train_config.DEFAULT_YAML)
        with open(_yaml_path, encoding='utf-8') as _yh:
            _lam_eff = float(_yaml.safe_load(_yh)['scene_cls_weight'])
        print(f"  [现值] λ_scene(yaml 生效)={_lam_eff} ← {train_config.DEFAULT_YAML}"
              f" | λ_scene(base_config 默认)={getattr(base_config, 'DEFAULT_SCENE_CLS_WEIGHT', None)}"
              f" (仅 yaml 缺该键时才用)")
    except Exception as _lam_e:
        print(f"  [现值] λ_scene(yaml 生效)=读取失败({_lam_e}) ⇒ 不落值, 请以 phase3_train 的 \"scene=… [trainer.data[...]]\" 行为准")
    print("  ↑ 本表逐条 grep 实测; 新增旋钮时先加 READ 者再改本表, 别只加 config 键")
    print("=" * 60)


def _dup_launcher_claim(name):
    """G-307(#282) 双开认领闸的探针: 列出在体的其它 run_train.py **主进程** [(pid, phase), ...], 只保留 --name 相同的。

    三条硬约束:
      1) 只认 argv[0] 是 python 解释器且前两个 token 里有 run_train.py ⇒ 不会被 `/bin/bash -c ...` 这类
         包装串或我自己的查询命令自匹配 (pgrep -f 假阳性那一类);
      2) 排除自身 pid;
      3) **闸本身永不抛** —— 任何异常都当成"没撞车"放行并出声。发车被误拦比双开更糟 (fail-open)。
    """
    hit = []
    try:
        me = str(os.getpid())
        for ent in os.listdir('/proc'):
            if not ent.isdigit() or ent == me:
                continue
            try:
                with open('/proc/%s/cmdline' % ent, 'rb') as fh:
                    raw = fh.read()
            except Exception:
                continue
            tok = [t.decode('utf-8', 'replace') for t in raw.split(b'\x00') if t]
            if len(tok) < 2 or not tok[0].startswith('python'):
                continue
            if not tok[1].endswith('run_train.py'):
                continue
            try:
                i_p, i_n = tok.index('--phase'), tok.index('--name')
            except ValueError:
                continue
            if i_p + 1 < len(tok) and i_n + 1 < len(tok) and tok[i_n + 1] == str(name):
                hit.append((ent, tok[i_p + 1]))
    except Exception as _claim_e:
        print(f"  [认领闸·探针异常⇒放行] {_claim_e} (fail-open, 不代表撞车)")
        return []
    return hit


def _claim_gate(phase, name, allow_env=False):
    """双开认领闸的决策面 (纯函数, 便于离线三对照): 返回 (action, pid, 文案)。

    action ∈ {'none', 'block', 'allow_dup', 'cross_warn'}:
      block      = 同 (--phase, --name) 已在体 ⇒ 硬拒 (best/last 互顶、显存互抢, 今日实测撞过);
      cross_warn = 同 --name 但跨相位 ⇒ **只出声不拦** —— 相位边界上"上一相进程未死透就发下一相"是
                   合法时序 (3a 收官与 3b 发车只差两分钟), 硬拦会把链卡死; 保护成果优先于洁癖。
      探针异常 ⇒ 'none' ⇒ **fail-open**: 发车被误拦比双开更糟, 所以闸坏了一定放行并出声。
    """
    hit = _dup_launcher_claim(name)
    same = [(p, ph) for p, ph in hit if str(ph) == str(phase)]
    if same:
        pid, _ = same[0]
        who = f"同 (--phase={phase}, --name={name}) 的 run_train.py 在体: pid={pid}"
        if allow_env:
            return ('allow_dup', pid, f"  [认领闸·V26_ALLOW_DUP=1 显式放行] {who} ⇒ 后果自负 (best/last 件互顶、显存互抢)")
        return ('block', pid, f"[认领闸·拒启] {who} ⇒ 双开会互顶 best/last 并抢显存。"
                              f"先确认那只的生死; 确实要并行请显式 V26_ALLOW_DUP=1")
    cross = [(p, ph) for p, ph in hit if str(ph) != str(phase)]
    if cross:
        pid, ph = cross[0]
        return ('cross_warn', pid,
                f"  [认领闸·跨相位提示·不拦] --name={name} 已有 pid={pid} 在跑 (--phase={ph}), 本次要起 --phase={phase}。"
                f"链按设计串行 ⇒ 若那只还没收尾, 两相会抢同一张卡; 确认过生死再继续发车。")
    return ('none', None, '')


def main():
    parser = argparse.ArgumentParser(description='V26 泛化能力专项训练')
    parser.add_argument('--phase', type=str, default='3c')
    parser.add_argument('--name', type=str, default='v26_s1')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='accident-both')
    parser.add_argument('--clip-len', type=int, default=16)
    parser.add_argument('--skip-quality-gate', action='store_true', default=False,
                        help='显式跳过数据质量门禁（不推荐）。G-239 起门禁实装：三道门现值全绿'
                             '（scene 标签 346,048 / 质量报告 90≥60 / cause 判定在场 35.61%%≥30%%）')
    parser.add_argument('--no-pfrpn', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--chain-from', type=str, default=None,
                        help='V26-G247/G258: 链式加载的权重路径 (任意阶段显式钉种子, 覆盖按运行名推导)')
    parser.add_argument('--lr0', type=float, default=None,
                        help='G-260: 覆盖学习率 (链式重启用微调档, 防 LR 回峰打散已收敛判界)')
    parser.add_argument('--resume', type=str, default=None,
                        help='G-274: 相位3 断点续训的时序检查点 (绝对路径)。'
                             '透传给 train.py --resume，由 phase3_train 的 load_training_state '
                             '恢复 权重/优化器/scheduler/AMP/epoch/best_metric。')
    parser.add_argument('--seed', type=int, default=None,
                        help='G-319(#273): 全局随机种子, 透传给 train.py --seed (相位3 落到 '
                             'random/torch/cuda/numpy)。默认 None＝不注入, argv 与旧版逐字节相同 '
                             '(由 train.py 自己的 default=0 兜底); 负数直接拒 (torch.manual_seed 会炸)。')
    # G-350: 车道线微调透传（默认全 None/False ⇒ 旧命令 argv 逐字节不变）
    parser.add_argument('--yaml', type=str, default=None,
                        help='G-350: 透传数据配置 yaml (如 config/culane_lane.yaml)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='G-350: 透传数据根目录 (覆盖 ds_config 默认)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='G-350: 透传 batch (lane 微调单卡用, 默认仍走 train.py)')
    parser.add_argument('--no-balanced-sampler', action='store_true', default=False,
                        help='G-350: 透传关 scene 均衡采样 (CULane 全 ignore, 均衡无意义)')
    args = parser.parse_args()
    if args.seed is not None and args.seed < 0:
        print(f"  [FATAL] --seed={args.seed} 为负 ⇒ torch.manual_seed 不接受, 拒绝发车")
        sys.exit(2)

    # G-307(#282): 发车前的双开认领闸 —— 同日实测撞两次 (巡检 cron 与一次性 cron 覆盖同一动作 / 同一只权重的
    #   贵尺发了两遍把 GPU1 顶到 40,164/40,960 MiB)。双开的代价不是浪费算力, 而是**互顶 best/last 件 + 抢同一
    #   张卡的显存, 报错的往往是在跑的那条链**。两级判据: 同一个 (--phase, --name) 已有活主进程 ⇒ 硬拒 (rc=3);
    #   同 --name 跨相位 ⇒ 只出声不拦 —— 相位收官与下一头发车本来就只隔几分钟, 硬拦会把链卡死。
    _act, _dup_pid, _msg = _claim_gate(args.phase, args.name, bool(os.environ.get('V26_ALLOW_DUP')))
    if _msg:
        print(_msg)
    if _act == 'block':
        sys.exit(3)

    # G-260 根治件: 链式种子路径透传给训练器, 供 build_optimizer 续接种子内
    # 优化器动量/AMP 状态 (种子里本来就有 443 组动量缓冲, 旧链式加载只读权重全丢)
    if args.chain_from:
        os.environ['V26_CHAIN_OPTIM_STATE'] = os.path.abspath(args.chain_from)

    apply_v26_overrides()

    # 导入并执行本包训练主循环 (scripts/train.py, 自包含副本)
    from train import main as train_main

    # 构造训练 args
    argv = [
        'train.py',
        '--phase', args.phase,
        '--name', args.name,
        '--dataset', args.dataset,
        '--clip-len', str(args.clip_len),
        '--epochs', str(args.epochs),
        '--device', args.device,
        '--no-pfrpn',
    ]
    if args.skip_quality_gate:      # G-239：门禁默认实装，仅显式要求时才跳过
        argv.append('--skip-quality-gate')
    if args.chain_from:
        argv += ['--chain-from', args.chain_from]
    if args.lr0 is not None:
        argv += ['--lr0', str(args.lr0)]
    if args.resume:
        argv += ['--resume', args.resume]
    if args.seed is not None:      # G-319(#273): 不设⇒不加这个旗标 ⇒ 老命令的 argv 逐字节不变
        argv += ['--seed', str(args.seed)]
    if args.yaml:                  # G-350: 不设⇒不加 ⇒ 老命令 argv 逐字节不变
        argv += ['--yaml', args.yaml]
    if args.data_root:
        argv += ['--data-root', args.data_root]
    if args.batch_size:
        argv += ['--batch-size', str(args.batch_size)]
    if args.no_balanced_sampler:
        argv.append('--no-balanced-sampler')
    sys.argv = argv

    train_main()


if __name__ == '__main__':
    main()
