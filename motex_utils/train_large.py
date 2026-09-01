"""大语料训练基建（motex 正式包成员，v5 及后续训练的正统入口）。

背景：wx 大语料（12 亿+ token，ids 文件 ~10GB）不能整读进内存，
且训练需要分块顺序读取（mmap 页局部性）、单周期/自定义 lr 计划、
里程碑评估（域内/acc/外推）与插拔公告。本模块把这些能力正式化，
参数化到与具体站点无关：
- load_ids_mmap()            mmap 惰性加载 ids
- chunked_seq_iter()         分块顺序滑窗迭代器（页缓存友好）
- eval_metrics()             域内 CE / acc / 外推 CE
- lr_schedule()              onecycle | cosine | linear | const(override)
- run_large_training()       完整训练循环（resume/checkpoint/早停/公告钩子）

用法示例见 dev/scripts/train/wx_big.py（薄调用即可，无需另写循环）。
"""
import json
import math
import os
import random
import time

import torch
from torch import nn


def load_ids_mmap(path):
    """mmap 加载 token ids（int64）。支持 .pt（torch mmap）与 .raw（int64 裸文件流式产物）。"""
    if path.endswith('.raw'):
        import numpy as np
        size = os.path.getsize(path) // 8
        mm = np.memmap(path, dtype='<i8', mode='r')
        return torch.from_numpy(mm).long()
    return torch.load(path, mmap=True).long().view(-1)


def chunked_seq_iter(ids, seq, seed=7, block_tokens=16_000_000):
    """分块顺序滑窗：ids 切大块（视图零拷贝）→ 每 epoch 打乱块序 → 块内顺序窗口。
    避免全随机窗口造成的共享盘换页风暴；返回 (x, y) 两个 tensor 窗口。"""
    r = random.Random(seed)
    n = len(ids)
    blocks = [ids[i:i + block_tokens] for i in range(0, n, block_tokens)]
    while True:
        r.shuffle(blocks)
        for b in blocks:
            m = len(b)
            st = r.randrange(0, min(seq, max(1, m - 1)))
            while st + seq + 1 < m:
                yield b[st:st + seq], b[st + 1:st + seq + 1]
                st += seq


@torch.no_grad()
def eval_metrics(net, val_ids, seq, device, batches=64, extrap_len=512):
    """三大评估指标（与历次训练口径一致）：
    - in_dist：验证集随机窗口的逐位置平均 CE
    - acc：上述窗口的 top-1 逐字准确率
    - extrap：验证集前段 512 窗口“看 256 续 256”的外推 CE（只统计后段）
    """
    loss = nn.CrossEntropyLoss(reduction='none')
    it = chunked_seq_iter(val_ids, seq, seed=3)
    tot1 = n1 = a1 = 0
    for _ in range(batches):
        x, y = next(it)
        lg, _, _ = net(x.unsqueeze(0).to(device), None, None)
        ce = loss(lg.reshape(-1, net.vocab_size), y.reshape(-1).to(device))
        tot1 += ce.sum().item(); n1 += ce.numel()
        a1 += (lg.reshape(-1, net.vocab_size).argmax(-1) == y.reshape(-1).to(device)).sum().item()
    tot2 = n2 = 0
    for st in range(0, 512 * 16, 512):
        c = val_ids[st:st + extrap_len]
        lg, _, _ = net(c[:extrap_len - 1].unsqueeze(0).to(device), None, None)
        ce = loss(lg.reshape(-1, net.vocab_size), c[1:extrap_len].unsqueeze(0).to(device).reshape(-1)).reshape(1, extrap_len - 1)
        tot2 += ce[:, (extrap_len - 1) // 2:].sum().item()
        n2 += ce[:, (extrap_len - 1) // 2:].numel()
    return tot1 / max(1, n1), a1 / max(1, n1), tot2 / max(1, n2)


def lr_schedule(opt, sched, total, warmup, lr, lr_override=0.0):
    """lr 计划工厂：
    - lr_override>0：恒 lr（降档精修）
    - sched='onecycle'：OneCycle（超级收敛，默认）
    - sched='cosine'：warmup 后 cosine 到 0
    - sched='wsd'：warmup-stable-decay（预训练长训候选；稳定段到 85%，尾段线性衰减到 0.1×）
    - 返回 scheduler（由调用方按 step0 步进）
    """
    if lr_override > 0:
        for g in opt.param_groups:
            g['lr'] = lr_override
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    if sched == 'cosine':
        return torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warmup if s < warmup else
            0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    if sched == 'wsd':
        stable_end = int(total * 0.85)
        return torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warmup if s < warmup else
            (1.0 if s < stable_end else max(0.1, 1.0 - 0.9 * (s - stable_end) / max(1, total - stable_end))))
    return torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total, pct_start=warmup / total,
        anneal_strategy='cos', div_factor=10.0, final_div_factor=200.0)


def run_large_training(
    build_net,           # callable(vocab) -> nn.Module（返回统一 (logits, state, aux) 接口）
    ids_path,            # ids pt（mmap）
    ckpt_dir, log_dir,   # 断点/日志目录
    *, vocab_size, seq=256, batch=16,
    lr=2e-4, warmup=1000, total=70000, patience=3, max_keep=8,
    sched='onecycle', lr_override=0.0, chunk=2000,
    announce=None,       # callable(str) 里程碑公告（可空）
    seed_data=7,
):
    """大语料训练主循环（正式版；wx_big.py 可退化为薄调用）。
    断点：{model,opt,step,best,stale}；日志：log_dir/train.log（JSON 行）。"""
    import torch
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    def teeprint(*a):
        txt = ' '.join(str(x) for x in a)
        print(txt, flush=True)
        with open(os.path.join(log_dir, 'train.log'), 'a', encoding='utf-8') as f:
            f.write(txt + '\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ids = load_ids_mmap(ids_path)
    teeprint(f'语料 token: {ids.numel():,} | vocab {vocab_size}')
    val_ids = ids[len(ids) - 200000:]
    train_ids = ids[:-200000]

    net = build_net(vocab_size).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.02)
    step0 = 0; best = float('inf'); stale = 0

    # resume
    latest = None
    if os.path.isdir(ckpt_dir):
        steps = sorted(int(f[11:-3]) for f in os.listdir(ckpt_dir)
                       if f.startswith('model_step_') and f.endswith('.pt'))
        if steps:
            latest = os.path.join(ckpt_dir, f'model_step_{max(steps):06d}.pt')
    if latest:
        sd = torch.load(latest, map_location=device)
        net.load_state_dict(sd['model']); opt.load_state_dict(sd['opt'])
        step0 = sd['step']; best = sd.get('best', best); stale = sd.get('stale', 0)
        teeprint(f'[resume] step {step0}, best={best:.4f}')

    sched_lr = lr_schedule(opt, sched, total, warmup, lr, lr_override)
    for _ in range(step0):
        sched_lr.step()

    it = chunked_seq_iter(train_ids, seq, seed=seed_data)
    loss = nn.CrossEntropyLoss()
    wall0 = time.time()
    log = []
    step = step0
    net.train()
    while step < total:
        xs, ys = [], []
        for _ in range(batch):
            x, y = next(it)
            xs.append(x); ys.append(y)
        xb = torch.stack(xs).to(device)
        yb = torch.stack(ys).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            lg, _, _ = net(xb, None, None)
            l = loss(lg.reshape(-1, net.vocab_size), yb.reshape(-1))
        l.float().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched_lr.step(); opt.zero_grad()
        step += 1
        if step % chunk == 0 or step == total:
            v1, acc, v2 = eval_metrics(net, val_ids, seq, device)
            minutes = (time.time() - wall0) / 60
            rec = {'step': step, 'in_dist': round(v1, 4), 'acc': round(acc, 4), 'extrap': round(v2, 4)}
            log.append(rec)
            torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'step': step,
                        'best': best, 'stale': stale},
                       os.path.join(ckpt_dir, f'model_step_{step:06d}.pt'))
            cks = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
            for old in cks[:-max_keep]:
                os.remove(os.path.join(ckpt_dir, old))
            teeprint(json.dumps(rec) + f' | {minutes:.0f}min | {time.strftime("%m-%d %H:%M")}')
            # Windows 工作集回收：单遍顺序读语料，已读 mmap 页不会再用，主动交回让 RSS 保持低位
            try:
                import ctypes
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            except Exception:
                pass
            if announce:
                announce(f'📈【大语料训练】step {step}/{total} | 域内 CE {v1:.3f} | acc {acc:.1%} | 外推 CE {v2:.3f} | {minutes:.0f}min')
            if v1 < best - 1e-4:
                best = v1; stale = 0
            else:
                stale += 1
                if stale >= patience:
                    teeprint(f'[early-stop] @{step} best={best:.3f}')
                    if announce:
                        announce(f'⏹ 早停 @step {step}（最佳域内 {best:.3f}）')
                    break
    json.dump(log, open(os.path.join(log_dir, 'history.json'), 'w'), indent=2, ensure_ascii=False)
    teeprint('done: ' + str(log[-3:] if len(log) > 3 else log))
    if announce:
        announce(f'🎉【大语料训练】完成 @step {step}：域内 {log[-1]["in_dist"]} / acc {log[-1]["acc"]:.1%}')
    return log