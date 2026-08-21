"""Motex 共用训练 / 评估 / 推理 / 断点保存工具。

供 motex/ 各版本 notebook 复用（模型需统一返回 (logits, state, aux_loss) 三元组，
无 MoE 的版本返回的 aux_loss 为 0.0）。
"""

import glob
import os
import time

import torch
from torch import nn

from deepseek_tokenizer import ds_token


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def evaluate_gpt(net, test_iter, device, max_batches=1):
    """在测试集上计算准确率（随机取 max_batches 个 batch）"""
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        skip = torch.randint(0, max_batches, (1,)).item()
        for i, (tokens, labels, valid_lens) in enumerate(test_iter):
            if i < skip:
                continue
            if i >= skip + max_batches:
                break
            tokens = tokens.to(device)
            labels = labels.to(device)
            valid_lens = valid_lens.to(device)
            logits, _, _ = net(tokens, valid_lens, None)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    net.train()
    return correct / total if total > 0 else 0


def save_checkpoint(net, optimizer, scheduler, scaler,
                    step, best_loss, total_loss, total_tokens,
                    ckpt_dir, max_keep):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = f'model_step_{step:06d}.pth'
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'step': step,
        'best_loss': best_loss,
        'total_loss': total_loss,
        'total_tokens': total_tokens,
    }, ckpt_path)
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'model_step_*.pth')))
    if len(all_ckpts) > max_keep:
        for old in all_ckpts[:len(all_ckpts) - max_keep]:
            os.remove(old)


def save_best_model(net, avg_loss, best_loss, best_model_path):
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(net.state_dict(), best_model_path)
        print(f"最佳模型已更新, loss = {best_loss:.4f}")
    return best_loss


def create_plot_components(plot_style, num_steps, ckpt_dir=None):
    """创建绘图组件（ckpt_dir 参数保留以备扩展）"""
    import matplotlib.pyplot as plt
    if plot_style == 'split':
        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax_loss.set_ylabel('loss')
        ax_loss.set_xlim(1, num_steps)
        ax_loss.grid(True)
        ax_acc.set_ylabel('acc')
        ax_acc.set_xlabel('step')
        ax_acc.set_ylim(0, 1)
        ax_acc.grid(True)
        line_loss, = ax_loss.plot([], [], 'b-', label='train loss')
        line_train_acc, = ax_acc.plot([], [], 'g-', label='train acc')
        line_test_acc, = ax_acc.plot([], [], 'r-', label='test acc')
        ax_loss.legend(loc='upper left')
        ax_acc.legend(loc='upper right')
        return fig, ax_loss, ax_acc, line_loss, line_train_acc, line_test_acc
    else:  # dual_axis
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.set_xlabel('step')
        ax1.set_xlim(1, num_steps)
        ax1.set_ylabel('loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.set_ylabel('acc', color='g')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='g')
        line_loss, = ax1.plot([], [], 'b-', label='train loss')
        line_train_acc, = ax2.plot([], [], 'g-', label='train acc')
        line_test_acc, = ax2.plot([], [], 'r-', label='test acc')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        return fig, ax1, ax2, line_loss, line_train_acc, line_test_acc


def update_plot(step, avg_loss, avg_train_acc, test_acc_cache,
                history, line_loss, line_train_acc, line_test_acc,
                axes, plot_style, num_steps, ckpt_dir=None):
    """更新动态图表，末尾将图保存到 ckpt_dir/training_plot.png"""
    import matplotlib.pyplot as plt
    from IPython import display
    history['steps'].append(step)
    history['loss'].append(avg_loss)
    history['train_acc'].append(avg_train_acc)
    history['test_acc'].append(test_acc_cache if test_acc_cache is not None else None)

    line_loss.set_data(history['steps'], history['loss'])
    line_train_acc.set_data(history['steps'], history['train_acc'])
    valid_idx = [i for i, v in enumerate(history['test_acc']) if v is not None]
    if valid_idx:
        line_test_acc.set_data(
            [history['steps'][i] for i in valid_idx],
            [history['test_acc'][i] for i in valid_idx]
        )

    if plot_style == 'split':
        ax_loss, ax_acc = axes
        ax_loss.relim(); ax_loss.autoscale_view()
        ax_acc.relim(); ax_acc.autoscale_view()
    else:
        ax1, ax2 = axes
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()

    display.display(plt.gcf(), display_id='training_plot', update=True)

    if num_steps <= step and ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        plt.savefig(os.path.join(ckpt_dir, "training_plot.png"))


def train_motex_ckpt_v2(net, loss, train_iter, test_iter, vocab_size, devices, num_steps,
                        lr=1e-4, warmup_steps=100, weight_decay=0.01,
                        ckpt_dir='./checkpoints', save_every=100, max_keep=50,
                        resume_from=None, accum_steps=1, plot_style='dual_axis',
                        aux_loss_coef=1e-2):
    """训练 Motex 模型（支持 AMP、梯度累积、断点恢复、动态绘图与 MoE aux_loss）"""
    from IPython import display
    from tqdm.notebook import tqdm

    os.makedirs(ckpt_dir, exist_ok=True)

    net.apply(init_weights)
    net = net.to(devices)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=weight_decay)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(num_steps - step) / max(1, num_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    device_type = 'cuda' if 'cuda' in str(devices) else 'cpu'
    scaler = torch.amp.GradScaler(device_type) if device_type == 'cuda' else torch.amp.GradScaler('cpu')

    start_step = 0
    best_loss = float('inf')
    best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
    if resume_from and os.path.isfile(resume_from):
        print(f"[resume] 找到文件: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=devices)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step']
        best_loss = checkpoint.get('best_loss', float('inf'))
        total_loss = checkpoint.get('total_loss', 0.0)
        total_tokens = checkpoint.get('total_tokens', 0)
        print(f"[resume] 从 step {start_step} 继续训练 (total_tokens={total_tokens})")
    else:
        total_loss, total_tokens = 0.0, 0
        if resume_from:
            print(f"[resume] 文件不存在: {resume_from}")

    fig, *axes, line_loss, line_train_acc, line_test_acc = create_plot_components(plot_style, num_steps, ckpt_dir=ckpt_dir)
    display.display(fig, display_id='training_plot')
    history = {'steps': [], 'loss': [], 'train_acc': [], 'test_acc': []}

    step = start_step
    train_acc_sum, train_acc_count = 0.0, 0
    test_acc_cache = None

    net.train()
    optimizer.zero_grad()
    wall_start = time.time()
    micro_batch_idx = 0
    pbar = tqdm(total=num_steps, desc="Training")
    if start_step > 0:
        pbar.update(start_step)

    while step < num_steps:
        for batch in train_iter:
            tokens, labels, valid_lens = batch
            tokens, labels, valid_lens = tokens.to(devices), labels.to(devices), valid_lens.to(devices)

            with torch.amp.autocast(device_type=device_type):
                logits, _, aux_loss = net(tokens, valid_lens, None)
                l = loss(logits.reshape(-1, vocab_size), labels.reshape(-1))
            l = (l / accum_steps) + aux_loss_coef * aux_loss

            scaler.scale(l).backward()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == labels).sum().item()
                total = labels.numel()
                batch_loss = l.item() * accum_steps
            total_loss += batch_loss * total
            total_tokens += total
            train_acc_sum += correct
            train_acc_count += total
            micro_batch_idx += 1

            if micro_batch_idx % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                avg_train_acc = train_acc_sum / train_acc_count if train_acc_count > 0 else 0

                if test_iter is not None and step % 100 == 0:
                    pbar.set_description("Testing...")
                    test_acc_cache = evaluate_gpt(net, test_iter, devices)
                    pbar.set_description("Training")

                if step % 10 == 0 or step == num_steps:
                    update_plot(step, avg_loss, avg_train_acc, test_acc_cache,
                                history, line_loss, line_train_acc, line_test_acc,
                                axes, plot_style, num_steps, ckpt_dir=ckpt_dir)
                if step % save_every == 0 or step >= num_steps:
                    save_checkpoint(net, optimizer, scheduler, scaler,
                                    step, best_loss, total_loss, total_tokens,
                                    ckpt_dir, max_keep)
                    best_loss = save_best_model(net, avg_loss, best_loss, best_model_path)
                pbar.update(1)
                if step >= num_steps:
                    break
    pbar.close()
    wall_end = time.time()
    final_loss = total_loss / total_tokens if total_tokens > 0 else 0
    final_acc = train_acc_sum / train_acc_count if train_acc_count > 0 else 0
    print(f'训练完成！平均损失 = {final_loss:.4f}, 平均训练准确率 = {final_acc:.4f}')
    print(f'训练速度: {total_tokens / (wall_end - wall_start):.1f} tokens/sec on {str(devices)}')
    print(f'最佳模型保存在: {best_model_path}')


def predict_motex(net, prompt, max_new_tokens, device,
                  bos_token_id=None, eos_token_id=None):
    """自回归生成（模型需返回 (logits, state, aux_loss)，state 为每层 KV 缓存）"""
    net.eval()
    prompt_ids = ds_token.encode(prompt)
    if bos_token_id is not None:
        bos_id = bos_token_id[0] if isinstance(bos_token_id, (list, tuple)) else bos_token_id
        input_ids = torch.tensor([[bos_id] + prompt_ids], device=device)
    else:
        input_ids = torch.tensor([prompt_ids], device=device)

    num_layers = net.decoder.num_layers
    # [修复] 分配两个缓存槽：state[0] 供各层 QKV 软注意力，state[1] 供 Motex_v2_1/v2_2
    #       混合注意力的稀疏硬通路（其投影与软通路不共享，需独立缓存）。
    #       对 v1/v2（无硬通路）第二个槽不会被使用，无副作用。
    state = [[None] * num_layers, [None] * num_layers]
    generated_ids = []

    eos_id = None
    if eos_token_id is not None:
        eos_id = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id

    with torch.no_grad():
        logits, state, _ = net(input_ids, valid_lens=None, state=state)
        next_token_id = logits[0, -1, :].argmax(dim=-1).item()
        for _ in range(max_new_tokens):
            if eos_id is not None and next_token_id == eos_id:
                break
            generated_ids.append(next_token_id)
            input_ids = torch.tensor([[next_token_id]], device=device)
            logits, state, _ = net(input_ids, valid_lens=None, state=state)
            next_token_id = logits[0, -1, :].argmax(dim=-1).item()

    output_ids = prompt_ids + generated_ids
    return ds_token.decode(output_ids, skip_special_tokens=True)
