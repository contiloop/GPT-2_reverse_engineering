import math
import inspect
import torch
import torch.nn as nn


def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    """
    Learning rate scheduler with warmup and cosine decay

    Args:
        it: current iteration
        max_lr: maximum learning rate
        min_lr: minimum learning rate
        warmup_steps: number of warmup steps
        max_steps: total number of steps

    Returns:
        learning rate for current iteration
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def get_optimizer(model, weight_decay, learning_rate, device_type):
    """
    Configure optimizer with weight decay groups

    Args:
        model: the model to optimize
        weight_decay: weight decay coefficient
        learning_rate: learning rate
        device_type: 'cuda' or 'cpu'

    Returns:
        configured optimizer
    """
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused
    )

    return optimizer, num_decay_params, num_nodecay_params, use_fused


def save_checkpoint(model, optimizer, step, val_loss, config, checkpoint_path):
    """
    Save training checkpoint

    Args:
        model: the model to save
        optimizer: the optimizer to save
        step: current training step
        val_loss: validation loss
        config: model configuration
        checkpoint_path: path to save checkpoint
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'config': config,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    Load training checkpoint

    Args:
        checkpoint_path: path to checkpoint
        model: model to load weights into
        optimizer: optimizer to load state into (optional)
        device: device to load checkpoint on

    Returns:
        tuple of (step, val_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['step'], checkpoint.get('val_loss', None)
