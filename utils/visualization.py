"""
Visualization utilities for wandb logging
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_attention_heatmap(attention_weights, tokens, layer_idx=0, head_idx=0, max_tokens=50):
    """
    Create attention heatmap visualization

    Args:
        attention_weights: attention weights tensor [n_layers, batch, n_heads, seq_len, seq_len]
        tokens: list of token strings
        layer_idx: which layer to visualize
        head_idx: which head to visualize
        max_tokens: maximum number of tokens to show

    Returns:
        matplotlib figure
    """
    # Extract attention for specific layer and head
    if len(attention_weights.shape) == 5:
        attn = attention_weights[layer_idx, 0, head_idx].detach().cpu().numpy()
    else:
        attn = attention_weights[0, head_idx].detach().cpu().numpy()

    # Limit to max_tokens for readability
    seq_len = min(attn.shape[0], max_tokens)
    attn = attn[:seq_len, :seq_len]
    tokens = tokens[:seq_len]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    ax.set_title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    return fig


def create_gradient_flow_plot(named_parameters):
    """
    Create gradient flow visualization across layers

    Args:
        named_parameters: model.named_parameters()

    Returns:
        matplotlib figure
    """
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.grad is not None and p.requires_grad:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(layers))

    ax.bar(x - 0.2, ave_grads, 0.4, label='Mean Gradient', alpha=0.7)
    ax.bar(x + 0.2, max_grads, 0.4, label='Max Gradient', alpha=0.7)

    ax.set_xlabel('Layers')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow Across Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=90, fontsize=6)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()

    return fig


def create_perplexity_plot(steps, train_losses, val_losses):
    """
    Create perplexity comparison plot

    Args:
        steps: list of steps
        train_losses: list of training losses
        val_losses: list of validation losses

    Returns:
        matplotlib figure
    """
    train_ppl = [np.exp(loss) for loss in train_losses]
    val_ppl = [np.exp(loss) for loss in val_losses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(steps, train_losses, label='Train Loss', linewidth=2)
    ax1.plot(steps, val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Perplexity plot
    ax2.plot(steps, train_ppl, label='Train Perplexity', linewidth=2)
    ax2.plot(steps, val_ppl, label='Val Perplexity', linewidth=2)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def get_gradient_stats(model):
    """
    Get gradient statistics for logging

    Args:
        model: the model

    Returns:
        dict with gradient statistics
    """
    total_norm = 0.0
    num_params = 0
    grad_norms = []

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            grad_norms.append(param_norm.item())
            num_params += 1

    total_norm = total_norm ** 0.5

    return {
        'grad_norm_total': total_norm,
        'grad_norm_mean': np.mean(grad_norms) if grad_norms else 0,
        'grad_norm_max': np.max(grad_norms) if grad_norms else 0,
        'grad_norm_min': np.min(grad_norms) if grad_norms else 0,
        'num_params_with_grad': num_params
    }


def create_sample_quality_table(samples, step):
    """
    Create a table for generated samples with metrics

    Args:
        samples: list of generated text samples
        step: current training step

    Returns:
        list of [step, sample, length, unique_tokens_ratio]
    """
    table_data = []
    for i, sample in enumerate(samples):
        tokens = sample.split()
        length = len(tokens)
        unique_ratio = len(set(tokens)) / length if length > 0 else 0
        table_data.append([step, i, sample, length, f"{unique_ratio:.2f}"])

    return table_data


def create_hellaswag_analysis_table(examples, predictions, labels):
    """
    Create detailed HellaSwag analysis table

    Args:
        examples: list of HellaSwag examples (limited for performance)
        predictions: list of predicted indices
        labels: list of actual labels

    Returns:
        list of table rows
    """
    table_data = []
    for i, (example, pred, label) in enumerate(zip(examples, predictions, labels)):
        if i >= 20:  # Limit to 20 examples to avoid huge tables
            break
        correct = pred == label
        table_data.append([
            i,
            example.get('ctx', '')[:100] + '...',  # truncate context
            pred,
            label,
            '✓' if correct else '✗'
        ])

    return table_data
