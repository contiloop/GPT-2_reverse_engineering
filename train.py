"""
Modularized GPT-2 Training Script
Clean, organized training loop using separate modules
"""
import os
import time
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

from models import GPT, GPTConfig
from data import DataLoaderLite
from utils import TrainingLogger, get_lr, get_optimizer, save_checkpoint, load_checkpoint
from utils.evaluation import get_most_likely_row, generate_samples
from configs import TrainingConfig
from hellaswag import render_example, iterate_examples


def setup_ddp():
    """Setup Distributed Data Parallel"""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process


def evaluate_validation(model, val_loader, device, device_type, config, ddp, ddp_world_size):
    """Run validation evaluation"""
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(config.val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / config.val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    return val_loss_accum.item()


def evaluate_hellaswag(raw_model, device, device_type, ddp, ddp_rank, ddp_world_size):
    """Run HellaSwag evaluation"""
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        if i % ddp_world_size != ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = raw_model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()

    acc_norm = num_correct_norm / num_total
    return acc_norm, num_correct_norm, num_total


def generate_text_samples(raw_model, enc, device, device_type, config, logger):
    """Generate text samples"""
    samples = generate_samples(
        raw_model,
        enc,
        "Hello, I'm a language model,",
        num_return_sequences=config.num_return_sequences,
        max_length=config.max_generation_length,
        device=device
    )
    for i, sample in enumerate(samples):
        logger.log_sample(i, sample)


def main():
    # Setup
    config = TrainingConfig()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process = setup_ddp()

    # Initialize logger with wandb
    logger = TrainingLogger(
        log_dir=config.log_dir,
        rank=ddp_rank,
        use_wandb=config.use_wandb,
        wandb_config={
            'project': config.wandb_project,
            'run_name': config.wandb_run_name,
            'entity': config.wandb_entity,
            'config': {
                # Model config
                'vocab_size': config.vocab_size,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_embd': config.n_embd,
                'block_size': config.block_size,
                # Training config
                'total_batch_size': config.total_batch_size,
                'micro_batch_size': config.micro_batch_size,
                'sequence_length': config.sequence_length,
                'max_steps': config.max_steps,
                'max_lr': config.max_lr,
                'min_lr': config.min_lr,
                'warmup_steps': config.warmup_steps,
                'weight_decay': config.weight_decay,
                'grad_clip': config.grad_clip,
                # Other
                'ddp_world_size': ddp_world_size,
                'device': device
            }
        }
    )

    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Log setup info
    logger.info(f"Using device: {device}")
    logger.info(f"DDP enabled: {ddp}, World size: {ddp_world_size}")

    # Calculate gradient accumulation steps
    B = config.micro_batch_size
    T = config.sequence_length
    assert config.total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = config.total_batch_size // (B * T * ddp_world_size)
    logger.info(f"Total batch size: {config.total_batch_size}")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")

    # Create data loaders
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                                   num_processes=ddp_world_size, split="train",
                                   data_root=config.data_root)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                                 num_processes=ddp_world_size, split="val",
                                 data_root=config.data_root)

    logger.info(f"Found {len(train_loader.shards)} train shards")
    logger.info(f"Found {len(val_loader.shards)} val shards")

    # Create model
    model_config = GPTConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size
    )
    model = GPT(model_config)
    model.to(device)
    torch.set_float32_matmul_precision('high')

    # Resume from checkpoint if specified
    resume_step = 0
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        checkpoint = torch.load(config.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        resume_step = checkpoint['step']

    # Setup DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Compile model
    if config.use_compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create optimizer
    optimizer, num_decay, num_nodecay, use_fused = get_optimizer(
        raw_model, config.weight_decay, config.max_lr, device_type
    )
    logger.info(f"Optimizer: AdamW (fused={use_fused})")
    logger.info(f"Decay params: {len([p for p in raw_model.parameters() if p.dim() >= 2])}, {num_decay:,} parameters")
    logger.info(f"No-decay params: {len([p for p in raw_model.parameters() if p.dim() < 2])}, {num_nodecay:,} parameters")

    # Resume optimizer state if checkpoint loaded
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Tokenizer for generation
    enc = tiktoken.get_encoding("gpt2")

    # Watch model with wandb (automatic gradient/parameter logging)
    if config.use_wandb and config.wandb_watch_model:
        logger.watch_model(raw_model, log_freq=100)

    # Training loop
    logger.info(f"Starting training from step {resume_step} to {config.max_steps}")

    for step in range(resume_step, config.max_steps):
        t0 = time.time()
        last_step = (step == config.max_steps - 1)

        # Validation
        if step % config.val_check_interval == 0 or last_step:
            val_loss = evaluate_validation(
                model, val_loader, device, device_type, config, ddp, ddp_world_size
            )
            logger.log_validation(step, val_loss)

        # HellaSwag evaluation
        if step % config.val_check_interval == 0 or last_step:
            acc_norm, num_correct, num_total = evaluate_hellaswag(
                raw_model, device, device_type, ddp, ddp_rank, ddp_world_size
            )
            logger.log_hellaswag(step, acc_norm, num_correct, num_total)

        # Generate samples
        if ((step > 0 and step % config.checkpoint_interval == 0) or last_step) and master_process:
            model.eval()
            generate_text_samples(raw_model, enc, device, device_type, config, logger)

        # Save checkpoint
        if ((step > 0 and step % config.checkpoint_interval == 0) or last_step) and master_process:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"{step}.pt")
            val_loss_value = val_loss if 'val_loss' in locals() else None
            save_checkpoint(raw_model, optimizer, step, val_loss_value,
                          raw_model.config, checkpoint_path)
            logger.log_checkpoint(checkpoint_path)

        # Training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update learning rate
        lr = get_lr(step, config.max_lr, config.min_lr, config.warmup_steps, config.max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        logger.log_train_step(step, loss_accum.item(), lr, norm, dt, tokens_per_sec)

    # Cleanup
    if ddp:
        destroy_process_group()

    logger.info("Training completed!")

    # Finish wandb run
    logger.finish()


if __name__ == "__main__":
    main()
