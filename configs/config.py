from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model settings
    vocab_size: int = 50304  # padded to nice number for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024

    # Training settings
    total_batch_size: int = 524288  # 2**19, ~0.5M tokens
    micro_batch_size: int = 64  # batch size per GPU
    sequence_length: int = 1024  # T
    max_steps: int = 19073

    # Learning rate settings
    max_lr: float = 6e-4
    min_lr: float = 6e-5  # max_lr * 0.1
    warmup_steps: int = 715

    # Optimization settings
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Data settings
    data_root: str = "edu_fineweb10B"

    # Logging and checkpointing
    log_dir: str = "log"
    checkpoint_dir: str = "checkpoints"
    val_check_interval: int = 250  # steps
    checkpoint_interval: int = 250  # steps
    val_loss_steps: int = 20  # number of validation steps

    # Generation settings
    num_return_sequences: int = 4
    max_generation_length: int = 32

    # Compilation settings
    use_compile: bool = True  # torch.compile

    # Resume training
    resume_from_checkpoint: str = None  # path to checkpoint

    # Weights & Biases settings
    use_wandb: bool = True  # enable wandb logging
    wandb_project: str = "nanogpt2"  # wandb project name
    wandb_run_name: str = None  # run name (None = auto-generate)
    wandb_entity: str = None  # wandb team/entity (None = personal)
    wandb_watch_model: bool = True  # watch model gradients/parameters (auto logging)

    def __post_init__(self):
        """Calculate derived values"""
        self.min_lr = self.max_lr * 0.1
