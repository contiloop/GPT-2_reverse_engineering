import logging
import os
from datetime import datetime
import numpy as np


class TrainingLogger:
    """Custom logger for training with DDP and wandb support"""

    def __init__(self, log_dir, rank=0, use_wandb=False, wandb_config=None):
        """
        Initialize logger

        Args:
            log_dir: directory to save logs
            rank: process rank (for DDP)
            use_wandb: whether to use wandb logging
            wandb_config: dict with wandb configuration
        """
        self.rank = rank
        self.is_master = (rank == 0)
        self.use_wandb = use_wandb and self.is_master

        if self.is_master:
            os.makedirs(log_dir, exist_ok=True)

            # 파일 핸들러 설정
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f"train_{timestamp}.log")
            self.log_file = log_file

            # 로거 설정
            self.logger = logging.getLogger('training')
            self.logger.setLevel(logging.INFO)
            # 기존 핸들러 제거 (중복 방지)
            self.logger.handlers.clear()

            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 파일 핸들러
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # 포맷터
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

            # 메트릭 로그 파일 (간단한 텍스트 형식)
            self.metrics_file = os.path.join(log_dir, "log.txt")
            # 기존 파일 지우기
            with open(self.metrics_file, 'w'):
                pass

        # Initialize wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb

                # Initialize wandb run
                wandb.init(
                    project=wandb_config.get('project', 'nanogpt2'),
                    name=wandb_config.get('run_name'),
                    entity=wandb_config.get('entity'),
                    config=wandb_config.get('config'),
                    resume='allow'
                )

                if self.is_master:
                    self.logger.info(f"wandb initialized: {wandb.run.name}")
            except ImportError:
                if self.is_master:
                    self.logger.warning("wandb not installed, skipping wandb logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def info(self, message):
        """Log info message (only master process)"""
        if self.is_master:
            self.logger.info(message)

    def warning(self, message):
        """Log warning message (only master process)"""
        if self.is_master:
            self.logger.warning(message)

    def error(self, message):
        """Log error message (only master process)"""
        if self.is_master:
            self.logger.error(message)

    def log_step(self, step, metric_type, value):
        """
        Log a single metric

        Args:
            step: training step
            metric_type: type of metric (e.g., 'train', 'val', 'hella')
            value: metric value
        """
        if self.is_master:
            with open(self.metrics_file, 'a') as f:
                f.write(f"{step} {metric_type} {value:.6f}\n")

    def log_train_step(self, step, loss, lr, norm, dt, tokens_per_sec):
        """
        Log training step metrics

        Args:
            step: training step
            loss: training loss
            lr: learning rate
            norm: gradient norm
            dt: time elapsed (seconds)
            tokens_per_sec: tokens processed per second
        """
        if self.is_master:
            # 텍스트 파일에 loss만 기록
            self.log_step(step, 'train', loss)

            # 콘솔/로그 파일에 상세 정보 출력
            message = (
                f"step {step:4d} | "
                f"loss: {loss:.6f} | "
                f"lr {lr:.4e} | "
                f"norm: {norm:.4f} | "
                f"dt: {dt*1000:.2f}ms | "
                f"tok/sec: {tokens_per_sec:.2f}"
            )
            self.logger.info(message)

            # wandb logging
            if self.use_wandb:
                perplexity = np.exp(loss)
                self.wandb.log({
                    'train/loss': loss,
                    'train/perplexity': perplexity,
                    'train/lr': lr,
                    'train/grad_norm': norm,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/step_time_ms': dt * 1000
                }, step=step)

    def log_validation(self, step, val_loss):
        """Log validation loss"""
        if self.is_master:
            self.log_step(step, 'val', val_loss)
            self.logger.info(f"validation loss: {val_loss:.4f}")

            # wandb logging
            if self.use_wandb:
                val_perplexity = np.exp(val_loss)
                self.wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': val_perplexity
                }, step=step)

    def log_hellaswag(self, step, accuracy, num_correct, num_total):
        """Log HellaSwag accuracy"""
        if self.is_master:
            self.log_step(step, 'hella', accuracy)
            self.logger.info(f"HellaSwag accuracy: {num_correct}/{num_total}={accuracy:.4f}")

            # wandb logging
            if self.use_wandb:
                self.wandb.log({
                    'eval/hellaswag_accuracy': accuracy,
                    'eval/hellaswag_correct': num_correct,
                    'eval/hellaswag_total': num_total
                }, step=step)

    def log_sample(self, sample_idx, text):
        """Log generated sample"""
        if self.is_master:
            self.logger.info(f"sample {sample_idx}: {text}")

    def log_checkpoint(self, checkpoint_path):
        """Log checkpoint save"""
        if self.is_master:
            self.logger.info(f"checkpoint saved at {checkpoint_path}")

    def watch_model(self, model, log_freq=100):
        """
        Watch model parameters and gradients (automatic logging)

        Args:
            model: the model to watch
            log_freq: log frequency
        """
        if self.is_master and self.use_wandb:
            # This automatically logs:
            # - Model graph/topology
            # - Gradient histograms
            # - Parameter histograms
            self.wandb.watch(model, log="all", log_freq=log_freq)
            self.logger.info(f"wandb watching model (log_freq={log_freq})")

    def finish(self):
        """Finish wandb run"""
        if self.use_wandb:
            self.wandb.finish()
            if self.is_master:
                self.logger.info("wandb run finished")
