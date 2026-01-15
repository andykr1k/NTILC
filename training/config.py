"""
Configuration for NTILC training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder training."""

    # Model architecture
    embedding_dim: int = 256
    encoder_model: str = "google/flan-t5-base"
    decoder_model: str = "google/flan-t5-base"
    pooling_strategy: str = "attention"  # "mean", "cls", "max", "attention"
    max_length: int = 128
    dropout: float = 0.1
    freeze_encoder: bool = False
    freeze_decoder: bool = False

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_steps: int = 1000
    gradient_clip: float = 0.5
    use_lr_scheduler: bool = True  # Use cosine annealing with warmup

    # Memory optimization
    use_gradient_checkpointing: bool = True  # Trade compute for memory
    # "float16" or "bfloat16" (bfloat16 more stable)
    torch_dtype: str = "bfloat16"

    # Data
    num_train_samples: int = 100000
    num_val_samples: int = 10000
    num_test_samples: int = 10000

    # Paths
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_dir: str = "./data"

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    use_wandb: bool = True
    wandb_project: str = "ntilc"
    wandb_entity: Optional[str] = "andykr1k"
    wandb_run_name: Optional[str] = None  # Auto-generated if None

    # Early stopping
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.001


@dataclass
class DataGeneratorConfig:
    """Configuration for synthetic data generation."""

    # Tool schemas to generate
    tools: list[str] = None

    # Generation parameters
    min_query_length: int = 5
    max_query_length: int = 50
    min_max_results: int = 1
    max_max_results: int = 100

    # Output format
    format_style: str = "python"  # "python" or "json"

    def __post_init__(self):
        if self.tools is None:
            self.tools = [
                "search",
                "calculate",
                "database_query",
                "send_email",
                "web_fetch",
                "file_read"
            ]
