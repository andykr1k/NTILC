"""
Configuration for NTILC training.

FIXED: Added anti-collapse defaults for intent embedding training.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class IntentEmbeddingConfig:
    """Configuration for intent embedding training (NEW ARCHITECTURE)."""

    # Model architecture
    intent_embedding_dim: int = 1024  # High-dimensional intent space
    projection_dim: int = 128  # Projected space for similarity
    encoder_model: str = "google/flan-t5-base"
    pooling_strategy: str = "attention"
    max_length: int = 512  # Longer for intent descriptions
    dropout: float = 0.5  # FIXED: Increased from 0.15 to prevent collapse

    # Freezing strategy
    freeze_encoder: bool = False
    freeze_encoder_layers: int = 4
    
    # Cluster configuration
    num_clusters: int = None  # None = dynamic clusters
    cluster_update_interval: int = 100  # Steps between cluster updates

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    use_lr_scheduler: bool = True
    label_smoothing: float = 0.1

    # Memory optimization
    use_gradient_checkpointing: bool = True
    torch_dtype: str = "bfloat16"
    gradient_accumulation_steps: int = 2

    # Data
    num_train_samples: int = 25000
    num_val_samples: int = 2500
    num_test_samples: int = 2500
    output_format: str = "python"
    regenerate_data: bool = True

    # Loss configuration (NEW: Circle Loss with corrected defaults)
    use_circle_loss: bool = True
    circle_loss_weight: float = 1.0
    circle_loss_margin: float = 0.4  # FIXED: Increased from 0.25 (harder target)
    circle_loss_gamma: float = 32.0  # FIXED: Decreased from 256 (more stable)
    
    # Contrastive loss (FIXED: Enabled by default to prevent collapse)
    use_contrastive_loss: bool = True
    contrastive_loss_weight: float = 0.3  # FIXED: Increased from 0.1
    contrastive_temperature: float = 0.07
    
    # Embedding regularization
    embedding_l2_weight: float = 0.001
    embedding_variance_weight: float = 0.005
    
    # ADDED: Anti-collapse measures
    embedding_noise_std: float = 0.05  # Add noise during training to prevent collapse
    query_augmentation_prob: float = 0.7  # Probability of augmenting queries

    # Paths
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_dir: str = "./data/train"

    # Logging
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    use_wandb: bool = True
    wandb_project: str = "ntilc"
    wandb_entity: Optional[str] = "andykr1k"
    # wandb_run_name: Optional[str] = "intent_embedding"

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001



    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Scheduled sampling (teacher forcing decay)
    use_scheduled_sampling: bool = True
    scheduled_sampling_start: float = 1.0
    scheduled_sampling_end: float = 0.5
    scheduled_sampling_warmup: int = 3000


@dataclass
class DataGeneratorConfig:
    """Configuration for synthetic data generation."""

    tools: List[str] = field(default_factory=lambda: [
        "search",
        "calculate",
        "database_query",
        "send_email",
        "web_fetch",
        "file_read"
    ])

    min_query_length: int = 5
    max_query_length: int = 50
    min_max_results: int = 1
    max_max_results: int = 100

    format_style: str = "python"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    compute_exact_match: bool = True
    compute_tool_accuracy: bool = True
    compute_parameter_accuracy: bool = True
    compute_embedding_stats: bool = True
    
    create_tsne_plots: bool = True
    create_confusion_matrix: bool = True
    save_examples: int = 100
    
    output_dir: str = "./output/evaluation"
