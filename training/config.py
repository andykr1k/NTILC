"""
Configuration for NTILC training.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder training."""

    # Model architecture
    embedding_dim: int = 256  # Reduced from 512 - better for 6 tools
    encoder_model: str = "google/flan-t5-base"
    decoder_model: str = "google/flan-t5-base"
    pooling_strategy: str = "attention"  # "mean", "cls", "max", "attention"
    max_length: int = 256  # Increased for longer tool calls
    dropout: float = 0.15  # Increased from 0.1

    # CRITICAL: Don't freeze - need to train for good embeddings
    freeze_encoder: bool = False  # Changed from True
    freeze_decoder: bool = False  # Changed from True
    
    # Partial freezing - freeze early layers, train later ones
    freeze_encoder_layers: int = 4  # Freeze first N encoder layers (T5-base has 12)
    freeze_decoder_layers: int = 4  # Freeze first N decoder layers

    # Training hyperparameters
    batch_size: int = 32  # Reduced from 64 for stability
    learning_rate: float = 5e-5  # Increased from 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 30  # Increased from 10
    warmup_steps: int = 1000  # Reduced from 1000
    warmup_ratio: float = 0.1  # Alternative: warmup as ratio of total steps
    gradient_clip: float = 1.0  # Increased from 0.5
    use_lr_scheduler: bool = True
    label_smoothing: float = 0.1  # Added: prevent overconfident predictions

    # Memory optimization
    use_gradient_checkpointing: bool = True
    torch_dtype: str = "bfloat16"
    gradient_accumulation_steps: int = 2  # Effective batch size = 32 * 2 = 64

    # Data
    num_train_samples: int = 1000000  # Reduced for faster iteration
    num_val_samples: int = 10000
    num_test_samples: int = 10000
    output_format: str = "python"  # "json" or "python" - use JSON for consistency
    regenerate_data: bool = True  # Force regeneration with new format

    # Loss configuration
    use_validity_loss: bool = False  # Can enable later
    validity_loss_weight: float = 0.1
    
    # Contrastive loss for embedding diversity
    use_contrastive_loss: bool = True  # NEW: prevent embedding collapse
    contrastive_loss_weight: float = 0.1
    contrastive_margin: float = 0.5
    contrastive_temperature: float = 0.07  # For InfoNCE-style loss
    
    # Embedding regularization
    embedding_l2_weight: float = 0.001  # L2 regularization on embeddings
    embedding_variance_weight: float = 0.01  # Encourage embedding diversity

    # Paths
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_dir: str = "./data/train"

    # Logging
    log_interval: int = 50  # Log more frequently
    eval_interval: int = 500
    save_interval: int = 1000
    use_wandb: bool = True
    wandb_project: str = "ntilc"
    wandb_entity: Optional[str] = "andykr1k"
    wandb_run_name: Optional[str] = None

    # Early stopping
    early_stopping_patience: int = 10  # Increased patience
    early_stopping_min_delta: float = 0.001
    
    # Scheduled sampling (teacher forcing decay)
    use_scheduled_sampling: bool = True
    scheduled_sampling_start: float = 1.0  # Start with full teacher forcing
    scheduled_sampling_end: float = 0.5  # End at 50% teacher forcing
    scheduled_sampling_warmup: int = 1000  # Steps before starting decay


@dataclass
class LLMIntegrationConfig:
    """Configuration for Phase 2: LLM integration training."""
    
    # Model architecture
    base_model: str = "google/flan-t5-base"  # Base LLM for NL understanding
    embedding_dim: int = 256  # Must match autoencoder
    autoencoder_checkpoint: str = "./checkpoints/best_model.pt"
    
    # Freeze autoencoder during LLM training
    freeze_autoencoder: bool = True
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # Loss configuration
    embedding_loss_weight: float = 1.0  # MSE loss weight
    auxiliary_tool_cls_weight: float = 0.1  # Auxiliary tool classification loss
    
    # Data
    num_train_samples: int = 1000000
    num_val_samples: int = 10000
    num_test_samples: int = 10000
    data_dir: str = "./data/nl_tool_pairs"
    output_format: str = "python"
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    torch_dtype: str = "bfloat16"
    gradient_accumulation_steps: int = 2
    
    # Paths
    output_dir: str = "./checkpoints/llm_integration"
    log_dir: str = "./logs/llm_integration"
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 500
    use_wandb: bool = True
    wandb_project: str = "ntilc"
    wandb_entity: Optional[str] = "andykr1k"
    wandb_run_name: Optional[str] = None
    
    # Early stopping
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.001


@dataclass
class DataGeneratorConfig:
    """Configuration for synthetic data generation."""

    # Tool schemas to generate
    tools: List[str] = field(default_factory=lambda: [
        "search",
        "calculate",
        "database_query",
        "send_email",
        "web_fetch",
        "file_read"
    ])

    # Generation parameters
    min_query_length: int = 5
    max_query_length: int = 50
    min_max_results: int = 1
    max_max_results: int = 100

    # Output format
    format_style: str = "json"  # Changed from "python" to "json"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Evaluation metrics
    compute_exact_match: bool = True
    compute_tool_accuracy: bool = True
    compute_parameter_accuracy: bool = True
    compute_embedding_stats: bool = True
    
    # Visualization
    create_tsne_plots: bool = True
    create_confusion_matrix: bool = True
    save_examples: int = 100  # Number of examples to save
    
    # Output
    output_dir: str = "./output/evaluation"
