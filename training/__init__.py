"""Training helpers exposed from the NTILC project root."""

__all__ = ["embed_texts", "load_checkpoint_bundle"]


def __getattr__(name: str):
    if name == "embed_texts":
        from .train_embedding_space import embed_texts

        return embed_texts
    if name == "load_checkpoint_bundle":
        from .train_embedding_space import load_checkpoint_bundle

        return load_checkpoint_bundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
