# Quick Commands

## Train Autoencoder

```
python training.train_autoencoder
```

## Train Autoencoder in Background

```
nohup python -m training.train_autoencoder > output.log &
```

## Analyze Embeddings

```
python analyze_embeddings.py --checkpoint ./checkpoints/best_model.pt --num_samples 100
```