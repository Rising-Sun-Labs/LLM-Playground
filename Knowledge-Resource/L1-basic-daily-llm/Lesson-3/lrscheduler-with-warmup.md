### LR scheduler:

```
LR = Learning
An LR scheduler is a technique that changes the learning rate during training rather than keeping it constant

1. High LR at the start helps the model learn faster
2. Lower LR later helps the model converge smoothly
3. Schedulers can help escape bad local minima and prevent overfitting

## Common schedulers in PyTorch
1. StepLR               -> drops LR every N epochs
2. ExponentialLR        -> decays LR exponentially
3. CosineAnnealingLR    -> smooth cosine-shaped decay (popular in transofrmers)
4. ReduceLROnPlateau    -> lowers LR if loss stops improving

# Example
Epoch  1: lr = 0.001
Epoch 10: lr = 0.0005
Epoch 20: lr = 0.0001

```

### Warmup:

```
warmup is when you start training with a small learning rate, and gradually increase it to the base LR over the first few steps or epochs

# Why warmup?
1. In the first steps, weights are random -> large LR can blow up gradients
2. Gradually increasing LR lets the optimizer stabilize
3. Improves convergence and training stability (especially for large models)

LR = 1e-3 and warmup steps=500:

# Example
Step   0: lr = 0.0
Step 100: lr = 0.0002
Step 250: lr = 0.0005
Step 500: lr = 0.001
Step 800: lr = 0.0008  (scheduler starts decaying after warmup)

```
