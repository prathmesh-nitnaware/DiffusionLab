# DiffusionLab Milestone: Classifier-Free Guidance (CFG) Experiment Report

## 1. Executive Summary

This report documents the implementation, verification, and empirical evaluation of **Classifier-Free Guidance (CFG)** for class-conditional diffusion models within DiffusionLab.

Classifier-Free Guidance enables class-conditional generation without requiring an auxiliary classifier network during sampling. By training a single model jointly on conditional and unconditional noise prediction via random conditioning dropout, we steer reverse diffusion trajectories using:

$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$

where $s \ge 0$ is the guidance scale, and $\emptyset$ denotes the learned null token.

---

## 2. Architecture & Mathematical Implementation

### 2.1 Conditioning & Null Token Representation
- **Classes**: 10 Fashion-MNIST classes (0–9: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Null Label Index**: Configured as $y_{\text{null}} = \text{num\_classes} = 10$.
- **Embedding Layer**: `nn.Embedding(num_classes + 1, time_emb_dim)`.
- **Condition Injection**:
  $$\text{cond\_emb} = \text{timestep\_mlp}(t) + \text{label\_emb}(y)$$
  Injected directly into ResNet block residual representations with zero structural changes to spatial convolutions.
- **Backward Compatibility**: If `num_classes` is `None` (or omitted), `label_emb` is disabled, preserving exact unconditional U-Net behavior and loading existing unconditional checkpoints seamlessly.

### 2.2 Training-Time Condition Dropout
- During training, each sample's class label $y_i$ is replaced with $y_{\text{null}}$ with probability $p_{\text{uncond}} = 0.10$:
  $$y_i \leftarrow \begin{cases} y_{\text{null}} & \text{with probability } p_{\text{uncond}} \\ y_i & \text{with probability } 1 - p_{\text{uncond}} \end{cases}$$

### 2.3 Batched CFG Noise Prediction
To eliminate model call overhead during sampling, conditional and unconditional predictions are computed simultaneously in a single concatenated forward pass:
```python
x_in = torch.cat([x, x], dim=0)
t_in = torch.cat([timesteps, timesteps], dim=0)
y_in = torch.cat([y, torch.full_like(y, self.null_label)], dim=0)
pred_all = self(x_in, t_in, y=y_in)
pred_cond, pred_uncond = pred_all.chunk(2, dim=0)
guided_eps = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

---

## 3. Experimental Protocol & Training Runs

### 3.1 Smoke Test Verification
- **Configuration**: `configs/smoke_test_conditional.yaml`
- **Purpose**: Verify forward pass, gradient flow, loss backpropagation, optimizer step, checkpointing, and conditional sampling on 1 batch.
- **Result**: Successfully completed in 0.20s; initial step loss = 1.0648, validation loss = 0.8538.

### 3.2 Conditional Baseline Training Run
- **Configuration**: `configs/fashion_mnist_conditional.yaml`
- **Dataset**: Fashion-MNIST (10 classes, subset of 2,000 samples, batch size 64)
- **Schedule**: Cosine beta schedule, $T = 200$ timesteps
- **Dropout Probability**: $p_{\text{uncond}} = 0.10$
- **Optimizer**: AdamW ($lr = 10^{-3}$, weight decay $= 10^{-4}$) with CosineAnnealingLR
- **Model Size**: 164,689 parameters (vs. 163,985 parameters for unconditional model, +704 parameters for label embeddings)

#### Recorded Training & Validation Loss Progression:
| Epoch | Step | Train MSE Loss | Validation MSE Loss | Epoch Duration |
|:-----:|:----:|:--------------:|:-------------------:|:--------------:|
| 1 | 28 | 0.3606 | 0.2573 | 11.95s |
| 2 | 56 | 0.1857 | 0.1570 | 11.44s |
| 3 | 84 | 0.1529 | 0.1494 | 10.66s |
| 4 | 112 | 0.1490 | **0.1223** (Best) | 10.91s |
| 5 | 140 | 0.1388 | 0.1509 | 10.68s |

- **Best Checkpoint**: `runs/fashion_mnist_conditional/checkpoints/best_model.pt` (Epoch 4, Validation MSE 0.1223).
- **Latest Checkpoint**: `runs/fashion_mnist_conditional/checkpoints/latest_model.pt`.

---

## 4. Sampling & Guidance Scale Analysis

### 4.1 Multi-Scale Guidance Evaluation
Using identical initial Gaussian latent noise $x_T \sim \mathcal{N}(0, I)$, samples were generated across 4 guidance scales: $s \in [0.0, 1.0, 2.0, 4.0]$:

- **$s = 0.0$ (Pure Unconditional)**: Samples ignore class condition $c$ and sample from the marginal distribution $p(x)$ using null-token embeddings. High diversity, lower class sharpness.
- **$s = 1.0$ (Standard Conditional, No Guidance Extrapolation)**: Standard conditional sampling without pushing away from unconditional noise. Coherent clothing shapes corresponding to target classes.
- **$s = 2.0$ (Recommended Guidance)**: Balanced sample sharpness and class definition. Features unique to specific garments (e.g., shoe soles, dress silhouettes, bag handles) become distinctly pronounced.
- **$s = 4.0$ (Aggressive Guidance)**: Extreme contrast and structural saturation. Features align strongly with class prototypes with reduced within-class variance.

Comparison grid saved to: `reports/experiments/exp_d_classifier_free_guidance/cfg_scale_comparison.png`.

### 4.2 Class-Specific Generation
Full 16-sample ($4 \times 4$) grids were generated using DDIM (25 steps, $\eta = 0.0, s = 2.0$) for all 10 Fashion-MNIST classes:
- Class 0: `class_00_t_shirt.png`
- Class 1: `class_01_trouser.png`
- Class 2: `class_02_pullover.png`
- Class 3: `class_03_dress.png`
- Class 4: `class_04_coat.png`
- Class 5: `class_05_sandal.png`
- Class 6: `class_06_shirt.png`
- Class 7: `class_07_sneaker.png`
- Class 8: `class_08_bag.png`
- Class 9: `class_09_ankle_boot.png`

---

## 5. Verification & Backward Compatibility Suite

All 36 unit and integration tests passed:
- `test_unet_unconditional_backward_compatibility`: PASS
- `test_unet_conditional_initialization_and_label_validation`: PASS
- `test_condition_dropout_logic`: PASS
- `test_batched_forward_with_cfg_mathematics`: PASS
- `test_samplers_with_cfg`: PASS
- `test_checkpoint_compatibility_loader`: PASS
- Existing test suites (schedules, forward process, samplers, UNet, training, checkpointing): 30/30 PASSED
- Ruff code formatting and linting: ALL CHECKS PASSED.
