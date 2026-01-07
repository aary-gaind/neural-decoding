EEG-MI-Transformer

Brain-Controlled Movement Decoding via Multi-Branch CNN + Transformer Hybrids

Overview

This repository implements a high-resolution EEG-based motor imagery decoding pipeline using a hybrid architecture combining multi-scale convolutional branches and Transformer-based temporal attention encoders. The system is designed for BCI motor imagery (MI) classification, providing both state-of-the-art decoding accuracy and interpretability through attention maps and feature analysis.

Key features:

Multi-scale temporal-spatial convolutions to capture local and channel-wise EEG dynamics.

Depthwise spatial convolutions to exploit inter-channel correlations while preserving computational efficiency.

Transformer-based self-attention for global temporal dependencies across sliding EEG windows.

CLS token embeddings for sequence-level classification.

Branch- and channel-level interpretability via attention visualization and activation analysis.

Scrappy, research-oriented training pipeline optimized for rapid experimentation with real EEG data.

Architecture
1. EEGBranch

Temporal convolution: (1, Kc) kernels to capture short-term EEG temporal patterns.

Depthwise spatial convolution: (C, 1) kernels across channels, grouped by temporal feature.

Batch normalization + ELU activation for stability and non-linearity.

Average pooling along temporal dimension and optional padding to preserve window length.

Output: (B, Tp, F1) sequence tokens for each branch.

2. Multi-Branch Encoder

Multiple EEGBranch instances with different kernel sizes for multi-scale temporal features.

Outputs concatenated across feature dimension: (B, Tp, F1*num_branches).

Feeds concatenated tokens into Transformer encoder.

3. Transformer Encoder

Standard Transformer blocks with:

Multi-head self-attention

Feedforward layers with GELU

Layer normalization with residual connections

Positional embeddings

CLS token prepended for sequence-level representation.

Enables global temporal context modeling, critical for capturing delayed neural dynamics in MI tasks.

4. EEGClassifier

Linear layer on CLS embedding for classification.

Optional dropout for regularization.

Outputs logits for num_classes MI tasks.

Dataset & Preprocessing

Supports BCIC IV 2a dataset with .gdf and .mat files.

Steps:

Remove EOG channels.

Epoch extraction from tmin=-0.25s to tmax=4.25s.

Sliding windows (default: 3s window, 0.25s step, max 6 windows per trial) to augment temporal data.

Train/validation/test split with stratification.

Conversion to PyTorch TensorDataset.

Output shape per window: (C=22 channels, T≈768 samples).

Training

Cross-entropy loss, Adam optimizer (lr=1e-3), batch size 36.

Multi-epoch training loop with train/validation evaluation.

Supports GPU acceleration with .to(device).

Research-Level Analysis
Attention Visualization

Single-sample attention maps per head to understand temporal focus.

Class-specific average attention for each MI task, enabling identification of critical EEG segments per movement.

Branch Activation Analysis

Temporal activations per CNN branch, averaged over features.

Reveals multi-scale temporal feature utilization.

Channel Importance

Computes absolute sum of depthwise convolution weights per channel.

Identifies critical EEG electrodes contributing to classification, informing neurophysiological interpretation.

Class-Specific Attention

Aggregates attention across all samples for each class.

Provides interpretable heatmaps of neural sequence dependencies.

Supports publication-quality figures for neuroAI studies.

Key Insights

Hybrid CNN-Transformer models effectively capture both local EEG oscillatory patterns and long-range temporal dependencies.

Multi-branch architecture allows scale-adaptive feature extraction, crucial for MI decoding.

Attention-based interpretability provides mechanistic insights into EEG dynamics, enabling correlation with sensorimotor rhythms and task-specific neural substrates.

Channel importance analysis aligns with physiological knowledge of motor cortex activation.
