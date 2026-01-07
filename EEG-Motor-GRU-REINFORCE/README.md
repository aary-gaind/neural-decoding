# EEG-REINFORCE-Attention

Dynamic spatial attention for EEG decoding using GRU-based reinforcement learning agents with multi-scale glimpses.

---

## Overview

This project implements a **reinforcement learning (RL) framework** for **EEG signal decoding**, where a GRU-based agent learns to dynamically attend to informative spatiotemporal regions of high-dimensional EEG data. The architecture combines:

- **Multi-scale Glimpse Network** – extracts hierarchical temporal patches across channels, providing a rich feature embedding of local EEG segments.
- **GRU Core** – maintains a recurrent hidden state representing accumulated knowledge across glimpses, enabling temporal integration.
- **Location Policy Network** – predicts the next attention coordinates in normalized channel-time space, trained via **REINFORCE**.
- **Value Network** – provides a baseline for variance reduction in policy gradient training.
- **Classification Head** – predicts trial-level labels from the final hidden state.

The agent is trained to maximize a **sparse reward** obtained only at the end of each trial, encouraging **efficient exploration** of the EEG sequence.

---

## Key Contributions

1. **REINFORCE-based Spatial Attention**
   - Agent sequentially samples **glimpses** of the EEG input, focusing on informative channels and temporal windows.
   - Policy network outputs **normalized (channel, time) coordinates**, enabling differentiable location sampling via **reparameterized Normal distributions**.
   - Sparse terminal reward reduces computation while emphasizing critical decision points.

2. **Multi-scale Glimpse Extraction**
   - Hierarchical temporal windows (e.g., 7, 15, 31 timepoints) capture both **fine-grained and coarse EEG dynamics**.
   - Glimpse embeddings combine **content and location information** to inform the recurrent GRU.

3. **Actor-Critic Architecture**
   - **Policy network** (actor) decides where to attend next.
   - **Value network** (critic) estimates expected return for variance reduction.
   - Losses are decoupled:
     - Cross-entropy for classification
     - REINFORCE for policy updates
     - MSE for value baseline

4. **Temporal Integration via GRU**
   - GRU hidden state accumulates **information across glimpses**, allowing long-range temporal dependencies to influence attention and classification.

5. **Visualization & Interpretability**
   - Plots of **glimpse trajectories** across time-channel space show which EEG regions the agent considers informative.
   - Accuracy vs. number of glimpses demonstrates **efficiency of sequential attention**.

---

## Architecture Details
EEG Input (C x T)
│
▼
Multi-Scale Glimpse Extraction
│
▼
Glimpse Embedding (Content + Location)
│
▼
GRU Core (Temporal Integration)
│
┌────┴─────┐
│ │
Policy Net Value Net
│ │
Next Baseline
Location Estimation
│
▼
Classification Head
## Training Workflow

1. **Sliding Window Preprocessing**
   - EEG trials are segmented using overlapping windows (e.g., 3s windows, 0.25s stride) to generate high-dimensional inputs.
2. **Forward Pass**
   - For each glimpse:
     - Extract multi-scale patches at current location.
     - Update GRU hidden state.
     - Sample next location via policy network.
     - Store log-probabilities and value predictions.
3. **Reward Computation**
   - Sparse reward: +1 if final classification is correct, else 0.
4. **Loss Computation**
   - **Classification Loss:** Cross-entropy between predicted logits and true label.
   - **Policy Loss:** REINFORCE with baseline subtraction.
   - **Value Loss:** MSE between predicted baseline and observed reward.
5. **Optimization**
   - Separate optimizers for classifier+GRU+glimpse, location policy, and value network.
   - Gradient updates applied per component to decouple learning.

---

## Evaluation & Visualization

- **Single-Trial Glimpse Visualization**
  - Scatter plot of glimpse coordinates over channel-time space, showing agent attention patterns.
- **All-Test Glimpse Visualization**
  - Aggregate glimpse locations for all test trials, colored by class.
- **Accuracy vs Number of Glimpses**
  - Plots show efficiency: how many glimpses are required to reach high classification accuracy.

---

## Requirements

- Python 3.9+
- PyTorch
- NumPy 1.26.4
- MNE
- Matplotlib
- Scikit-learn
- SciPy

```bash
pip install torch numpy==1.26.4 mne matplotlib scikit-learn scipy
