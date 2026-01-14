# VAE-GNN-DREAMER  
**Spatio-Temporal VAE + Graph Neural Networks for EEG Emotion Recognition**

This repository implements a **hybrid deep learning pipeline for EEG-based emotion recognition** using the **DREAMER dataset**.  
The architecture integrates:

- **3D Convolutional Variational Autoencoder (VAE)** for spatio-temporal EEG modeling  
- **Frequency-band Graph Neural Networks (GNNs)** built from mutual information (MI) connectivity  
- **Temporal modeling with GRU**  
- **Multi-task learning** with focal loss, triplet-center loss, and VAE reconstruction loss  

The system jointly predicts **binary valence and arousal** while learning structured latent EEG representations.

---

## Dataset

**DREAMER Dataset**
- 23 subjects  
- 18 emotion-eliciting videos per subject  
- 14 EEG channels (Emotiv EPOC)  
- Self-reported ratings: Valence, Arousal, Dominance (1–5)

### Label Processing
- Valence ≥ 3 → 1, else 0  
- Arousal ≥ 3 → 1, else 0  

A **4-class emotion label** is formed for metric learning:
emotion_class = 2 * arousal + valence

---

## EEG Preprocessing

### Channel Layout
The 14 EEG channels are embedded into a **9×9 spatial grid** matching the international 10-20 system:

AF3, F7, F3, FC5, T7, P7, O1,
O2, P8, T8, FC6, F4, F8, AF4

### Temporal Window
- Each EEG trial is truncated to the **last 1280 samples**
- Input tensor shape:

(Batch, 1, Time=1280, Height=9, Width=9)

---

## Frequency Band Graph Construction

EEG signals are filtered into **five frequency bands**:

| Band  | Frequency (Hz) |
|------|----------------|
| Delta | 1–4 |
| Theta | 4–8 |
| Alpha | 8–13 |
| Beta  | 13–30 |
| Gamma | 30–45 |

For each band:
1. **Differential Entropy (DE)** is computed per channel (node features)
2. **Mutual Information (MI)** is computed between all channel pairs
3. MI matrices are normalized and converted into graphs
4. Graphs are processed using **PyTorch Geometric**

Each EEG trial yields **5 graphs (one per band)**.

---

## Model Architecture

### 1. Spatio-Temporal VAE
- 3D convolutional encoder/decoder
- Learns latent representation `z ∈ ℝ¹²⁸`
- Preserves spatial electrode structure and temporal dynamics

### 2. GNN + GRU Branch
- GCN layers per frequency band
- Outputs node embeddings: `(Band, Node, Feature)`
- Flattened and passed through a GRU for temporal modeling

### 3. Feature Fusion
Final feature vector:
[ GRU_Output || VAE_Latent_z ]
Dimension: **256**

---

## Loss Functions

The model is trained end-to-end using a weighted sum of:

### 1. Focal Loss
- Handles class imbalance in valence/arousal prediction
- Applied independently to both labels

### 2. Triplet-Center Loss
- Enforces structured separation in 4-class emotion embedding space
- Uses learnable class centers

### 3. VAE Loss
- Mean squared reconstruction loss
- KL divergence regularization

Total loss:
L = β₁ · L_focal + β₂ · L_triplet + β₃ · L_vae

---

## Training Pipeline

1. Load EEG tensors for VAE branch
2. Load cached graph objects for GNN branch
3. Synchronize batching across:
   - VAE DataLoader
   - GNN DataLoader
   - Label DataLoader
4. Forward pass through:
   - VAE
   - GNN → GRU
   - Feature fusion
5. Backpropagate combined loss
6. Gradient clipping and NaN stabilization applied

---

## Dependencies

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- PyTorch Geometric  
- NumPy, SciPy  
- scikit-learn  
- pandas  

Install PyG dependencies:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

