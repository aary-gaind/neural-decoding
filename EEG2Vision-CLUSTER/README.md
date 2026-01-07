# EEG–CLIP Concept-Level Alignment from Single-Trial EEG

## Motivation
EEG signals are noisy, low-SNR, and ambiguous at the single-trial level.  
For natural image decoding, strict instance-level EEG → image matching is often too brittle.

Instead, this project targets concept-level alignment:
- EEG embeddings should cluster according to semantic concepts
- Retrieval should favor the correct concept even when the exact visual instance is ambiguous

Key design decision:
- Use CLIP as a frozen visual backbone
- Learn only the EEG → embedding mapping
- Add a lightweight CLIP projection head to adapt visual geometry for EEG alignment

---

## Dataset
We use **THINGS-EEG** (single-subject runs for this configuration).

- EEG: 63 channels × 250 timepoints per trial (preprocessed, 250 Hz)
- Labels: concept IDs in `[0, N_concepts-1]` (≈1654 concepts)
- Images: multiple image instances per concept (typically ~10)
- Training: single-trial EEG
- Evaluation: held-out single trials

> Note: Many published THINGS-EEG results rely on trial averaging at test time.
> This project focuses on single-trial decoding unless explicitly stated otherwise.

---

## Method

### Data Organization
Concept folders follow:
{concept_index}_{concept_name}


Example:


0085_banana/


Constructed mappings:
- `concept_to_images`: concept_id → image paths
- `concept_id_to_name`: concept_id → concept name
- `image_list`: aligned to EEG trials for instance-level baselines

For concept-level learning, CLIP feature banks are built:
- `concept_to_clip512_gpu`: concept_id → (M_i, 512) CLIP features
- Optional: `Z_proto128`: (N_concepts, 128) concept prototypes

---

## EEG Encoder Architecture

Input:
- `x ∈ (B, C, T)` where `C = 63`, `T = 250`

### ChannelTimeMLP
- Reshape `(B, C, T) → (B*C, T)`
- MLP: `T → 128 → D` (GELU)
- Reshape back `(B, C, D)`

### Channel Self-Attention
- Multi-head self-attention across channel tokens
- Residual connection + LayerNorm + FFN

### Temporal-Spatial Convolution
- Depthwise temporal convolution
- Pointwise convolution
- LayerNorm

### Flatten + Projection
- Flatten `(B, C, D) → (B, C·D)`
- Linear projection to 128
- LayerNorm

Output:
- `z_eeg ∈ (B, 128)`

---

## Visual Targets (CLIP)

- Frozen CLIP image encoder
- `encode_image(image) → (512,)`
- Unit-normalized embeddings

Trainable projection:
- `Linear(512 → 128)` or small MLP
- Output normalized to unit length

Purpose:
- Reduce dimensionality
- Adapt CLIP geometry for EEG alignment

---

## Training Objective

### Baseline: Instance-Level InfoNCE
Diagonal InfoNCE using paired EEG–image samples.
Used as a baseline; sensitive to image instance ambiguity.

### Multi-Positive Concept-Level InfoNCE
For EEG sample `i` with label `y_i`:
- Positives: all images belonging to concept `y_i`
- Negatives: images from other concepts (sampled)

Loss:
L_i = -[logsumexp(sim(z_i, positives)/τ)
- logsumexp(sim(z_i, all)/τ)]

Encourages:
- Alignment with any valid instance of the concept
- Separation from other concepts

---

## Negative Sampling
To reduce computation:
- Sample a fixed number of negative concepts per batch
- Include all images from those concepts
- Combine with positives present in the batch

Candidate pool size:
M ≈ positives + (neg_concepts × images_per_concept)

## Mixed Precision Notes
CLIP outputs float16 on GPU by default.
To avoid dtype mismatch:

z = clip_model.encode_image(image).float()
Z = clip_proj(Z)

---

## Evaluation

### Concept Retrieval
- Build concept prototypes from projected CLIP embeddings
- Compute EEG embeddings for test trials
- Similarity via dot product

Metrics:
- Top-1 accuracy
- Top-5 accuracy

### Rank Distribution
Compute the rank of the true concept per trial.
Left-skewed distributions indicate systematic semantic alignment.

### Shuffled Baseline
Labels are shuffled to verify non-trivial learning.

Chance levels (~1654 concepts):
- Top-1 ≈ 0.0006
- Top-5 ≈ 0.0030

---

## Results (Single Subject, Single Trial)

- Top-1: ~0.012
- Top-5: ~0.043

Shuffled baseline:
- Top-1: ~0.0006
- Top-5: ~0.0033

The model performs ~20× above chance on Top-1 retrieval.

---

## Running the Notebook

1. Environment:
   - PyTorch + CUDA (Colab recommended)
   - CLIP via OpenAI CLIP library

2. Image directory:
 /content/Image_set_Resize/train_images


3. Steps:
   - Build concept mappings
   - Cache CLIP image embeddings
   - Train EEG encoder + CLIP projection
   - Evaluate retrieval and rank statistics
