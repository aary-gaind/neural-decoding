EEG-GazeCL

Contrastive Spatial Decoding of Gaze from EEG

EEG-GazeCL is a deep learning framework for decoding spatial gaze targets from EEG signals.
It learns spatially consistent embeddings using a Vision-Transformer-style EEG encoder and distance-aware contrastive learning.

Evaluated on the EEGEyeNet dataset, EEG-GazeCL achieves improved spatial accuracy, reduced catastrophic errors, and smoother gaze predictions compared to standard classification baselines.

🔍 Key Ideas

EEG gaze decoding is inherently spatial, not just categorical.

Standard cross-entropy treats all errors equally — spatial contrastive learning differentiates nearby vs. far errors.

Nearby gaze targets → nearby embeddings; far targets → well-separated embeddings.

Regularization is applied early in training, then annealed.

🧠 Model Overview

Input: EEG trials of shape (128 channels × 500 time samples)

Backbone: EEG Vision Transformer (EEGViT)

Temporal convolution → creates time patches

Spatial convolution → groups channels

Transformer encoder with CLS token

Outputs:

Gaze class (discrete dot)

Latent embedding (used for spatial regularization)

📐 Spatial Contrastive Learning

We introduce a distance-aware spatial contrastive loss:

∥
𝑧
𝑖
−
𝑧
𝑗
∥
∝
∥
𝑔
𝑖
−
𝑔
𝑗
∥
∥z
i
	​

−z
j
	​

∥∝∥g
i
	​

−g
j
	​

∥

Where:

𝑧
𝑖
z
i
	​

 = EEG embedding

𝑔
𝑖
g
i
	​

 = 2D gaze position

Key design choices:

Contrastive loss applied only in early epochs (annealing)

Gaussian distance weighting

Combined with standard cross-entropy

📊 Evaluation Metrics

Beyond standard classification accuracy, we measure:

Median pixel error

Mean pixel error

P(exact): exact dot prediction

P(within 1 step): nearest neighbor

P(within 2 steps): two spatial steps

This gives a faithful evaluation of spatial gaze decoding quality.

📁 Dataset

EEGEyeNet – Position Task (Dot Targets)

Each subject file contains:

EEG: (N, T, C) EEG trials

labels: (N, 3) → [trial_id, x, y]

Multiple subjects are stored as separate .npz files and merged during training.

🧪 Results (Single Subject)

After spatial contrastive learning + annealing:

Exact accuracy: ~45%

Within 1 step: ~54%

Within 2 steps: ~71%

Median pixel error: ~55 px

Errors are mostly local, with rare catastrophic failures.

🚀 Project Structure
EEG-GazeCL/
├── data/                # EEGEyeNet subject files
├── models/              # EEGViT and loss implementations
├── notebooks/           # Colab / Jupyter demos
├── training/            # Training scripts
├── evaluation/          # Metrics & plotting scripts
├── utils/               # Data loaders, preprocessing
└── README.md
