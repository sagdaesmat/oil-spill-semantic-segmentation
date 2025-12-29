# SAR Oil Spill Segmentation

This project focuses on semantic segmentation of oil spills in SAR (Synthetic Aperture Radar) imagery using deep learning.  
The system is designed to detect and classify different oil spill–related surface phenomena at the pixel level, enabling accurate analysis and visualization for downstream applications.

The project follows a **modular, notebook-driven pipeline** with a clear separation between preprocessing, model training, and inference, making it suitable for research, experimentation, and frontend integration.

---

## Project Objectives

- Perform pixel-wise semantic segmentation on SAR images
- Detect and classify oil spill–related classes (e.g. oil, emulsion, sheen, ship)
- Build a reproducible and well-structured ML pipeline
- Support easy integration with a GUI or frontend application
- Enable clear visualization and quantitative evaluation of results

---

## Dataset Overview

- **Input**: SAR images (Sentinel-1)
- **Labels**: Pixel-level segmentation masks
- **Classes**:
  - Background
  - Emulsion
  - Oil
  - Oil Platform
  - Sheen
  - Ship

The dataset is split into **train / validation / test** subsets and processed into a PyTorch-ready format.

---

## System Design

The system follows a modular and scalable architecture, clearly separating data processing, model training, and inference.

Training is performed offline, while inference is designed to be easily integrated into a frontend or API.

### Overall Pipeline

```mermaid
flowchart TD

A[Raw SAR Data<br/>Sentinel-1 Images]
B[Ground Truth Masks<br/>Pixel-level]

A --> C[Data Engineering & Preparation]
B --> C

C --> D[Dataset Interface<br/>Train / Val / Test Splits]

D --> E[Modeling & Training<br/>CNN + Custom Attention]

E --> F[Inference Pipeline<br/>Patch Prediction]

F --> G[GUI / Visualization<br/>Image + Mask Overlay]
Design Principles
Clear separation between preprocessing, training, and inference

Offline training (never triggered by the frontend)

Inference-ready pipeline for GUI or API integration

Reproducible experiments using organized notebooks

Modular design allowing future model upgrades

Repository Structure
text
Copy code
sar-oil-spill-segmentation/
│
├── data/
│   ├── raw/                # Original SAR images and labels (not pushed)
│   ├── processed/          # Preprocessed tensors and masks
│   └── splits/             # Train / validation / test splits
│
├── data_engineering/       # Preprocessing and dataset preparation scripts
│
├── models/                 # Model architectures and loss functions
│
├── inference/              # Inference utilities and visualization helpers
│
├── app/                    # Frontend or GUI integration (optional)
│
├── experiments/            # Training experiments and checkpoints
│
├── docs/
│   ├── system_architecture.md
│   └── data_pipeline.md
│
├── README.md
└── requirements.txt
Notebooks Overview
The project is organized around three main notebooks, each responsible for a distinct stage in the pipeline:

01_preprocessing_and_analysis.ipynb
Dataset loading

Data augmentation

Class distribution analysis

Visualization of images and masks

Validation of preprocessing correctness

02_modeling_and_training.ipynb
Model architecture definition

Loss functions and metrics

Training loop

Validation and early stopping

Model checkpointing

03_final_inference_and_visualization.ipynb
Load trained model

Inference on unseen images

Overlay visualization (image + mask)

Class-wise pixel statistics

Output preparation for frontend usage

Model Architecture
Encoder: ResNet-based CNN backbone

Decoder: U-Net–style upsampling path

Attention:

Channel Attention

Spatial Attention

Loss:

Cross-Entropy Loss

Dice Loss

Evaluation Metric:

Mean Intersection over Union (mIoU)

Per-class IoU

Team Contributions
The project is developed by a team of five members with clearly defined roles:

Data Engineering (2 members)
Dataset preparation, preprocessing, augmentation, and analysis

Modeling & Training (2 members)
Model architecture design, training, evaluation, and optimization

Frontend (1 member)
GUI development and visualization integration

Each module is developed independently to allow parallel work and clean integration.

Usage Notes
Training is performed offline using the training notebook

Inference can be executed independently without retraining

The final trained model can be directly loaded for visualization or frontend integration

Raw data and generated outputs are not committed to the repository

Future Work
Improve class imbalance handling for rare classes

Add post-processing for mask refinement

Deploy inference as a REST API

Integrate real-time frontend interaction

Experiment with transformer-based architectures

License
This project is intended for academic and educational use.
