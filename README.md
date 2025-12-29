Oil Spill Semantic Segmentation System

This project presents an end-to-end deep learning system for pixel-level semantic segmentation of oil spill phenomena in aerial imagery, using the LADOS (Large-scale Aerial Dataset for Oil Spill Detection) dataset.

The system is designed to detect and localize multiple oil-related surface patterns in complex marine environments, supporting environmental monitoring and maritime safety applications.

1. Problem Statement

Manual inspection of aerial imagery for oil spill monitoring is time-consuming and error-prone.
This project aims to automate the process by building a deep learning–based semantic segmentation model that identifies oil-related classes at the pixel level, even in visually ambiguous scenes.

2. Dataset

Dataset: LADOS – Aerial Imagery Dataset for Oil Spill Detection

Data Source: Aerial images collected from realistic marine scenarios

Task Type: Multi-class semantic segmentation

Classes

Background

Emulsion

Oil

Oil Platform

Sheen

Ship

Dataset Characteristics

High intra-class variability

Strong class imbalance

Visual similarity between certain classes (e.g., Ship vs Oil Platform)

3. Preprocessing & Data Analysis

Image resizing to 640 × 640 (aligned with the LADOS research paper)

Data augmentation using Albumentations:

Horizontal & vertical flips

Rotation and scaling

Brightness / contrast adjustment

Gaussian blur

ImageNet normalization (required for pretrained backbones)

Pixel-wise class distribution analysis

Class weighting and weighted sampling to mitigate class imbalance

All preprocessing, dataset analysis, and visualizations are implemented in the preprocessing notebook.

4. Model Architecture
Encoder

ResNet50 pretrained on ImageNet

Used strictly as a feature extraction backbone

Decoder

Custom decoder with skip connections

Progressive upsampling using bilinear interpolation

Attention Modules (Handcrafted)

To comply with course requirements, attention mechanisms are implemented from scratch:

Channel Attention

Spatial Attention

These modules enhance feature discrimination, especially in visually ambiguous regions.

Output Layer

1×1 convolution for pixel-wise class prediction

Output resolution matches the input image size

5. Training Strategy

Loss Functions:

Cross Entropy Loss (with class weights)

Dice Loss

Optimizer: AdamW

Learning Rate Scheduler: ReduceLROnPlateau

Early Stopping: Based on validation mIoU

Evaluation Metric: Mean Intersection over Union (mIoU)

6. Evaluation Results

Validation mIoU: ≈ 0.54

Test mIoU: ≈ 0.56

Per-class IoU is reported for detailed performance analysis.

7. System Architecture (Flow-Based)

Raw Aerial Image
→ Preprocessing Pipeline
→ (Resize → Normalize → Augment)
→ Dataset Loader (Train / Validation / Test Splits)
→ Deep Learning Model (Encoder → Decoder → Attention)
→ Pixel-wise Class Prediction
→ Segmentation Mask Generation
→ Post-processing & Statistics Extraction
→ Visualization Overlay (Image + Mask)
→ Frontend / GUI / API Integration

Design Principles

Clear separation between preprocessing, training, and inference

Offline training (never triggered by the frontend)

Inference-ready pipeline for seamless GUI or API integration

Reproducible experiments using notebook-driven workflows

8. Project Structure

notebooks/
├── 01_preprocessing_and_analysis.ipynb
│ - Dataset loading
│ - Data augmentation
│ - Class distribution analysis
│ - Data visualization

├── 02_modeling_and_training.ipynb
│ - Model architecture
│ - Training loop
│ - Validation and early stopping
│ - Model checkpointing

└── 03_final_inference_and_visualization.ipynb
- Load trained model
- Test on unseen images
- Visualization and overlays

9. Team Contributions

This project was developed by a team of five members with clearly separated responsibilities:

Data Team (2 Members)

Dataset analysis

Preprocessing and augmentation

Class imbalance handling

Data visualization

Model Team (2 Members)

Model architecture design

Handcrafted attention modules

Training, evaluation, and optimization

Frontend Team (1 Member)

User interface design

Model integration for inference

Visualization of segmentation outputs

10. How to Run

Install dependencies:
pip install -r requirements.txt

Run notebooks in order:
01_preprocessing_and_analysis.ipynb
02_modeling_and_training.ipynb
03_final_inference_and_visualization.ipynb

11. Future Improvements

Improve separation between visually similar classes

Experiment with transformer-based encoders

Deploy the model using a backend API with a web-based frontend

Extend the system to panoptic segmentation

License

MIT License
