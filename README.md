# Land Use Classification using Convolutional Neural Networks

## Project Overview
This repository contains our implementation of various CNN architectures for land use classification using the UC Merced dataset. We compared a custom CNN against ResNet-18 (trained from scratch and fine-tuned) to identify the most effective approach for aerial image classification.

## Team Members
- Alina Alimova
- Aniya Bagheri

## Course Information
- **Course**: COMP534 - Applied AI
- **University**: University of Liverpool

## Dataset
The UC Merced Land-use dataset consists of 2,100 aerial scene images (256×256 pixels) across 21 land-use classes. Each class contains 100 images, representing diverse urban land use categories including:
- Agricultural fields
- Airplane runways
- Baseball diamonds
- Beach areas
- Buildings
- Dense residential areas
- Forest
- Harbor
- And 13 other categories

## Methodology

### Data Preprocessing
- **Resizing**: All images resized to 128×128 pixels
- **Normalization**: Using ImageNet mean and standard deviation ([0.485, 0.456, 0.406] and [0.229, 0.224, 0.225])
- **Data Augmentation**: Random horizontal flipping and ±10° rotation (training set only)
- **Data Split**: 72% training (1,512 images), 18% validation (378 images), 10% test (210 images) with stratified sampling

### Experimental Protocol
We implemented a stratified train-validation-test split approach to ensure representative distribution across all 21 land-use classes:
- Stratified sampling maintained class balance across all splits (72 samples per class in training, 18 in validation, 10 in test)
- Verification of data leakage was conducted to ensure no overlap between splits

### Model Architectures

1. **Custom CNN**:
   - 3 convolutional blocks with increasing channels [64→128→256]
   - ReLU activation and MaxPool operations after each convolutional layer
   - 256-unit fully connected layer with 0.3 dropout
   - Final output layer with 21 units (one per class)

2. **ResNet-18**:
   - Implementation 1: Trained from scratch with random initialization
   - Implementation 2: Fine-tuned with ImageNet pre-trained weights

### Training Details
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: Learning rate reduction by factor of 0.1 every 7 epochs
- **Batch Size**: 16
- **Epochs**: 25

## Results

| Model | Strategy | Best Validation Accuracy | Epochs to Best | Final Training Loss |
|-------|----------|--------------------------|----------------|---------------------|
| Custom CNN | From Scratch | 71.69% | 21 | 0.4579 |
| ResNet-18 | From Scratch | 82.80% | 16 | 0.3553 |
| ResNet-18 | Fine-tuned | 97.35% | 20 | 0.0435 |

### Final Model Performance (ResNet-18 Fine-tuned)
- **Test Accuracy**: 95.24%
- **Macro F1-Score**: 0.9508
- **Weighted F1-Score**: 0.9524
- **Classes with Perfect Classification**: 10 out of 21

### Class-Specific Performance
The fine-tuned ResNet-18 model achieved perfect classification (F1-score = 1.0) for 10 out of 21 classes with distinctive visual patterns, while showing strong but lower performance (F1-scores 0.82-0.95) on more challenging classes:

- Most challenging classes were residential areas and golf courses, where visual similarities caused occasional misclassifications
- Sparse residential areas showed high precision but low recall
- Medium residential areas showed lower precision but high recall

## Key Findings
- Transfer learning with fine-tuning significantly outperformed training from scratch
- ResNet-18's residual connections likely contributed to stronger performance over our simpler custom CNN
- The model struggled most with visually similar classes (different residential densities)
- Data augmentation and proper stratification were crucial for achieving balanced performance across classes

## Repository Structure
```
COMP534-Assignment2/
├── comp534_a2.py         # Main implementation code
├── images/               # Results visualizations
│   ├── validation_accuracy_comparison.png
│   ├── training_accuracy_comparison.png
│   └── confusion_matrix.png
└── documentation/        # Detailed PDF report
    └── comp534_a2.pdf    # Original assignment report
```

## Requirements
```
numpy
matplotlib
torch
torchvision
scikit-learn
seaborn
tqdm
Pillow
```

## License
© 2024 Alina Alimova and Aniya Bagheri. All rights reserved.

This repository contains coursework completed for COMP534 - Applied AI at the University of Liverpool.
Unauthorized copying, reproduction, or reuse is prohibited without permission from the authors.
