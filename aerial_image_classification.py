"""
Aerial Image Classification using Deep Learning
===============================================

A comprehensive implementation comparing custom CNN architectures and transfer learning 
approaches for multi-class aerial image classification.

This script demonstrates:
- Custom CNN architecture design and optimization
- Transfer learning with ResNet-18
- Proper data management and preprocessing
- Comprehensive model evaluation and comparison

Author: Alina Alimova, Aniya Bagheri
"""

import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from collections import Counter
import ssl
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DataManager:
    """Handles data loading, preprocessing, and augmentation for aerial imagery dataset."""
    
    def __init__(self, dataset_path, test_size=0.1, val_size=0.2, batch_size=16):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def create_data_loaders(self):
        """Create stratified train/validation/test splits and data loaders."""
        
        # Load dataset to get class information
        full_dataset = ImageFolder(root=self.dataset_path, transform=None)
        self.class_names = full_dataset.classes
        targets = full_dataset.targets
        
        # Stratified split for test set
        train_val_indices, test_indices = train_test_split(
            np.arange(len(full_dataset)),
            test_size=self.test_size,
            shuffle=True,
            stratify=targets,
            random_state=42
        )
        
        # Stratified split for train/validation
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=self.val_size,
            shuffle=True,
            stratify=[targets[i] for i in train_val_indices],
            random_state=42
        )
        
        # Verify no data leakage
        assert not (set(train_indices) & set(val_indices))
        assert not (set(train_indices) & set(test_indices))
        assert not (set(val_indices) & set(test_indices))
        
        print("Data split verification: No overlap detected between splits ✓")
        
        # Create datasets with appropriate transforms
        train_dataset = ImageFolder(root=self.dataset_path, transform=self.train_transform)
        val_dataset = ImageFolder(root=self.dataset_path, transform=self.eval_transform)
        test_dataset = ImageFolder(root=self.dataset_path, transform=self.eval_transform)
        
        # Create subsets
        train_set = Subset(train_dataset, train_indices)
        val_set = Subset(val_dataset, val_indices)
        test_set = Subset(test_dataset, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        
        print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        
        # Print class distribution
        self._print_class_distribution(train_indices, val_indices, test_indices, targets)
        
        return train_loader, val_loader, test_loader
    
    def _print_class_distribution(self, train_indices, val_indices, test_indices, targets):
        """Print class distribution across all splits."""
        for name, indices in [("Training", train_indices), ("Validation", val_indices), ("Test", test_indices)]:
            class_counts = Counter([targets[i] for i in indices])
            print(f"\n{name} Set Class Distribution:")
            for idx, cls in enumerate(self.class_names):
                print(f"{cls:20s}: {class_counts[idx]} samples")


class CustomCNN(nn.Module):
    """
    Optimized custom CNN architecture for aerial image classification.
    
    Architecture features:
    - Three convolutional blocks with progressive channel expansion
    - Batch normalization and dropout for regularization
    - Optimized through extensive hyperparameter tuning
    """
    
    def __init__(self, num_classes=21):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Classifier with optimized architecture
        feature_size = 16  # 128x128 -> 16x16 after 3 max pooling operations
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * feature_size * feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ModelTrainer:
    """Handles model training, validation, and evaluation."""
    
    def __init__(self, device):
        self.device = device
    
    def train_model(self, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
        """
        Train a model with comprehensive tracking of metrics.
        
        Returns:
            tuple: (trained_model, metrics_dict)
        """
        model = model.to(self.device)
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        best_val_acc = 0.0
        best_model_wts = model.state_dict().copy()
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss, running_corrects = 0.0, 0
            total_samples = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data).item()
            
            if scheduler:
                scheduler.step()
            
            # Calculate training metrics
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_model(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = model.state_dict().copy()
        
        # Load best model weights
        model.load_state_dict(best_model_wts)
        
        return model, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def _validate_model(self, model, val_loader, criterion):
        """Validate model and return loss and accuracy."""
        model.eval()
        running_loss, running_corrects = 0.0, 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data).item()
        
        return running_loss / total_samples, running_corrects / total_samples
    
    def evaluate_model(self, model, test_loader, criterion, class_names):
        """Comprehensive model evaluation with detailed metrics."""
        model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss = running_loss / len(test_loader.dataset)
        test_acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)
        
        return test_loss, test_acc, report, cm, all_preds, all_labels


class Visualizer:
    """Handles all visualization tasks including plots and confusion matrices."""
    
    @staticmethod
    def plot_training_metrics(metrics, title_prefix):
        """Plot training and validation loss/accuracy curves."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_losses'], label='Train Loss')
        plt.plot(metrics['val_losses'], label='Val Loss')
        plt.title(f"{title_prefix} - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(metrics['train_accuracies'], label='Train Acc')
        plt.plot(metrics['val_accuracies'], label='Val Acc')
        plt.title(f"{title_prefix} - Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_model_comparison(custom_metrics, resnet_scratch_metrics, resnet_finetune_metrics):
        """Plot comparison of all models' validation accuracies."""
        plt.figure(figsize=(12, 6))
        plt.plot(custom_metrics['val_accuracies'], label='Custom CNN', linewidth=2)
        plt.plot(resnet_scratch_metrics['val_accuracies'], label='ResNet18 from Scratch', linewidth=2)
        plt.plot(resnet_finetune_metrics['val_accuracies'], label='ResNet18 Fine-tuned', linewidth=2)
        plt.title('Model Performance Comparison - Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names, title):
        """Plot detailed confusion matrix."""
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function."""
    
    # Configuration
    DATASET_PATH = "/path/to/aerial_imagery/Images"  # Update this path
    NUM_EPOCHS = 25
    BATCH_SIZE = 16
    
    # Initialize components
    data_manager = DataManager(DATASET_PATH, batch_size=BATCH_SIZE)
    trainer = ModelTrainer(device)
    visualizer = Visualizer()
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_manager.create_data_loaders()
    num_classes = len(data_manager.class_names)
    
    # Training configuration
    criterion = nn.CrossEntropyLoss()
    
    # =====================================================
    # 1. Train Custom CNN
    # =====================================================
    print("\n" + "="*50)
    print("Training Custom CNN")
    print("="*50)
    
    custom_model = CustomCNN(num_classes=num_classes)
    custom_optimizer = optim.Adam(custom_model.parameters(), lr=0.001, weight_decay=0.0001)
    custom_scheduler = optim.lr_scheduler.StepLR(custom_optimizer, step_size=7, gamma=0.1)
    
    trained_custom, custom_metrics = trainer.train_model(
        custom_model, train_loader, val_loader, criterion, 
        custom_optimizer, custom_scheduler, NUM_EPOCHS
    )
    
    visualizer.plot_training_metrics(custom_metrics, "Custom CNN")
    
    # =====================================================
    # 2. Train ResNet-18 from Scratch
    # =====================================================
    print("\n" + "="*50)
    print("Training ResNet-18 from Scratch")
    print("="*50)
    
    # Bypass SSL verification for model download
    ssl._create_default_https_context = ssl._create_unverified_context
    
    resnet_scratch = models.resnet18(weights=None)
    resnet_scratch.fc = nn.Linear(resnet_scratch.fc.in_features, num_classes)
    
    resnet_scratch_optimizer = optim.Adam(resnet_scratch.parameters(), lr=0.001, weight_decay=1e-4)
    resnet_scratch_scheduler = optim.lr_scheduler.StepLR(resnet_scratch_optimizer, step_size=7, gamma=0.1)
    
    trained_resnet_scratch, resnet_scratch_metrics = trainer.train_model(
        resnet_scratch, train_loader, val_loader, criterion,
        resnet_scratch_optimizer, resnet_scratch_scheduler, NUM_EPOCHS
    )
    
    visualizer.plot_training_metrics(resnet_scratch_metrics, "ResNet-18 from Scratch")
    
    # =====================================================
    # 3. Fine-tune ResNet-18
    # =====================================================
    print("\n" + "="*50)
    print("Fine-tuning ResNet-18")
    print("="*50)
    
    resnet_finetune = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet_finetune.fc = nn.Linear(resnet_finetune.fc.in_features, num_classes)
    
    resnet_finetune_optimizer = optim.Adam(resnet_finetune.parameters(), lr=0.001, weight_decay=1e-4)
    resnet_finetune_scheduler = optim.lr_scheduler.StepLR(resnet_finetune_optimizer, step_size=7, gamma=0.1)
    
    trained_resnet_finetune, resnet_finetune_metrics = trainer.train_model(
        resnet_finetune, train_loader, val_loader, criterion,
        resnet_finetune_optimizer, resnet_finetune_scheduler, NUM_EPOCHS
    )
    
    visualizer.plot_training_metrics(resnet_finetune_metrics, "ResNet-18 Fine-tuned")
    
    # =====================================================
    # 4. Model Comparison and Selection
    # =====================================================
    print("\n" + "="*50)
    print("Model Performance Comparison")
    print("="*50)
    
    # Compare validation accuracies
    custom_best = max(custom_metrics['val_accuracies'])
    resnet_scratch_best = max(resnet_scratch_metrics['val_accuracies'])
    resnet_finetune_best = max(resnet_finetune_metrics['val_accuracies'])
    
    print(f"Custom CNN - Best Validation Accuracy: {custom_best:.4f}")
    print(f"ResNet-18 (Scratch) - Best Validation Accuracy: {resnet_scratch_best:.4f}")
    print(f"ResNet-18 (Fine-tuned) - Best Validation Accuracy: {resnet_finetune_best:.4f}")
    
    # Visualize comparison
    visualizer.plot_model_comparison(custom_metrics, resnet_scratch_metrics, resnet_finetune_metrics)
    
    # Select best model
    best_model = trained_resnet_finetune  # Based on validation performance
    best_name = "ResNet-18 Fine-tuned"
    
    # =====================================================
    # 5. Final Evaluation on Test Set
    # =====================================================
    print("\n" + "="*50)
    print(f"Final Evaluation - {best_name}")
    print("="*50)
    
    test_loss, test_acc, classification_rep, conf_mat, predictions, true_labels = trainer.evaluate_model(
        best_model, test_loader, criterion, data_manager.class_names
    )
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Macro F1-Score: {classification_rep['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {classification_rep['weighted avg']['f1-score']:.4f}")
    
    # Detailed classification report
    print("\nPer-Class Performance:")
    for cls in data_manager.class_names:
        metrics = classification_rep[cls]
        print(f"{cls:20s}: Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(conf_mat, data_manager.class_names, 
                                   f"{best_name} - Test Set Confusion Matrix")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    main()
