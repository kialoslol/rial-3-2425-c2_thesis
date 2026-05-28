"""
Visualization utilities for defect detection model
- Training history plots
- Segmentation prediction visualization
- Confusion matrix
- Metric comparisons
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import cv2


class DefectVisualization:
    """Comprehensive visualization tools"""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define defect colors for visualization
        self.defect_colors = {
            0: (0, 0, 0),      # Background (black)
            1: (255, 0, 0),    # Cracks (red)
            2: (0, 255, 0),    # Spalls (green)
            3: (0, 0, 255)     # Moisture (blue)
        }
        
        sns.set_style("whitegrid")
    
    def plot_training_history(
        self,
        history: Dict[str, List],
        save_path: str = None
    ):
        """Plot training and validation metrics over epochs"""
        
        metrics = ['loss', 'accuracy', 'iou', 'dice', 'f1_score', 'map']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Extract metric from history
            train_values = [h.get(metric, 0) for h in history['train']]
            val_values = [h.get(metric, 0) for h in history['val']]
            
            epochs = range(1, len(train_values) + 1)
            
            ax.plot(epochs, train_values, 'o-', label='Train', linewidth=2, markersize=4)
            ax.plot(epochs, val_values, 's-', label='Val', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} Over Epochs', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / 'training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {save_path}")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str] = None,
        save_path: str = None
    ):
        """Plot confusion matrix heatmap"""
        
        if class_names is None:
            class_names = ['Background', 'Cracks', 'Spalls', 'Moisture']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(
            cm_normalized,
            annot=confusion_matrix,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_segmentation_results(
        self,
        image: np.ndarray,
        ground_truth_mask: np.ndarray,
        predicted_mask: np.ndarray,
        title: str = "Segmentation Results",
        save_path: str = None
    ):
        """Plot original image, GT mask, and predicted mask side-by-side"""
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth mask
        gt_colored = self._colorize_mask(ground_truth_mask)
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Predicted mask
        pred_colored = self._colorize_mask(predicted_mask)
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Overlay prediction on image
        overlay = (0.6 * image.astype(np.uint8) + 0.4 * pred_colored).astype(np.uint8)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / f'{title.replace(" ", "_").lower()}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Segmentation results saved to {save_path}")
    
    def plot_batch_predictions(
        self,
        images: np.ndarray,
        ground_truths: np.ndarray,
        predictions: np.ndarray,
        indices: List[int] = None,
        save_path: str = None
    ):
        """Plot batch of predictions"""
        
        batch_size = images.shape[0]
        if indices is None:
            indices = range(min(4, batch_size))
        
        n_samples = len(indices)
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for row, idx in enumerate(indices):
            # Original
            axes[row, 0].imshow(images[idx].astype(np.uint8))
            axes[row, 0].set_title(f'Image {idx}', fontsize=10, fontweight='bold')
            axes[row, 0].axis('off')
            
            # GT
            gt_colored = self._colorize_mask(ground_truths[idx])
            axes[row, 1].imshow(gt_colored)
            axes[row, 1].set_title(f'GT {idx}', fontsize=10, fontweight='bold')
            axes[row, 1].axis('off')
            
            # Prediction
            pred_colored = self._colorize_mask(predictions[idx])
            axes[row, 2].imshow(pred_colored)
            axes[row, 2].set_title(f'Pred {idx}', fontsize=10, fontweight='bold')
            axes[row, 2].axis('off')
            
            # Overlay
            overlay = (0.6 * images[idx].astype(np.uint8) + 0.4 * pred_colored).astype(np.uint8)
            axes[row, 3].imshow(overlay)
            axes[row, 3].set_title(f'Overlay {idx}', fontsize=10, fontweight='bold')
            axes[row, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / 'batch_predictions.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Batch predictions saved to {save_path}")
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, float],
        save_path: str = None
    ):
        """Plot bar chart of all metrics"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / 'metrics_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics comparison saved to {save_path}")
    
    def plot_defect_distribution(
        self,
        masks: List[np.ndarray],
        class_names: List[str] = None,
        save_path: str = None
    ):
        """Plot distribution of defect types in dataset"""
        
        if class_names is None:
            class_names = ['Background', 'Cracks', 'Spalls', 'Moisture']
        
        # Count pixels per class
        class_counts = np.zeros(len(class_names))
        
        for mask in masks:
            unique, counts = np.unique(mask.flatten(), return_counts=True)
            for u, c in zip(unique, counts):
                if u < len(class_names):
                    class_counts[u] += c
        
        # Normalize
        class_percentages = class_counts / class_counts.sum() * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['black', 'red', 'green', 'blue'][:len(class_names)]
        wedges, texts, autotexts = ax.pie(
            class_percentages,
            labels=class_names,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Defect Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.output_dir / 'defect_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Defect distribution saved to {save_path}")
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert grayscale mask to RGB using class colors"""
        
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.defect_colors.items():
            colored[mask == class_id] = color
        
        return colored


# Example usage
if __name__ == "__main__":
    viz = DefectVisualization()
    
    # Example: Plot metrics
    sample_metrics = {
        'accuracy': 0.92,
        'iou': 0.87,
        'dice': 0.89,
        'f1_score': 0.88,
        'map': 0.85
    }
    viz.plot_metrics_comparison(sample_metrics)
