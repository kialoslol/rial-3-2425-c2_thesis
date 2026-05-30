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
        """Plot train vs val curves for each metric, one subplot per metric."""

        metric_config = {
            'loss':      {'label': 'Loss',     'lower_better': True},
            'accuracy':  {'label': 'Accuracy', 'lower_better': False},
            'mean_iou':  {'label': 'IoU',      'lower_better': False},
            'mean_dice': {'label': 'Dice',     'lower_better': False},
            'macro_f1':  {'label': 'F1 Score', 'lower_better': False},
            'mAP':       {'label': 'mAP',      'lower_better': False},
        }

        # Only plot metrics that actually exist in history
        available = [
            m for m in metric_config
            if any(m in h for h in history.get('train', []))
        ]

        n = len(available)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.array(axes).flatten()

        for idx, metric in enumerate(available):
            ax = axes[idx]
            cfg = metric_config[metric]

            train_vals = [h.get(metric, np.nan) for h in history['train']]
            val_vals   = [h.get(metric, np.nan) for h in history['val']]
            epochs     = list(range(1, len(train_vals) + 1))

            ax.plot(epochs, train_vals, color='#1f77b4', linewidth=2,
                    marker='o', markersize=4, label='Train')
            ax.plot(epochs, val_vals,   color='#ff7f0e', linewidth=2,
                    marker='s', markersize=4, linestyle='--', label='Val')

            # Mark best validation epoch
            val_arr = np.array(val_vals, dtype=float)
            if not np.all(np.isnan(val_arr)):
                best_ep = int(np.nanargmin(val_arr) if cfg['lower_better']
                              else np.nanargmax(val_arr)) + 1
                best_val = val_arr[best_ep - 1]
                ax.axvline(best_ep, color='gray', linestyle=':', linewidth=1, alpha=0.7)
                ax.annotate(
                    f'Best: {best_val:.4f}\n(ep {best_ep})',
                    xy=(best_ep, best_val),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=8, color='#ff7f0e',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
                )

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(cfg['label'], fontsize=11)
            ax.set_title(f'{cfg["label"]} — Train vs Val', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=1)

        # Hide unused subplots
        for idx in range(len(available), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('Training & Validation Metrics per Epoch', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path is None:
            save_path = str(self.output_dir / 'training_history.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {save_path}")
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = None,
        save_path: str = None
    ):
        """Confusion matrix heatmap with per-class precision, recall, and F1."""

        if class_names is None:
            class_names = ['Background', 'Cracks', 'Spalls', 'Moisture']

        n = cm.shape[0]

        # Row-normalize for background colour (recall per row)
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1          # avoid division by zero
        cm_norm = cm.astype(float) / row_sums

        # Build annotation: "count\n(xx.x%)" in each cell
        annot = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                pct = cm_norm[i, j] * 100
                annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"

        # Per-class metrics
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recall    = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1        = np.where((precision + recall) > 0,
                             2 * precision * recall / (precision + recall), 0.0)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 6),
            gridspec_kw={'width_ratios': [3, 1]}
        )

        # ── Left: heatmap ────────────────────────────────────────────────────
        ax_cm = axes[0]
        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt='',
            cmap='Blues',
            vmin=0, vmax=1,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': 'Row-normalised recall', 'shrink': 0.8},
            ax=ax_cm,
            annot_kws={'size': 10}
        )
        ax_cm.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax_cm.set_ylabel('True Label',      fontsize=12, fontweight='bold')
        ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax_cm.tick_params(axis='x', rotation=30)
        ax_cm.tick_params(axis='y', rotation=0)

        # ── Right: per-class bar chart of Precision / Recall / F1 ───────────
        ax_bar = axes[1]
        x      = np.arange(n)
        width  = 0.25
        ax_bar.barh(x - width, precision, width, label='Precision', color='#1f77b4')
        ax_bar.barh(x,         recall,    width, label='Recall',    color='#ff7f0e')
        ax_bar.barh(x + width, f1,        width, label='F1',        color='#2ca02c')

        ax_bar.set_yticks(x)
        ax_bar.set_yticklabels(class_names, fontsize=10)
        ax_bar.set_xlabel('Score', fontsize=11)
        ax_bar.set_title('Per-class Metrics', fontsize=12, fontweight='bold')
        ax_bar.set_xlim(0, 1.05)
        ax_bar.legend(loc='lower right', fontsize=9)
        ax_bar.grid(axis='x', alpha=0.3)

        # Value labels on bars
        for bars, vals in [(ax_bar.patches[:n], precision),
                           (ax_bar.patches[n:2*n], recall),
                           (ax_bar.patches[2*n:], f1)]:
            for bar, val in zip(bars, vals):
                ax_bar.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8
                )

        # Overall accuracy in title area
        overall_acc = np.diag(cm).sum() / cm.sum()
        fig.suptitle(
            f'Confusion Matrix Analysis    (Overall Accuracy: {overall_acc:.2%})',
            fontsize=14, fontweight='bold', y=1.02
        )

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


if __name__ == "__main__":
    import json, sys

    # ── Load real training history saved by EarlyFusionPipelineV2.train() ───
    # Default path matches where CNNVIT_v2.py saves it (checkpoint_dir)
    history_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/training_history.json"

    with open(history_path) as f:
        history = json.load(f)

    # ── Pull confusion matrix from the best val epoch ───────────────────────
    # Best epoch = highest mean_iou across all val epochs
    val_ious  = [h.get("mean_iou", 0.0) for h in history["val"]]
    best_ep   = int(np.argmax(val_ious))
    cm_raw    = history["val"][best_ep].get("confusion_matrix")

    if cm_raw is None:
        print("WARNING: confusion_matrix not found in history — retrain with the updated CNNVIT.py")
        cm = None
    else:
        cm = np.array(cm_raw, dtype=np.int64)

    print(f"Loaded {len(history['train'])} epochs from: {history_path}")
    print(f"Best val epoch: {best_ep + 1}  (mean_iou = {val_ious[best_ep]:.4f})")

    class_names = ["Background", "Crack", "Spall", "Delamination", "Moisture"]
    viz = DefectVisualization(output_dir="./visualizations")

    viz.plot_training_history(history)

    if cm is not None:
        viz.plot_confusion_matrix(cm, class_names=class_names)

    print("Done — check ./visualizations/")
