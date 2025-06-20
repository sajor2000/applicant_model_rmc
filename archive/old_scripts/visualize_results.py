"""
Visualize Model Results and Performance
======================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_visualizations():
    """Create performance visualizations"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Medical Admissions AI System Performance', fontsize=16)
    
    # 1. Training Score Distribution
    ax1 = axes[0, 0]
    scores = ([0]*54 + [2]*1 + [3]*2 + [4]*1 + [5]*11 + [7]*20 + [9]*25 + 
              [11]*76 + [13]*38 + [15]*168 + [16]*1 + [17]*71 + 
              [19]*152 + [21]*74 + [23]*66 + [25]*78)
    
    bucket_colors = ['red', 'orange', 'yellow', 'green']
    bucket_boundaries = [0, 10, 16, 22, 26]
    
    # Create histogram with bucket coloring
    n, bins, patches = ax1.hist(scores, bins=26, edgecolor='black', alpha=0.7)
    
    # Color bars by bucket
    for i, patch in enumerate(patches):
        if bins[i] < bucket_boundaries[1]:
            patch.set_facecolor(bucket_colors[0])
        elif bins[i] < bucket_boundaries[2]:
            patch.set_facecolor(bucket_colors[1])
        elif bins[i] < bucket_boundaries[3]:
            patch.set_facecolor(bucket_colors[2])
        else:
            patch.set_facecolor(bucket_colors[3])
    
    ax1.axvline(x=19, color='red', linestyle='--', linewidth=2, label='Interview Threshold')
    ax1.set_xlabel('Application Review Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Training Data Score Distribution (2022-2023)')
    ax1.legend()
    
    # 2. Bucket Distribution Comparison
    ax2 = axes[0, 1]
    buckets = ['Reject\n(0-9)', 'Waitlist\n(11-15)', 'Interview\n(17-21)', 'Accept\n(23-25)']
    training_pcts = [13.6, 33.7, 35.6, 17.2]
    predicted_2024 = [98.7, 1.3, 0.0, 0.0]
    
    x = np.arange(len(buckets))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, training_pcts, width, label='Training (2022-23)', 
                     color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    bars2 = ax2.bar(x + width/2, predicted_2024, width, label='Predicted (2024)', 
                     color=['darkred', 'darkorange', 'gold', 'darkgreen'], alpha=0.7)
    
    ax2.set_xlabel('Buckets')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Bucket Distribution: Training vs 2024 Predictions')
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets)
    ax2.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
    
    # 3. Model Performance Metrics
    ax3 = axes[1, 0]
    metrics = ['Exact\nMatch', 'Adjacent\n(±1)', 'QWK']
    values = [57.7, 98.8, 72.4]
    colors = ['steelblue', 'lightsteelblue', 'darkblue']
    
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Performance (%)')
    ax3.set_title('Model Performance Metrics')
    ax3.set_ylim(0, 110)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, value + 2, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Feature Importance Groups
    ax4 = axes[1, 1]
    feature_groups = ['Original\nFeatures', 'Bucket\nIndicators', 'Ratios', 
                      'Interactions', 'Polynomials', 'Thresholds']
    importance = [78.4, 7.5, 6.6, 5.7, 1.7, 0.2]
    
    # Create pie chart
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(feature_groups)))
    wedges, texts, autotexts = ax4.pie(importance, labels=feature_groups, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90)
    ax4.set_title('Feature Group Importance')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('black')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('model_performance_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as: model_performance_visualization.png")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    
    # Validation confusion matrix
    cm = np.array([[10, 12, 1, 0],
                   [3, 33, 19, 1],
                   [0, 21, 30, 9],
                   [0, 0, 5, 24]])
    
    # Normalize by true labels
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=buckets, yticklabels=buckets,
                cbar_kws={'label': 'Normalized Accuracy'})
    
    plt.title('Validation Set Confusion Matrix\n(counts shown, colors show row-normalized accuracy)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved as: confusion_matrix.png")
    
    # plt.show()  # Comment out for non-GUI environments

if __name__ == "__main__":
    print("Creating visualizations...")
    create_visualizations()
    print("\n✓ All visualizations complete!")