#!/usr/bin/env python3
"""
Evaluate Optimized GPT-4o Model on 2024 Holdout Data
====================================================

Final test of the refined model on unseen 2024 data.
"""

import pandas as pd
import numpy as np
import logging
import sys
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add src directory
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import the model class and feature engineering
from optimize_gpt4o_final import RefinedCascadeClassifier, engineer_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_2024_test_data():
    """Load 2024 holdout test data."""
    logger.info("Loading 2024 holdout test data...")
    
    # Load structured data
    df_struct = pd.read_excel("data/2024 Applicants Reviewed by Trusted Reviewers/1. Applicants.xlsx")
    
    # Load LLM scores
    llm_path = "data/2024 Applicants Reviewed by Trusted Reviewers/llm_scores_2024.csv"
    
    if not Path(llm_path).exists():
        logger.error("2024 LLM scores not found! Generating simulated scores...")
        # Generate simulated scores for testing
        import simulate_gpt4o_scores
        simulate_gpt4o_scores.process_year('2024')
    
    df_llm = pd.read_csv(llm_path)
    
    # Standardize AMCAS ID
    if 'Amcas_ID' in df_struct.columns:
        df_struct['AMCAS ID'] = df_struct['Amcas_ID'].astype(str)
    df_llm['AMCAS ID'] = df_llm['AMCAS ID'].astype(str)
    
    # Merge
    df_test = pd.merge(df_struct, df_llm, on='AMCAS ID', how='inner')
    logger.info(f"Loaded {len(df_test)} test records for 2024")
    
    return df_test


def score_to_quartile(score):
    """Convert application score to quartile."""
    if score <= 9:
        return 0  # Q4 (Reject)
    elif score <= 15:
        return 1  # Q3 (Waitlist)
    elif score <= 22:
        return 2  # Q2 (Interview)
    else:
        return 3  # Q1 (Accept)


def evaluate_cascade_model(model_data, X_test, y_test):
    """Evaluate the cascade model on test data."""
    model = model_data['model']
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert scores to quartiles
    y_test_quartiles = np.array([score_to_quartile(s) for s in y_test])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_quartiles, y_pred)
    
    # Adjacent accuracy (within 1 quartile)
    adjacent = np.abs(y_pred - y_test_quartiles) <= 1
    adjacent_acc = np.mean(adjacent)
    
    # Per-quartile metrics
    quartile_metrics = {}
    for q in range(4):
        mask = y_test_quartiles == q
        if np.any(mask):
            q_acc = accuracy_score(y_test_quartiles[mask], y_pred[mask])
            quartile_metrics[f'Q{4-q}'] = {
                'accuracy': q_acc,
                'count': np.sum(mask),
                'predicted_correctly': np.sum(y_pred[mask] == q)
            }
    
    # Confusion matrix
    cm = confusion_matrix(y_test_quartiles, y_pred, labels=[0, 1, 2, 3])
    
    return {
        'accuracy': accuracy,
        'adjacent_accuracy': adjacent_acc,
        'confusion_matrix': cm,
        'quartile_metrics': quartile_metrics,
        'predictions': y_pred,
        'true_quartiles': y_test_quartiles
    }


def create_evaluation_report(results, output_dir):
    """Create comprehensive evaluation report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Q4', 'Q3', 'Q2', 'Q1'],
                yticklabels=['Q4', 'Q3', 'Q2', 'Q1'])
    plt.title('2024 Holdout Test - Confusion Matrix', fontsize=14)
    plt.ylabel('True Quartile')
    plt.xlabel('Predicted Quartile')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_2024.png", dpi=300)
    plt.close()
    
    # 2. Performance Summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Quartile accuracies
    quartiles = []
    accuracies = []
    counts = []
    for q_name, metrics in sorted(results['quartile_metrics'].items()):
        quartiles.append(q_name)
        accuracies.append(metrics['accuracy'])
        counts.append(metrics['count'])
    
    bars = ax1.bar(quartiles, accuracies, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Per-Quartile Accuracy')
    ax1.axhline(y=results['accuracy'], color='black', linestyle='--', label=f'Overall: {results["accuracy"]:.1%}')
    ax1.legend()
    
    # Add value labels
    for bar, acc, count in zip(bars, accuracies, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}\n(n={count})', ha='center', va='bottom')
    
    # Error distribution
    errors = np.abs(results['predictions'] - results['true_quartiles'])
    error_counts = [np.sum(errors == i) for i in range(4)]
    
    ax2.bar(range(4), error_counts, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Quartile Distance')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['Exact', '±1', '±2', '±3'])
    
    # Add percentage labels
    total = len(errors)
    for i, count in enumerate(error_counts):
        ax2.text(i, count + 1, f'{count/total:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_summary_2024.png", dpi=300)
    plt.close()
    
    # 3. Write detailed report
    with open(f"{output_dir}/evaluation_report_2024.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("2024 HOLDOUT TEST EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Set Size: {len(results['predictions'])} applicants\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-"*30 + "\n")
        f.write(f"Exact Match Accuracy: {results['accuracy']:.1%}\n")
        f.write(f"Adjacent Accuracy (±1): {results['adjacent_accuracy']:.1%}\n\n")
        
        f.write("PER-QUARTILE BREAKDOWN:\n")
        f.write("-"*30 + "\n")
        for q_name, metrics in sorted(results['quartile_metrics'].items()):
            f.write(f"\n{q_name} Performance:\n")
            f.write(f"  Count: {metrics['count']}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.1%}\n")
            f.write(f"  Correctly Predicted: {metrics['predicted_correctly']}\n")
        
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write("-"*30 + "\n")
        f.write("True\\Pred   Q4    Q3    Q2    Q1\n")
        labels = ['Q4', 'Q3', 'Q2', 'Q1']
        for i, label in enumerate(labels):
            f.write(f"{label:8}")
            for j in range(4):
                f.write(f"{results['confusion_matrix'][i,j]:6}")
            f.write("\n")
        
        f.write("\n\nKEY INSIGHTS:\n")
        f.write("-"*30 + "\n")
        
        # Calculate key metrics
        exact_pct = results['accuracy'] * 100
        adj_pct = results['adjacent_accuracy'] * 100
        
        if exact_pct >= 82:
            f.write(f"✓ Model achieved target accuracy of >82% ({exact_pct:.1f}%)\n")
        else:
            f.write(f"✗ Model below target accuracy: {exact_pct:.1f}% (target: 82%)\n")
        
        if adj_pct >= 95:
            f.write(f"✓ Excellent adjacent accuracy: {adj_pct:.1f}%\n")
        else:
            f.write(f"• Adjacent accuracy: {adj_pct:.1f}%\n")
        
        # Identify best and worst performing quartiles
        best_q = max(results['quartile_metrics'].items(), key=lambda x: x[1]['accuracy'])
        worst_q = min(results['quartile_metrics'].items(), key=lambda x: x[1]['accuracy'])
        
        f.write(f"\nBest performing: {best_q[0]} ({best_q[1]['accuracy']:.1%})\n")
        f.write(f"Needs improvement: {worst_q[0]} ({worst_q[1]['accuracy']:.1%})\n")
    
    logger.info(f"Evaluation report saved to {output_dir}")


def main():
    """Main evaluation function."""
    logger.info("🚀 EVALUATING OPTIMIZED GPT-4O MODEL ON 2024 HOLDOUT")
    logger.info("="*60)
    
    # Load the trained model
    model_path = "models/refined_gpt4o_latest.pkl"
    logger.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    
    # Load 2024 test data
    test_df = load_2024_test_data()
    
    # Extract features using the fitted feature engineer
    feature_engineer = model_data['feature_engineer']
    X_test = feature_engineer.transform(test_df)
    
    # Apply the same feature engineering as training
    X_test_enhanced, _ = engineer_features(X_test, feature_engineer.feature_names)
    
    # Handle different column names for application score
    if 'Application Review Score' in test_df.columns:
        y_test = test_df['Application Review Score'].values
    elif 'Application_Review_Score' in test_df.columns:
        y_test = test_df['Application_Review_Score'].values
    else:
        logger.error("Could not find application score column!")
        return
    
    logger.info(f"Test set shape: {X_test_enhanced.shape}")
    
    # Evaluate model
    results = evaluate_cascade_model(model_data, X_test_enhanced, y_test)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/evaluation_2024_{timestamp}"
    
    # Generate comprehensive report
    create_evaluation_report(results, output_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 2024 HOLDOUT TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Exact Match Accuracy: {results['accuracy']:.1%}")
    logger.info(f"Adjacent Accuracy (±1): {results['adjacent_accuracy']:.1%}")
    
    for q_name, metrics in sorted(results['quartile_metrics'].items()):
        logger.info(f"{q_name} Accuracy: {metrics['accuracy']:.1%} (n={metrics['count']})")
    
    logger.info("="*60)
    
    if results['accuracy'] >= 0.82:
        logger.info("🎉 TARGET ACHIEVED! Model exceeds 82% accuracy!")
    else:
        logger.info(f"📈 Model accuracy: {results['accuracy']:.1%} (Target: 82%)")
    
    return results


if __name__ == "__main__":
    main()