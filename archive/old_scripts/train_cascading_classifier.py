"""
Cascading Binary Classifiers for Fine-Grained Applicant Evaluation
==================================================================

Train multiple binary classifiers in cascade to better distinguish
between different levels of applicants, especially in the middle range.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CascadingClassifier:
    """Cascading binary classifiers for multi-level classification."""
    
    def __init__(self):
        self.models = {}
        self.thresholds = {}
        self.imputers = {}
        self.scalers = {}
        self.feature_cols = None
        
    def predict_cascade(self, X):
        """Make predictions using the cascade of models."""
        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples, dtype=int)
        probabilities = np.zeros((n_samples, 4))  # For 4 classes
        
        # Stage 1: Reject vs Non-Reject
        prob_not_reject = self.models['stage1'].predict_proba(X)[:, 1]
        is_reject = prob_not_reject < self.thresholds['stage1']
        final_predictions[is_reject] = 0  # Reject
        probabilities[:, 0] = 1 - prob_not_reject
        
        # For non-rejects, proceed to stage 2
        non_reject_mask = ~is_reject
        if non_reject_mask.sum() > 0:
            X_stage2 = X[non_reject_mask]
            
            # Stage 2: Waitlist vs Higher (Interview/Accept)
            prob_higher = self.models['stage2'].predict_proba(X_stage2)[:, 1]
            is_waitlist = prob_higher < self.thresholds['stage2']
            
            # Update predictions for waitlist
            non_reject_indices = np.where(non_reject_mask)[0]
            waitlist_indices = non_reject_indices[is_waitlist]
            final_predictions[waitlist_indices] = 1  # Waitlist
            probabilities[non_reject_mask, 1] = 1 - prob_higher
            
            # For higher candidates, proceed to stage 3
            higher_mask = np.zeros(n_samples, dtype=bool)
            higher_indices = non_reject_indices[~is_waitlist]
            higher_mask[higher_indices] = True
            
            if higher_mask.sum() > 0:
                X_stage3 = X[higher_mask]
                
                # Stage 3: Interview vs Accept
                prob_accept = self.models['stage3'].predict_proba(X_stage3)[:, 1]
                is_accept = prob_accept >= self.thresholds['stage3']
                
                # Update predictions
                accept_indices = higher_indices[is_accept]
                interview_indices = higher_indices[~is_accept]
                
                final_predictions[accept_indices] = 3  # Accept
                final_predictions[interview_indices] = 2  # Interview
                
                probabilities[higher_mask, 2] = 1 - prob_accept
                probabilities[higher_mask, 3] = prob_accept
        
        return final_predictions, probabilities


def load_and_prepare_data():
    """Load data and prepare for cascading classification."""
    print("\n1. Loading and preparing data...")
    
    # Load filtered data
    df_2022 = pd.read_excel("data_filtered/2022_filtered_applicants.xlsx")
    df_2023 = pd.read_excel("data_filtered/2023_filtered_applicants.xlsx")
    df_2024 = pd.read_excel("data_filtered/2024_filtered_applicants.xlsx")
    
    # Load LLM scores
    llm_train = pd.read_csv("llm_scores_2022_2023_20250619_172837.csv")
    llm_train = llm_train.rename(columns={'AMCAS_ID_original': 'amcas_id'})
    
    llm_test_files = list(Path(".").glob("llm_scores_2024_*.csv"))
    if llm_test_files:
        llm_test = pd.read_csv(sorted(llm_test_files)[-1])
        llm_test = llm_test.rename(columns={'AMCAS_ID_original': 'amcas_id'})
    else:
        llm_test = None
    
    # Get LLM columns
    llm_cols = [col for col in llm_train.columns if col.startswith('llm_')]
    
    # Merge LLM scores
    df_2022 = df_2022.merge(llm_train[['amcas_id'] + llm_cols], on='amcas_id', how='left')
    df_2023 = df_2023.merge(llm_train[['amcas_id'] + llm_cols], on='amcas_id', how='left')
    
    if llm_test is not None:
        df_2024 = df_2024.merge(llm_test[['amcas_id'] + llm_cols], on='amcas_id', how='left')
    
    # Combine training data
    df_train = pd.concat([df_2022, df_2023], ignore_index=True)
    
    print(f"   Training samples: {len(df_train)}")
    print(f"   Test samples: {len(df_2024)}")
    
    return df_train, df_2024


def prepare_features(df, feature_cols=None):
    """Prepare features for modeling."""
    exclude_cols = ['application_review_score', 'amcas_id', 'appl_year', 
                   'year', 'AMCAS_ID', 'is_high_scorer']
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    # Handle categorical features
    categorical_mappings = {
        'gender': {'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': -1},
        'citizenship': {'US_Citizen': 0, 'Permanent_Resident': 1, 'International': 2, 'Other': 3},
        'service_rating_categorical': {
            'Exceptional': 5, 'Outstanding': 4, 'Excellent': 3,
            'Good': 2, 'Average': 1, 'Below Average': 0, 'Poor': -1
        }
    }
    
    for col, mapping in categorical_mappings.items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(-2)
    
    # Convert remaining categoricals
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    return X, feature_cols


def train_stage_classifier(X_train, y_train, stage_name, class_weight='balanced'):
    """Train a single stage classifier with cross-validation."""
    print(f"\n   Training {stage_name}...")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Create classifier
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Skip if only one class in validation
        if len(np.unique(y_fold_val)) < 2:
            continue
            
        fold_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_fold_train == 0).sum() / (y_fold_train == 1).sum() if (y_fold_train == 1).sum() > 0 else 1,
            random_state=fold,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        fold_model.fit(X_fold_train, y_fold_train, verbose=False)
        
        y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(auc)
    
    if cv_scores:
        print(f"   CV AUC: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
    
    # Train final model
    model.fit(X_train, y_train, verbose=False)
    
    # Find optimal threshold
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"   Optimal threshold: {optimal_threshold:.3f}")
    
    return model, optimal_threshold


def train_cascading_classifier(df_train, df_test):
    """Train the cascading classifier system."""
    print("\n2. Preparing features...")
    
    # Prepare features
    X_all, feature_cols = prepare_features(df_train)
    y_all = df_train['application_review_score'].values
    
    # Create imputer and scaler for all data
    imputer = SimpleImputer(strategy='median')
    X_all_imputed = imputer.fit_transform(X_all)
    
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_imputed)
    
    print(f"   Total features: {X_all.shape[1]}")
    
    # Initialize cascade classifier
    cascade = CascadingClassifier()
    cascade.feature_cols = feature_cols
    
    print("\n3. Training Stage 1: Reject vs Non-Reject")
    # Stage 1: Reject (≤9) vs Non-Reject (>9)
    y_stage1 = (y_all > 9).astype(int)
    model1, threshold1 = train_stage_classifier(X_all_scaled, y_stage1, "Stage 1 (Reject vs Others)")
    cascade.models['stage1'] = model1
    cascade.thresholds['stage1'] = threshold1
    
    print("\n4. Training Stage 2: Waitlist vs Higher")
    # Stage 2: Among non-rejects, Waitlist (10-15) vs Higher (>15)
    non_reject_mask = y_all > 9
    X_stage2 = X_all_scaled[non_reject_mask]
    y_stage2_scores = y_all[non_reject_mask]
    y_stage2 = (y_stage2_scores > 15).astype(int)
    model2, threshold2 = train_stage_classifier(X_stage2, y_stage2, "Stage 2 (Waitlist vs Interview+)")
    cascade.models['stage2'] = model2
    cascade.thresholds['stage2'] = threshold2
    
    print("\n5. Training Stage 3: Interview vs Accept")
    # Stage 3: Among higher, Interview (16-22) vs Accept (≥23)
    higher_mask = y_all > 15
    X_stage3 = X_all_scaled[higher_mask]
    y_stage3_scores = y_all[higher_mask]
    y_stage3 = (y_stage3_scores >= 23).astype(int)
    model3, threshold3 = train_stage_classifier(X_stage3, y_stage3, "Stage 3 (Interview vs Accept)")
    cascade.models['stage3'] = model3
    cascade.thresholds['stage3'] = threshold3
    
    # Store preprocessing objects
    cascade.imputers['main'] = imputer
    cascade.scalers['main'] = scaler
    
    # Evaluate on training set
    print("\n6. Evaluating cascade on training set...")
    train_predictions, train_probs = cascade.predict_cascade(X_all_scaled)
    
    # Convert actual scores to buckets
    bucket_names = ['Reject', 'Waitlist', 'Interview', 'Accept']
    true_buckets = np.zeros(len(y_all), dtype=int)
    true_buckets[y_all <= 9] = 0  # Reject
    true_buckets[(y_all >= 10) & (y_all <= 15)] = 1  # Waitlist
    true_buckets[(y_all >= 16) & (y_all <= 22)] = 2  # Interview
    true_buckets[y_all >= 23] = 3  # Accept
    
    # Confusion matrix
    print("\n   Training Set Confusion Matrix:")
    cm = confusion_matrix(true_buckets, train_predictions)
    print("   True\\Pred  Reject  Waitlist  Interview  Accept")
    for i, name in enumerate(bucket_names):
        row = f"   {name:10}"
        for j in range(4):
            row += f"{cm[i,j]:8d}"
        print(row)
    
    # Accuracy metrics
    exact_match = np.mean(train_predictions == true_buckets)
    adjacent = np.mean(np.abs(train_predictions - true_buckets) <= 1)
    print(f"\n   Training Exact Match: {exact_match:.3f}")
    print(f"   Training Adjacent: {adjacent:.3f}")
    
    # Evaluate on test set
    print("\n7. Evaluating on 2024 test set...")
    X_test, _ = prepare_features(df_test, feature_cols)
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    test_predictions, test_probs = cascade.predict_cascade(X_test_scaled)
    
    # True buckets for test set
    y_test = df_test['application_review_score'].values
    true_test_buckets = np.zeros(len(y_test), dtype=int)
    true_test_buckets[y_test <= 9] = 0
    true_test_buckets[(y_test >= 10) & (y_test <= 15)] = 1
    true_test_buckets[(y_test >= 16) & (y_test <= 22)] = 2
    true_test_buckets[y_test >= 23] = 3
    
    # Test confusion matrix
    print("\n   Test Set Confusion Matrix:")
    cm_test = confusion_matrix(true_test_buckets, test_predictions)
    print("   True\\Pred  Reject  Waitlist  Interview  Accept")
    for i, name in enumerate(bucket_names):
        row = f"   {name:10}"
        for j in range(4):
            row += f"{cm_test[i,j]:8d}"
        print(row)
    
    test_exact = np.mean(test_predictions == true_test_buckets)
    test_adjacent = np.mean(np.abs(test_predictions - true_test_buckets) <= 1)
    print(f"\n   Test Exact Match: {test_exact:.3f}")
    print(f"   Test Adjacent: {test_adjacent:.3f}")
    
    # Analyze predictions by bucket
    print("\n   Test Set Predictions by True Bucket:")
    for i, name in enumerate(bucket_names):
        mask = true_test_buckets == i
        if mask.sum() > 0:
            pred_dist = np.bincount(test_predictions[mask], minlength=4)
            print(f"   {name} (n={mask.sum()}):")
            for j, pred_name in enumerate(bucket_names):
                pct = pred_dist[j] / mask.sum() * 100
                print(f"     -> {pred_name}: {pred_dist[j]} ({pct:.1f}%)")
    
    # Feature importance across stages
    print("\n8. Top Features by Stage:")
    for stage_name, model in cascade.models.items():
        print(f"\n   {stage_name.upper()} - Top 10 Features:")
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        
        for idx in top_indices[:10]:
            if idx < len(feature_cols):
                print(f"     - feature_{idx}: {importances[idx]:.3f}")
    
    # Save results
    results = pd.DataFrame({
        'amcas_id': df_test['amcas_id'],
        'true_score': y_test,
        'true_bucket': [bucket_names[i] for i in true_test_buckets],
        'predicted_bucket': [bucket_names[i] for i in test_predictions],
        'reject_prob': test_probs[:, 0],
        'waitlist_prob': test_probs[:, 1],
        'interview_prob': test_probs[:, 2],
        'accept_prob': test_probs[:, 3]
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.to_csv(f"cascade_predictions_{timestamp}.csv", index=False)
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    model_data = {
        'cascade': cascade,
        'feature_cols': feature_cols,
        'training_metrics': {
            'exact_match': exact_match,
            'adjacent': adjacent
        },
        'test_metrics': {
            'exact_match': test_exact,
            'adjacent': test_adjacent
        }
    }
    
    model_path = f"models/cascade_classifier_{timestamp}.pkl"
    joblib.dump(model_data, model_path)
    joblib.dump(model_data, "models/cascade_classifier_latest.pkl")
    
    print(f"\n   Model saved to: {model_path}")
    
    # Create visualizations
    create_cascade_visualizations(train_predictions, true_buckets, 
                                test_predictions, true_test_buckets,
                                bucket_names)
    
    return cascade, results


def create_cascade_visualizations(train_pred, train_true, test_pred, test_true, bucket_names):
    """Create visualizations for cascade performance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training confusion matrix heatmap
    ax1 = axes[0, 0]
    cm_train = confusion_matrix(train_true, train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                xticklabels=bucket_names, yticklabels=bucket_names, ax=ax1)
    ax1.set_title('Training Set Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Test confusion matrix heatmap
    ax2 = axes[0, 1]
    cm_test = confusion_matrix(test_true, test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges',
                xticklabels=bucket_names, yticklabels=bucket_names, ax=ax2)
    ax2.set_title('Test Set (2024) Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 3. Accuracy by bucket
    ax3 = axes[1, 0]
    train_acc_by_bucket = []
    test_acc_by_bucket = []
    
    for i in range(4):
        train_mask = train_true == i
        test_mask = test_true == i
        
        if train_mask.sum() > 0:
            train_acc = np.mean(train_pred[train_mask] == i)
            train_acc_by_bucket.append(train_acc)
        else:
            train_acc_by_bucket.append(0)
            
        if test_mask.sum() > 0:
            test_acc = np.mean(test_pred[test_mask] == i)
            test_acc_by_bucket.append(test_acc)
        else:
            test_acc_by_bucket.append(0)
    
    x = np.arange(len(bucket_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, train_acc_by_bucket, width, label='Training', alpha=0.8)
    bars2 = ax3.bar(x + width/2, test_acc_by_bucket, width, label='Test', alpha=0.8)
    
    ax3.set_xlabel('True Bucket')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Per-Bucket Accuracy')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bucket_names)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # 4. Prediction distribution
    ax4 = axes[1, 1]
    train_dist = np.bincount(train_pred, minlength=4) / len(train_pred)
    test_dist = np.bincount(test_pred, minlength=4) / len(test_pred)
    true_train_dist = np.bincount(train_true, minlength=4) / len(train_true)
    true_test_dist = np.bincount(test_true, minlength=4) / len(test_true)
    
    x = np.arange(len(bucket_names))
    width = 0.2
    
    ax4.bar(x - 1.5*width, true_train_dist, width, label='True Train', alpha=0.8, color='blue')
    ax4.bar(x - 0.5*width, train_dist, width, label='Pred Train', alpha=0.8, color='lightblue')
    ax4.bar(x + 0.5*width, true_test_dist, width, label='True Test', alpha=0.8, color='orange')
    ax4.bar(x + 1.5*width, test_dist, width, label='Pred Test', alpha=0.8, color='lightsalmon')
    
    ax4.set_xlabel('Bucket')
    ax4.set_ylabel('Proportion')
    ax4.set_title('Distribution Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bucket_names)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('cascade_analysis.png', dpi=300, bbox_inches='tight')
    print("\n   Visualizations saved to cascade_analysis.png")


def main():
    print("="*80)
    print("CASCADING CLASSIFIER FOR FINE-GRAINED EVALUATION")
    print("Training multiple stages to distinguish all applicant levels")
    print("="*80)
    
    # Load data
    df_train, df_test = load_and_prepare_data()
    
    # Train cascading classifier
    cascade, results = train_cascading_classifier(df_train, df_test)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("Cascading classifier provides fine-grained distinctions")
    print("="*80)


if __name__ == "__main__":
    main()