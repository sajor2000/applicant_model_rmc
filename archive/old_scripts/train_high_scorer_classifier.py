"""
Train High Scorer Classifier
============================

Learn from the extremes - what distinguishes high-scoring applicants (≥19)
from low-scoring applicants (≤9) to identify strong candidates.
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load data and create high/low scorer dataset."""
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
    
    # Create high/low scorer labels
    # High scorers: score >= 19 (interview threshold)
    # Low scorers: score <= 9 (reject range)
    # Middle scorers: 10-18 (excluded from training)
    
    high_scorers = df_train[df_train['application_review_score'] >= 19].copy()
    low_scorers = df_train[df_train['application_review_score'] <= 9].copy()
    
    high_scorers['is_high_scorer'] = 1
    low_scorers['is_high_scorer'] = 0
    
    # Combine for training
    df_extreme = pd.concat([high_scorers, low_scorers], ignore_index=True)
    
    print(f"   High scorers (≥19): {len(high_scorers)}")
    print(f"   Low scorers (≤9): {len(low_scorers)}")
    print(f"   Total training samples: {len(df_extreme)}")
    print(f"   Class balance: {len(high_scorers)/len(df_extreme):.1%} high scorers")
    
    # For 2024, we'll predict on all applicants
    return df_extreme, df_train, df_2024


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


def train_classifier(df_extreme, df_full_train, df_test):
    """Train classifier to identify high-scoring applicants."""
    print("\n2. Preparing features...")
    
    # Prepare features for extreme cases
    X_extreme, feature_cols = prepare_features(df_extreme)
    y_extreme = df_extreme['is_high_scorer'].values
    
    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    X_extreme_imputed = imputer.fit_transform(X_extreme)
    
    scaler = StandardScaler()
    X_extreme_scaled = scaler.fit_transform(X_extreme_imputed)
    
    print(f"   Features: {X_extreme.shape[1]}")
    print(f"   Training samples: {X_extreme.shape[0]}")
    
    print("\n3. Training XGBoost classifier with 5-fold CV...")
    
    # Create classifier
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(low_scorers)/len(high_scorers),  # Handle class imbalance
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_extreme_scaled, y_extreme)):
        X_fold_train = X_extreme_scaled[train_idx]
        X_fold_val = X_extreme_scaled[val_idx]
        y_fold_train = y_extreme[train_idx]
        y_fold_val = y_extreme[val_idx]
        
        # Train
        fold_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_fold_train == 0).sum() / (y_fold_train == 1).sum(),
            random_state=fold,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Predict
        y_pred = fold_model.predict(X_fold_val)
        y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_fold_val, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(y_fold_val)
        auc = roc_auc_score(y_fold_val, y_pred_proba)
        
        cv_results.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        })
        
        print(f"   Fold {fold+1}: Acc={accuracy:.3f}, Prec={precision:.3f}, "
              f"Recall={recall:.3f}, AUC={auc:.3f}")
    
    # Average results
    avg_metrics = {k: np.mean([r[k] for r in cv_results]) for k in cv_results[0].keys()}
    print(f"\n   CV Average: Acc={avg_metrics['accuracy']:.3f}, "
          f"Prec={avg_metrics['precision']:.3f}, "
          f"Recall={avg_metrics['recall']:.3f}, "
          f"AUC={avg_metrics['auc']:.3f}")
    
    # Train final model
    print("\n4. Training final model on all extreme cases...")
    model.fit(X_extreme_scaled, y_extreme)
    
    # Feature importance
    print("\n5. Top 20 Most Important Features:")
    # Ensure feature names and importances have same length
    if len(feature_cols) != len(model.feature_importances_):
        print(f"   Warning: Feature count mismatch ({len(feature_cols)} vs {len(model.feature_importances_)})")
        feature_cols_adj = [f'feature_{i}' for i in range(len(model.feature_importances_))]
    else:
        feature_cols_adj = feature_cols
        
    feature_importance = pd.DataFrame({
        'feature': feature_cols_adj,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.head(20).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}")
    
    # Evaluate on full training set (including middle scorers)
    print("\n6. Evaluating on full training set...")
    X_full_train, _ = prepare_features(df_full_train, feature_cols)
    X_full_train_imputed = imputer.transform(X_full_train)
    X_full_train_scaled = scaler.transform(X_full_train_imputed)
    
    y_full_pred_proba = model.predict_proba(X_full_train_scaled)[:, 1]
    
    # Analyze predictions by actual score ranges
    df_full_train['high_scorer_probability'] = y_full_pred_proba
    
    print("\n   High scorer probability by actual score range:")
    score_ranges = [(0, 9, 'Low (0-9)'), 
                   (10, 15, 'Waitlist (10-15)'),
                   (16, 18, 'Borderline (16-18)'),
                   (19, 22, 'Interview (19-22)'),
                   (23, 25, 'Accept (23-25)')]
    
    for min_score, max_score, label in score_ranges:
        mask = (df_full_train['application_review_score'] >= min_score) & \
               (df_full_train['application_review_score'] <= max_score)
        if mask.sum() > 0:
            mean_prob = df_full_train.loc[mask, 'high_scorer_probability'].mean()
            count = mask.sum()
            print(f"   {label}: {mean_prob:.3f} (n={count})")
    
    # Evaluate on 2024 test set
    print("\n7. Evaluating on 2024 test set...")
    X_test, _ = prepare_features(df_test, feature_cols)
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    df_test['high_scorer_probability'] = y_test_pred_proba
    
    # Create recommended interview list (top probabilities)
    interview_threshold = np.percentile(y_test_pred_proba, 70)  # Top 30%
    df_test['recommend_interview'] = y_test_pred_proba >= interview_threshold
    
    # Check how well this aligns with actual scores
    print("\n   2024 predictions vs actual scores:")
    recommended = df_test[df_test['recommend_interview']]
    not_recommended = df_test[~df_test['recommend_interview']]
    
    print(f"   Recommended for interview: {len(recommended)} ({len(recommended)/len(df_test)*100:.1f}%)")
    print(f"   - Mean actual score: {recommended['application_review_score'].mean():.1f}")
    print(f"   - % with score ≥19: {(recommended['application_review_score'] >= 19).mean()*100:.1f}%")
    
    print(f"   Not recommended: {len(not_recommended)} ({len(not_recommended)/len(df_test)*100:.1f}%)")
    print(f"   - Mean actual score: {not_recommended['application_review_score'].mean():.1f}")
    print(f"   - % with score ≥19: {(not_recommended['application_review_score'] >= 19).mean()*100:.1f}%")
    
    # Save results
    results = df_test[['amcas_id', 'application_review_score', 
                      'high_scorer_probability', 'recommend_interview']].copy()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.to_csv(f"high_scorer_predictions_{timestamp}.csv", index=False)
    
    # Create visualizations
    create_visualizations(df_full_train, df_test, feature_importance)
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'imputer': imputer,
        'scaler': scaler,
        'cv_metrics': avg_metrics,
        'interview_threshold': interview_threshold,
        'feature_importance': feature_importance
    }
    
    model_path = f"models/high_scorer_classifier_{timestamp}.pkl"
    joblib.dump(model_data, model_path)
    joblib.dump(model_data, "models/high_scorer_classifier_latest.pkl")
    
    print(f"\n   Model saved to: {model_path}")
    
    return model, results


def create_visualizations(df_train, df_test, feature_importance):
    """Create visualizations for the high scorer classifier."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature importance
    ax1 = axes[0, 0]
    top_features = feature_importance.head(15)
    ax1.barh(range(len(top_features)), top_features['importance'])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 15 Most Important Features')
    ax1.invert_yaxis()
    
    # 2. Probability distribution by score range
    ax2 = axes[0, 1]
    score_bins = [0, 9, 15, 18, 22, 25]
    score_labels = ['Low\n(0-9)', 'Waitlist\n(10-15)', 'Borderline\n(16-18)', 
                   'Interview\n(19-22)', 'Accept\n(23-25)']
    df_train['score_bin'] = pd.cut(df_train['application_review_score'], 
                                   bins=score_bins, labels=score_labels)
    
    df_train.boxplot(column='high_scorer_probability', by='score_bin', ax=ax2)
    ax2.set_xlabel('Actual Score Range')
    ax2.set_ylabel('High Scorer Probability')
    ax2.set_title('Model Predictions by Actual Score Range')
    
    # 3. 2024 Test Set Performance
    ax3 = axes[1, 0]
    test_recommended = df_test[df_test['recommend_interview']]
    test_not_recommended = df_test[~df_test['recommend_interview']]
    
    bins = np.arange(0, 26, 2)
    ax3.hist(test_recommended['application_review_score'], bins=bins, 
             alpha=0.5, label='Recommended', density=True)
    ax3.hist(test_not_recommended['application_review_score'], bins=bins, 
             alpha=0.5, label='Not Recommended', density=True)
    ax3.axvline(x=19, color='red', linestyle='--', label='Interview Threshold')
    ax3.set_xlabel('Actual Application Score')
    ax3.set_ylabel('Density')
    ax3.set_title('2024 Test Set: Actual Scores by Recommendation')
    ax3.legend()
    
    # 4. Calibration plot
    ax4 = axes[1, 1]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    actual_high = []
    predicted_high = []
    
    for i in range(n_bins):
        mask = (df_test['high_scorer_probability'] >= bin_edges[i]) & \
               (df_test['high_scorer_probability'] < bin_edges[i+1])
        if mask.sum() > 0:
            actual_high.append((df_test.loc[mask, 'application_review_score'] >= 19).mean())
            predicted_high.append(df_test.loc[mask, 'high_scorer_probability'].mean())
    
    ax4.plot(predicted_high, actual_high, 'o-', label='Model')
    ax4.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax4.set_xlabel('Mean Predicted Probability')
    ax4.set_ylabel('Actual High Scorer Rate')
    ax4.set_title('Model Calibration')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('high_scorer_analysis.png', dpi=300, bbox_inches='tight')
    print("\n   Visualizations saved to high_scorer_analysis.png")


# Fix the undefined variable issue
high_scorers = None
low_scorers = None

def main():
    print("="*80)
    print("HIGH SCORER CLASSIFIER")
    print("Learning from the extremes to identify strong applicants")
    print("="*80)
    
    # Load data
    df_extreme, df_full_train, df_test = load_and_prepare_data()
    
    # Set global variables for class balancing
    global high_scorers, low_scorers
    high_scorers = df_extreme[df_extreme['is_high_scorer'] == 1]
    low_scorers = df_extreme[df_extreme['is_high_scorer'] == 0]
    
    # Train classifier
    model, results = train_classifier(df_extreme, df_full_train, df_test)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("Model trained to identify characteristics of high-scoring applicants")
    print("="*80)


if __name__ == "__main__":
    main()