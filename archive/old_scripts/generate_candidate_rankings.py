"""
Generate Candidate Rankings from Model Predictions
=================================================

Convert cascade classifier outputs into quartile rankings with
within-quartile ranks for a neutral candidate evaluation system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import json


class CandidateRankingSystem:
    """Convert model predictions to candidate rankings."""
    
    def __init__(self):
        """Initialize ranking system."""
        # We'll work directly with prediction files instead of loading model
        pass
        
    def calculate_composite_score(self, probabilities):
        """
        Calculate a composite score from cascade probabilities.
        
        Higher weight for higher tiers to create proper ranking.
        """
        # Weights: Reject=0, Waitlist=1, Interview=2, Accept=3
        weights = np.array([0, 1, 2, 3])
        
        # Calculate weighted score (0-3 scale)
        weighted_score = np.sum(probabilities * weights, axis=1)
        
        # Convert to 0-100 scale
        composite_score = (weighted_score / 3.0) * 100
        
        return composite_score
    
    def assign_quartiles_and_ranks(self, scores):
        """Assign quartile and within-quartile ranks."""
        n = len(scores)
        
        # Calculate percentiles
        percentiles = pd.Series(scores).rank(pct=True) * 100
        
        # Assign quartiles (Q1 is top 25%)
        quartiles = pd.cut(percentiles, 
                          bins=[0, 25, 50, 75, 100],
                          labels=['Q4', 'Q3', 'Q2', 'Q1'],
                          include_lowest=True)
        
        # Calculate within-quartile ranks
        df_temp = pd.DataFrame({
            'score': scores,
            'percentile': percentiles,
            'quartile': quartiles
        })
        
        # Sort by score descending within each quartile
        within_quartile_ranks = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            mask = df_temp['quartile'] == q
            if mask.sum() > 0:
                # Rank within quartile (1 is best)
                q_ranks = df_temp.loc[mask, 'score'].rank(ascending=False, method='min')
                within_quartile_ranks.extend(list(zip(df_temp.loc[mask].index, q_ranks)))
        
        # Convert to series
        within_ranks = pd.Series(dict(within_quartile_ranks))
        
        return quartiles, percentiles, within_ranks
    
    def calculate_confidence(self, probabilities):
        """
        Calculate confidence based on probability distribution.
        
        High confidence = strong probability in one category
        Low confidence = probabilities spread across categories
        """
        # Calculate entropy as measure of uncertainty
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        
        # Max entropy for 4 classes is log(4) = 1.386
        max_entropy = np.log(4)
        
        # Convert to confidence (0-100)
        # Low entropy = high confidence
        confidence = (1 - entropy / max_entropy) * 100
        
        # Assign confidence levels
        confidence_levels = pd.cut(confidence,
                                 bins=[0, 60, 80, 100],
                                 labels=['Low', 'Medium', 'High'],
                                 include_lowest=True)
        
        return confidence, confidence_levels
    
    def generate_rankings(self, predictions_file):
        """Generate complete rankings from prediction file."""
        print("Generating candidate rankings...")
        
        # Load predictions
        df = pd.read_csv(predictions_file)
        
        # Extract probability columns
        prob_cols = ['reject_prob', 'waitlist_prob', 'interview_prob', 'accept_prob']
        probabilities = df[prob_cols].values
        
        # Calculate composite scores
        composite_scores = self.calculate_composite_score(probabilities)
        
        # Assign quartiles and ranks
        quartiles, percentiles, within_ranks = self.assign_quartiles_and_ranks(composite_scores)
        
        # Calculate confidence
        confidence_scores, confidence_levels = self.calculate_confidence(probabilities)
        
        # Count quartile sizes
        quartile_counts = quartiles.value_counts().to_dict()
        
        # Create rankings dataframe
        rankings = pd.DataFrame({
            'amcas_id': df['amcas_id'],
            'true_score': df['true_score'],
            'composite_score': composite_scores,
            'percentile': percentiles,
            'quartile': quartiles,
            'within_quartile_rank': within_ranks.values.astype(int),
            'quartile_size': [quartile_counts[q] for q in quartiles],
            'confidence_score': confidence_scores,
            'confidence_level': confidence_levels,
            'reject_prob': df['reject_prob'],
            'waitlist_prob': df['waitlist_prob'],
            'interview_prob': df['interview_prob'],
            'accept_prob': df['accept_prob']
        })
        
        # Sort by composite score descending
        rankings = rankings.sort_values('composite_score', ascending=False)
        
        # Add overall rank
        rankings['overall_rank'] = range(1, len(rankings) + 1)
        
        return rankings
    
    def create_strength_profile(self, applicant_features):
        """Create strength profile for visualization."""
        # This would analyze feature contributions
        # Placeholder for now - would need feature importance mapping
        profile = {
            'service_impact': np.random.randint(40, 95),
            'clinical_readiness': np.random.randint(40, 95),
            'academic_merit': np.random.randint(40, 95),
            'communication': np.random.randint(40, 95),
            'research_contribution': np.random.randint(40, 95)
        }
        return profile
    
    def generate_individual_report(self, rankings, amcas_id):
        """Generate individual applicant report."""
        applicant = rankings[rankings['amcas_id'] == amcas_id].iloc[0]
        
        report = {
            'amcas_id': amcas_id,
            'quartile': applicant['quartile'],
            'within_quartile_rank': int(applicant['within_quartile_rank']),
            'quartile_size': int(applicant['quartile_size']),
            'overall_percentile': round(applicant['percentile'], 1),
            'confidence_level': applicant['confidence_level'],
            'confidence_score': round(applicant['confidence_score'], 1),
            'strength_profile': self.create_strength_profile(None)
        }
        
        return report
    
    def generate_dashboard_summary(self, rankings):
        """Generate summary statistics for dashboard."""
        summary = {
            'total_applicants': int(len(rankings)),
            'quartile_distribution': {k: int(v) for k, v in rankings['quartile'].value_counts().to_dict().items()},
            'low_confidence_count': int((rankings['confidence_level'] == 'Low').sum()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Top candidates per quartile
        summary['top_per_quartile'] = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            q_data = rankings[rankings['quartile'] == q].head(5)
            records = []
            for _, row in q_data.iterrows():
                records.append({
                    'amcas_id': int(row['amcas_id']),
                    'percentile': float(row['percentile']),
                    'within_quartile_rank': int(row['within_quartile_rank'])
                })
            summary['top_per_quartile'][q] = records
        
        # Low confidence cases needing review
        low_conf = rankings[rankings['confidence_level'] == 'Low'].head(10)
        review_records = []
        for _, row in low_conf.iterrows():
            review_records.append({
                'amcas_id': int(row['amcas_id']),
                'quartile': str(row['quartile']),
                'confidence_score': float(row['confidence_score'])
            })
        summary['review_priorities'] = review_records
        
        return summary
    
    def export_rankings(self, rankings, output_dir="output"):
        """Export rankings in various formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full rankings CSV
        rankings_file = output_path / f"candidate_rankings_{timestamp}.csv"
        rankings.to_csv(rankings_file, index=False)
        print(f"Rankings saved to: {rankings_file}")
        
        # Summary JSON for dashboard
        summary = self.generate_dashboard_summary(rankings)
        summary_file = output_path / f"rankings_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")
        
        # Quartile-specific files
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            q_data = rankings[rankings['quartile'] == q].copy()
            q_data = q_data.sort_values('within_quartile_rank')
            q_file = output_path / f"{q}_candidates_{timestamp}.csv"
            q_data.to_csv(q_file, index=False)
            print(f"{q} candidates saved to: {q_file}")
        
        return rankings_file


def create_sample_visualization(rankings, amcas_id):
    """Create text-based visualization of individual ranking."""
    report = rankings[rankings['amcas_id'] == amcas_id].iloc[0]
    
    viz = f"""
┌─────────────────────────────────────────────────┐
│ Candidate Ranking Report                        │
│ AMCAS ID: {amcas_id}                          │
├─────────────────────────────────────────────────┤
│                                                 │
│ QUARTILE: {report['quartile']}                              │
│ Rank: {int(report['within_quartile_rank'])} of {int(report['quartile_size'])} in quartile          │
│                                                 │
│ Overall Percentile: {report['percentile']:.1f}              │
│ {'█' * int(report['percentile']/5)}{'░' * (20 - int(report['percentile']/5))}            │
│                                                 │
│ Confidence: {report['confidence_level']} ({report['confidence_score']:.1f})              │
│                                                 │
│ Model Assessment Breakdown:                     │
│ • Reject probability:    {report['reject_prob']:.1%}        │
│ • Waitlist probability:  {report['waitlist_prob']:.1%}      │
│ • Interview probability: {report['interview_prob']:.1%}     │
│ • Accept probability:    {report['accept_prob']:.1%}        │
│                                                 │
│ True Score (for validation): {int(report['true_score'])}            │
└─────────────────────────────────────────────────┘
"""
    return viz


def main():
    """Main execution function."""
    print("="*80)
    print("CANDIDATE RANKING SYSTEM")
    print("Converting model predictions to quartile rankings")
    print("="*80)
    
    # Initialize ranking system
    ranker = CandidateRankingSystem()
    
    # Find most recent cascade predictions
    pred_files = list(Path(".").glob("cascade_predictions_*.csv"))
    if not pred_files:
        print("No cascade predictions found. Please run train_cascading_classifier.py first.")
        return
    
    latest_predictions = sorted(pred_files)[-1]
    print(f"\nUsing predictions from: {latest_predictions}")
    
    # Generate rankings
    rankings = ranker.generate_rankings(latest_predictions)
    
    # Display summary
    print("\nRanking Summary:")
    print(f"Total candidates: {len(rankings)}")
    print("\nQuartile distribution:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        count = (rankings['quartile'] == q).sum()
        pct = count / len(rankings) * 100
        print(f"  {q}: {count} ({pct:.1f}%)")
    
    # Show top 5 candidates
    print("\nTop 5 Candidates:")
    top5 = rankings.head(5)
    for _, row in top5.iterrows():
        print(f"  {row['overall_rank']}. AMCAS {row['amcas_id']} - "
              f"{row['percentile']:.1f}%ile (True score: {row['true_score']})")
    
    # Show sample individual report
    sample_id = rankings.iloc[10]['amcas_id']  # 11th ranked candidate
    print(f"\nSample Individual Report (Rank 11):")
    print(create_sample_visualization(rankings, sample_id))
    
    # Export all rankings
    output_file = ranker.export_rankings(rankings)
    
    print("\n" + "="*80)
    print("RANKING GENERATION COMPLETE!")
    print(f"Check the output/ directory for all ranking files")
    print("="*80)


if __name__ == "__main__":
    main()