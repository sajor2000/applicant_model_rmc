"""
Process 2024 Test Data Through Azure OpenAI
===========================================

This script processes all 2024 applicants through Azure OpenAI to generate
LLM scores for the holdout test set.
"""

from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

from step1_azure_connection import AzureOpenAIConnection
from step2_extract_unstructured_data import UnstructuredDataExtractor
from step3_batch_processing import BatchProcessor, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def main():
    print("Medical Admissions LLM Scoring Pipeline - 2024 Test Set")
    print("="*60)
    
    # Step 1: Initialize Azure connection
    print("\n1. Initializing Azure OpenAI connection...")
    try:
        azure = AzureOpenAIConnection()
        print("✅ Azure connection successful")
    except Exception as e:
        print(f"❌ Failed to initialize Azure: {e}")
        return
    
    # Step 2: Extract unstructured data for 2024
    print("\n2. Extracting unstructured data from 2024...")
    extractor = UnstructuredDataExtractor("data")
    
    # Get data for 2024
    data_2024 = extractor.extract_year_data(2024)
    print(f"✅ Extracted data for {len(data_2024)} applicants")
    
    # Check data completeness
    complete_count = sum(1 for content in data_2024.values() 
                        if all(key in content for key in ['personal_statement', 'secondary_essays', 'experiences']))
    print(f"   - Complete data: {complete_count}")
    print(f"   - Partial data: {len(data_2024) - complete_count}")
    
    # Step 3: Process through Azure OpenAI
    print("\n3. Processing 2024 applicants through Azure OpenAI...")
    print(f"   Total applicants: {len(data_2024)}")
    print(f"   Estimated time: {len(data_2024) * 2 / 60:.1f} minutes")
    print(f"   Estimated cost: ${len(data_2024) * 0.04:.2f}")
    print("\n   Starting automatic processing...")
    
    # Process in batches
    processor = BatchProcessor(azure, batch_size=10, delay_between_batches=1.0)
    
    # Process all applicants
    llm_scores_df = processor.process_all_applicants(data_2024)
    
    # Step 4: Save results
    print("\n4. Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"llm_scores_2024_{timestamp}.csv"
    llm_scores_df.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")
    
    # Show summary statistics
    print("\n5. Summary Statistics:")
    score_cols = [col for col in llm_scores_df.columns 
                 if col.startswith('llm_') and col not in ['llm_red_flag_count', 'llm_green_flag_count']]
    
    if score_cols:
        print("\nScore distributions (0-10 scale):")
        for col in sorted(score_cols):
            mean = llm_scores_df[col].mean()
            std = llm_scores_df[col].std()
            print(f"   {col}: {mean:.2f} ± {std:.2f}")
        
        print(f"\nOverall essay scores (0-100): {llm_scores_df['llm_overall_essay_score'].mean():.1f} ± {llm_scores_df['llm_overall_essay_score'].std():.1f}")
        print(f"Average red flags: {llm_scores_df['llm_red_flag_count'].mean():.1f}")
        print(f"Average green flags: {llm_scores_df['llm_green_flag_count'].mean():.1f}")
    
    print("\n✅ 2024 test set processing complete!")
    print(f"   Processed: {len(llm_scores_df)} applicants")
    print(f"   Output file: {output_file}")
    
    # Also save just the filename for easy reference
    with open("latest_2024_llm_scores.txt", "w") as f:
        f.write(output_file)
    
    return output_file

if __name__ == "__main__":
    output_file = main()