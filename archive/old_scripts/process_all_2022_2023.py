"""
Process All 2022-2023 Applicants Through Azure OpenAI
=====================================================

This script automatically processes all applicants without user interaction.
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
    print("Medical Admissions LLM Scoring Pipeline - Full Processing")
    print("="*60)
    
    # Step 1: Initialize Azure connection
    print("\n1. Initializing Azure OpenAI connection...")
    try:
        azure = AzureOpenAIConnection()
        print("✅ Azure connection successful")
    except Exception as e:
        print(f"❌ Failed to initialize Azure: {e}")
        return
    
    # Step 2: Extract unstructured data
    print("\n2. Extracting unstructured data from 2022-2023...")
    extractor = UnstructuredDataExtractor("data")
    
    # Get data for 2022 and 2023
    data_2022 = extractor.extract_year_data(2022)
    data_2023 = extractor.extract_year_data(2023)
    
    # Combine all applicants
    all_applicants = {**data_2022, **data_2023}
    print(f"✅ Extracted data for {len(all_applicants)} applicants")
    
    # Check data completeness
    complete_count = sum(1 for content in all_applicants.values() 
                        if all(key in content for key in ['personal_statement', 'secondary_essays', 'experiences']))
    print(f"   - Complete data: {complete_count}")
    print(f"   - Partial data: {len(all_applicants) - complete_count}")
    
    # Step 3: Process through Azure OpenAI
    print("\n3. Processing applicants through Azure OpenAI...")
    print(f"   Total applicants: {len(all_applicants)}")
    print(f"   Estimated time: {len(all_applicants) * 2 / 60:.1f} minutes")
    print(f"   Estimated cost: ${len(all_applicants) * 0.04:.2f}")
    print("\n   Starting automatic processing...")
    
    # Process in batches
    processor = BatchProcessor(azure, batch_size=10, delay_between_batches=1.0)
    
    # Process all applicants
    llm_scores_df = processor.process_all_applicants(all_applicants)
    
    # Step 4: Save results
    print("\n4. Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"llm_scores_2022_2023_{timestamp}.csv"
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
    
    print("\n✅ Processing complete!")
    print(f"   Processed: {len(llm_scores_df)} applicants")
    print(f"   Output file: {output_file}")
    
    return output_file

if __name__ == "__main__":
    output_file = main()