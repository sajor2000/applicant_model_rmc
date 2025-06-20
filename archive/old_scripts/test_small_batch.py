"""
Test processing a small batch of actual applicants
=================================================
"""

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from step1_azure_connection import AzureOpenAIConnection
from step2_extract_unstructured_data import UnstructuredDataExtractor
from step3_batch_processing import BatchProcessor, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import pandas as pd

print("Testing Medical Admissions LLM Pipeline")
print("="*50)

# Initialize components
print("\n1. Initializing Azure connection...")
azure = AzureOpenAIConnection()

print("\n2. Extracting essays for first 3 applicants from 2022...")
extractor = UnstructuredDataExtractor("data")
data_2022 = extractor.extract_year_data(2022)

# Get first 3 applicants
test_data = dict(list(data_2022.items())[:3])
print(f"   Found {len(test_data)} applicants with essays")

# Show what we have for each applicant
for amcas_id, content in test_data.items():
    print(f"\n   {amcas_id}:")
    if 'personal_statement' in content:
        print(f"     - Personal statement: {len(content['personal_statement'])} chars")
    if 'secondary_essays' in content:
        print(f"     - Secondary essays: {len(content['secondary_essays'])} essays")
    if 'experiences' in content:
        print(f"     - Experiences: {len(content['experiences'])} chars")

# Process through LLM
print("\n3. Processing through Azure OpenAI...")
processor = BatchProcessor(azure, batch_size=3)
results_df = processor.process_all_applicants(test_data)

print("\n4. Results:")
print(results_df)

# Save test results
output_file = "test_batch_results.csv"
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Show summary statistics
print("\nScore Summary:")
score_cols = [col for col in results_df.columns if col.startswith('llm_') and col != 'llm_red_flag_count' and col != 'llm_green_flag_count']
print(results_df[score_cols].describe())