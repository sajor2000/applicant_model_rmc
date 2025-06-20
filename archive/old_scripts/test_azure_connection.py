"""
Test Azure OpenAI Connection
============================

Run this to verify your Azure setup is working correctly.
"""

import os
from dotenv import load_dotenv
from step1_azure_connection import AzureOpenAIConnection

# Load environment variables from .env file
load_dotenv()

print("Testing Azure OpenAI Connection")
print("="*50)

# Check environment variables
print("\nEnvironment Variables:")
print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'NOT SET')}")
print(f"Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'NOT SET')}")
print(f"API Key: {'SET' if os.getenv('AZURE_OPENAI_API_KEY') else 'NOT SET'}")

# Test connection
try:
    print("\nInitializing connection...")
    azure = AzureOpenAIConnection()
    
    print("Testing connection...")
    if azure.test_connection():
        print("✅ Connection successful!")
        
        # Test with a sample essay evaluation
        print("\nTesting essay evaluation...")
        test_scores = azure.evaluate_essays(
            personal_statement="This is a test personal statement about my passion for medicine.",
            secondary_essays={"Challenge": "I overcame adversity by..."},
            experiences="I volunteered at a hospital for 200 hours.",
            system_prompt="You are a medical admissions evaluator. Return a JSON with numeric scores.",
            user_prompt_template="Evaluate this application:\n{personal_statement}\n{secondary_essays}\n{experiences}"
        )
        
        print("\nSample scores received:")
        for key, value in test_scores.items():
            if key.startswith('llm_'):
                print(f"  {key}: {value}")
                
    else:
        print("❌ Connection failed!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you've added your API key to the .env file")
    print("2. The key should replace 'YOUR-KEY-HERE' in the .env file")
    print("3. Get the key from Azure Portal > your resource > Keys and Endpoint")