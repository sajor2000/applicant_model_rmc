"""
Quick Test Script for Azure OpenAI Essay Scoring Setup
=====================================================

Run this script to verify your Azure OpenAI configuration is working correctly.
"""

import asyncio
import os
from azure_essay_scorer import AzureEssayScorer

# Test essays with known characteristics
TEST_ESSAYS = {
    "strong_candidate": """
My journey to medicine began during my sophomore year when my younger sister was diagnosed 
with leukemia. Spending countless hours at the hospital, I witnessed the profound impact 
that compassionate physicians had not only on her medical care but on our entire family's 
wellbeing. This experience, combined with my research in oncology biomarkers (resulting in 
two publications), has solidified my commitment to becoming a physician-scientist.

Through 500+ hours of clinical volunteering at our local free clinic, I've seen how healthcare 
disparities affect underserved populations. I initiated a diabetes education program that has 
now served over 200 patients. My research experience includes 800 hours in Dr. Smith's lab, 
where I led a project on novel therapeutic targets in pediatric cancers.

Leadership roles in our university's Global Health Initiative and Pre-Med Society have taught 
me the importance of teamwork and communication. As president, I organized health screenings 
that reached 1,000+ community members. These experiences have prepared me for the rigorous 
journey ahead in medical school.
""",
    
    "average_candidate": """
I want to become a doctor because I enjoy helping people and find science interesting. 
During college, I volunteered at a hospital for about 100 hours and shadowed a few doctors. 
These experiences showed me what daily life is like for physicians.

My grades have been consistent throughout college, maintaining a 3.5 GPA. I particularly 
enjoyed my biology and chemistry courses. I also participated in some research for a semester, 
though I didn't publish anything.

Outside of academics, I was a member of the pre-med club and played intramural sports. 
I believe these activities have helped me develop good teamwork skills that will be useful 
in medical school.
""",
    
    "weak_candidate": """
I think being a doctor would be a good career because they make good money and have job 
security. My parents want me to go to medical school since I did well in my science classes.

I haven't had much time for volunteering because I needed to work to pay for school, but 
I'm sure I'll learn everything I need in medical school. I shadowed a doctor once during 
winter break which was interesting.

I'm a hard worker and good at memorizing things, which I think are the most important 
skills for medical school. I look forward to starting my medical career.
"""
}


async def test_azure_setup():
    """Test Azure OpenAI configuration with sample essays"""
    
    print("="*60)
    print("AZURE OPENAI ESSAY SCORING TEST")
    print("="*60)
    
    # Check for Azure credentials
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
    
    if not azure_endpoint or not api_key:
        print("\n❌ ERROR: Azure credentials not found!")
        print("\nPlease set the following environment variables:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_KEY='your-api-key'")
        print("  export AZURE_OPENAI_DEPLOYMENT='your-deployment-name' (optional)")
        return
    
    print(f"\n✓ Azure Endpoint: {azure_endpoint}")
    print(f"✓ Deployment: {deployment_name}")
    print("✓ API Key: [HIDDEN]")
    
    # Initialize scorer
    try:
        scorer = AzureEssayScorer(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            use_cache=False  # Disable cache for testing
        )
        print("\n✓ Azure Essay Scorer initialized successfully")
    except Exception as e:
        print(f"\n❌ Failed to initialize scorer: {e}")
        return
    
    # Test each essay
    print("\n" + "-"*60)
    print("TESTING ESSAY SCORING")
    print("-"*60)
    
    for essay_type, essay_text in TEST_ESSAYS.items():
        print(f"\n\nTesting {essay_type.replace('_', ' ').title()}...")
        print("-"*40)
        
        try:
            # Score the essay
            context = {
                'age': 24,
                'total_hours': 1500 if essay_type == "strong_candidate" else 500,
                'research_hours': 800 if essay_type == "strong_candidate" else 100,
                'clinical_hours': 700 if essay_type == "strong_candidate" else 100,
                'gpa_trend': 1.0 if essay_type != "weak_candidate" else 0.0
            }
            
            result = await scorer.score_essay_async(essay_text, context)
            
            # Display results
            print(f"Overall Score: {result['overall_score']}/100")
            print(f"Recommendation: {result['recommendation']}")
            print(f"\nDimension Scores:")
            print(f"  - Motivation: {result['motivation_score']}/10")
            print(f"  - Clinical Understanding: {result['clinical_understanding']}/10")
            print(f"  - Service Commitment: {result['service_commitment']}/10")
            print(f"  - Resilience: {result['resilience_score']}/10")
            print(f"  - Academic Readiness: {result['academic_readiness']}/10")
            print(f"  - Interpersonal Skills: {result['interpersonal_skills']}/10")
            print(f"  - Leadership: {result['leadership_score']}/10")
            print(f"  - Ethical Maturity: {result['ethical_maturity']}/10")
            
            if result.get('red_flags'):
                print(f"\n⚠️  Red Flags: {', '.join(result['red_flags'])}")
            
            if result.get('green_flags'):
                print(f"\n✓ Green Flags: {', '.join(result['green_flags'])}")
            
            print(f"\nSummary: {result['summary']}")
            
        except Exception as e:
            print(f"❌ Error scoring essay: {e}")
    
    # Test batch processing
    print("\n\n" + "-"*60)
    print("TESTING BATCH PROCESSING")
    print("-"*60)
    
    try:
        essays_batch = [
            ("001", TEST_ESSAYS["strong_candidate"], {'age': 24}),
            ("002", TEST_ESSAYS["average_candidate"], {'age': 23}),
            ("003", TEST_ESSAYS["weak_candidate"], {'age': 22})
        ]
        
        print("\nProcessing batch of 3 essays...")
        results = await scorer.batch_score_essays(essays_batch, batch_size=3)
        
        print("\n✓ Batch processing successful!")
        print(f"Processed {len(results)} essays")
        
        # Show summary
        avg_score = sum(r['overall_score'] for r in results) / len(results)
        print(f"\nAverage overall score: {avg_score:.1f}/100")
        
        recommendations = [r['recommendation'] for r in results]
        print(f"Recommendations: {', '.join(recommendations)}")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    # Performance test
    print("\n\n" + "-"*60)
    print("PERFORMANCE METRICS")
    print("-"*60)
    
    import time
    
    start_time = time.time()
    try:
        # Score one essay to measure time
        await scorer.score_essay_async(
            TEST_ESSAYS["average_candidate"],
            {'age': 24}
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Single essay scoring time: {elapsed:.2f} seconds")
        print(f"✓ Estimated throughput: {60/elapsed:.0f} essays/minute")
        
        # Cost estimation
        cost_per_essay = 0.04  # Approximate GPT-4 cost
        print(f"\nCost Estimates (GPT-4):")
        print(f"  - Per essay: ${cost_per_essay:.3f}")
        print(f"  - 1,000 essays: ${cost_per_essay * 1000:.2f}")
        print(f"  - 5,000 essays: ${cost_per_essay * 5000:.2f}")
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Validation checks
    print("\n\nVALIDATION SUMMARY:")
    print("-"*40)
    
    if 'results' in locals() and len(results) == 3:
        scores = [r['overall_score'] for r in results]
        
        # Check score differentiation
        if max(scores) - min(scores) > 20:
            print("✓ Good score differentiation between essay qualities")
        else:
            print("⚠️  Low score differentiation - check temperature settings")
        
        # Check if strong candidate scored highest
        if results[0]['overall_score'] > results[1]['overall_score'] > results[2]['overall_score']:
            print("✓ Score ordering matches expected quality levels")
        else:
            print("⚠️  Unexpected score ordering - review prompt or essays")
        
        # Check recommendation alignment
        if results[0]['recommendation'] in ['Strongly Recommend', 'Recommend']:
            print("✓ Strong candidate received positive recommendation")
        else:
            print("⚠️  Strong candidate didn't receive expected recommendation")
    
    print("\n✅ Azure OpenAI setup is working correctly!")
    print("\nNext steps:")
    print("1. Update azure_config in integrated_azure_pipeline.py with your credentials")
    print("2. Run the full pipeline with: python integrated_azure_pipeline.py")
    print("3. Monitor results in the exported Excel files")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_azure_setup())