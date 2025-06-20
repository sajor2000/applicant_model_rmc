#!/usr/bin/env python3
"""Generate synthetic application data for testing purposes."""

import pandas as pd
import numpy as np
import random
import argparse
from pathlib import Path
from datetime import datetime

# Synthetic essay templates
ESSAY_TEMPLATES = [
    "Synthetic essay demonstrating {motivation} for medicine. Discusses {experience_type} experiences including {specific_activity}. Shows understanding of {healthcare_aspect} and commitment to {career_goal}. Demonstrates {personal_quality} through {challenge_type}.",
    "Example essay from {background} perspective. Previous experience in {field} provided {insight_type} understanding. Extensive {activity_type} with {hours} hours of involvement. Shows {quality1} and {quality2} through various experiences.",
    "Artificial narrative focusing on {main_theme}. Describes journey from {starting_point} to medical aspirations. Highlights {achievement} and lessons learned from {experience}. Expresses interest in {specialty} based on {reason}.",
]

MOTIVATIONS = [
    "strong commitment", "deep passion", "genuine interest", 
    "longstanding dedication", "evolving understanding"
]

EXPERIENCE_TYPES = [
    "clinical", "research", "volunteer", "leadership", "community service"
]

ACTIVITIES = [
    "hospital volunteering", "clinic shadowing", "research projects",
    "community health initiatives", "medical interpretation", "health education"
]

HEALTHCARE_ASPECTS = [
    "patient care", "health equity", "medical innovation", 
    "preventive medicine", "healthcare access"
]

CAREER_GOALS = [
    "primary care", "specialty medicine", "academic medicine",
    "global health", "underserved populations"
]

PERSONAL_QUALITIES = [
    "resilience", "empathy", "leadership", "cultural competence",
    "analytical thinking", "communication skills"
]

BACKGROUNDS = [
    "first-generation student", "career-changer", "international student",
    "non-traditional", "research-focused", "service-oriented"
]


def generate_synthetic_essay():
    """Generate a synthetic essay description."""
    template = random.choice(ESSAY_TEMPLATES)
    
    essay = template.format(
        motivation=random.choice(MOTIVATIONS),
        experience_type=random.choice(EXPERIENCE_TYPES),
        specific_activity=random.choice(ACTIVITIES),
        healthcare_aspect=random.choice(HEALTHCARE_ASPECTS),
        career_goal=random.choice(CAREER_GOALS),
        personal_quality=random.choice(PERSONAL_QUALITIES),
        challenge_type=random.choice(["personal challenges", "academic obstacles", "life experiences"]),
        background=random.choice(BACKGROUNDS),
        field=random.choice(["public health", "research", "education", "business", "emergency services"]),
        insight_type=random.choice(["systematic", "clinical", "population-level", "hands-on"]),
        activity_type=random.choice(["volunteering", "research", "clinical work", "teaching"]),
        hours=random.randint(100, 2000),
        quality1=random.choice(PERSONAL_QUALITIES),
        quality2=random.choice(PERSONAL_QUALITIES),
        main_theme=random.choice(["service to others", "scientific discovery", "health advocacy", "personal growth"]),
        starting_point=random.choice(["early exposure", "transformative experience", "academic interest", "family influence"]),
        achievement=random.choice(["research publication", "program development", "leadership role", "community impact"]),
        experience=random.choice(["clinical volunteering", "research project", "shadowing", "personal challenge"]),
        specialty=random.choice(["internal medicine", "pediatrics", "surgery", "psychiatry", "family medicine"]),
        reason=random.choice(["clinical experiences", "personal connection", "mentor influence", "population need"])
    )
    
    return f"SYNTHETIC DATA: {essay}"


def generate_synthetic_applications(count=10):
    """Generate synthetic application data."""
    
    applications = []
    
    for i in range(count):
        # Generate synthetic data with realistic distributions
        service_rating = np.random.choice([1, 2, 3, 4], p=[0.05, 0.20, 0.45, 0.30])
        
        # Clinical hours - correlated with service rating
        base_clinical = service_rating * 300 + np.random.randint(-200, 400)
        healthcare_hours = max(0, base_clinical + np.random.randint(-100, 500))
        
        # Other experiences
        research_hours = np.random.choice([0, 100, 500, 1000, 2000], p=[0.1, 0.3, 0.3, 0.2, 0.1])
        volunteer_med = np.random.randint(0, 800)
        volunteer_non_med = np.random.randint(0, 600)
        
        # Demographics
        age = np.random.randint(22, 32)
        gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.45, 0.50, 0.05])
        citizenship = np.random.choice(['US_Citizen', 'Permanent_Resident', 'International'], p=[0.85, 0.10, 0.05])
        first_gen = np.random.choice([0, 1], p=[0.75, 0.25])
        
        application = {
            'amcas_id': f'FAKE{i+1:03d}',
            'service_rating_numerical': service_rating,
            'healthcare_total_hours': healthcare_hours,
            'exp_hour_research': research_hours,
            'exp_hour_volunteer_med': volunteer_med,
            'exp_hour_volunteer_non_med': volunteer_non_med,
            'age': age,
            'gender': gender,
            'citizenship': citizenship,
            'first_generation_ind': first_gen,
            'essay_text': generate_synthetic_essay()
        }
        
        applications.append(application)
    
    return pd.DataFrame(applications)


def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(description='Generate synthetic application data')
    parser.add_argument('--count', type=int, default=10, help='Number of applications to generate')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Generating {args.count} synthetic applications...")
    
    # Generate data
    df = generate_synthetic_applications(args.count)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f'synthetic_applications_{timestamp}.csv')
    
    # Save to file
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} synthetic applications to: {output_path}")
    
    # Display sample
    print("\nSample of generated data:")
    print(df.head())
    
    print("\n⚠️  REMINDER: This is SYNTHETIC DATA for testing only!")
    print("Never include real applicant information in the repository.")


if __name__ == "__main__":
    main()