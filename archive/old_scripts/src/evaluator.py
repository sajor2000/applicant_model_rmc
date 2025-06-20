import streamlit as st
import openai
import json

def evaluate_applicant(applicant_data, essay_text):
    """
    Evaluates a single applicant using the OpenAI API.

    Args:
        applicant_data (pd.Series): A row from the applicant DataFrame.
        essay_text (str): The applicant's full essay text.

    Returns:
        dict: A dictionary containing the AI's structured evaluation.
    """
    try:
        # 1. Securely get the API key from Streamlit secrets
        openai.api_key = st.secrets["openai_api_key"]

        # 2. Construct a detailed prompt for the AI
        prompt = f"""
        You are an expert medical school admissions reviewer. Your task is to provide a holistic and unbiased evaluation of a candidate based on their provided data and personal essay. 

        **Candidate Data:**
        - MCAT Score: {applicant_data.get('MCAT', 'N/A')}
        - GPA: {applicant_data.get('GPA', 'N/A')}
        - Research Hours: {applicant_data.get('Research Hours', 'N/A')}
        - Clinical Hours: {applicant_data.get('Clinical Hours', 'N/A')}
        - Volunteering Hours: {applicant_data.get('Volunteering Hours', 'N/A')}

        **Candidate Essay:**
        ---begin_essay---
        {essay_text}
        ---end_essay---

        **Evaluation Task:**
        Based on all the information above, please provide a structured JSON object with your evaluation. The JSON object must contain the following keys:
        - "overall_score": A score from 1 to 100, representing the applicant's overall strength.
        - "academic_strength": A score from 1 to 10, assessing their academic profile (GPA, MCAT).
        - "experience_strength": A score from 1 to 10, assessing their relevant experiences (research, clinical, volunteering).
        - "essay_strength": A score from 1 to 10, assessing the quality, clarity, and impact of their essay.
        - "summary": A 2-3 sentence qualitative summary of the applicant's profile, highlighting their key strengths and weaknesses.
        - "red_flags": A brief description of any potential red flags or areas of concern, or "None" if there are none.

        Do not include any text outside of the JSON object.
        """

        # 3. Make the API call
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview", # Or another suitable model
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful medical school admissions assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        # 4. Parse and return the JSON response
        evaluation = json.loads(response.choices[0].message.content)
        return evaluation

    except openai.APIError as e:
        st.error(f"OpenAI API Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during evaluation: {e}")
        return None
