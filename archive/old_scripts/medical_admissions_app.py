import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Medical Admissions AI Assistant")

# --- Sidebar --- #
st.sidebar.title("Controls")
st.sidebar.info(
    "**Welcome to the Medical Admissions AI Assistant!**\n\n" 
    "1. Upload a CSV/Excel file with applicant data.\n" 
    "2. Upload the corresponding applicant essays in PDF format.\n" 
    "3. Click 'Evaluate Applicants' to start the analysis."
)

# --- Main Application --- #
st.title("Medical Admissions AI Assistant")
st.write("This tool uses AI to analyze applicant data and essays to provide a holistic evaluation.")

# --- Step 1: File Uploaders --- #
st.header("1. Upload Applicant Files")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Data")
    applicant_data_file = st.file_uploader(
        "Upload a CSV or Excel file with applicant information.", 
        type=['csv', 'xlsx']
    )

with col2:
    st.subheader("Applicant Essays")
    applicant_essay_files = st.file_uploader(
        "Upload all applicant essays in PDF format.", 
        type=['pdf'], 
        accept_multiple_files=True
    )

# --- Step 2: Evaluation Trigger --- #
st.header("2. Start Evaluation")
from src.evaluator import evaluate_applicant

if st.button("Evaluate Applicants", type="primary"):
    if applicant_data_file is not None and applicant_essay_files:
        with st.spinner('Processing and evaluating... Please wait.'):
            try:
                # --- File Processing Logic ---
                if applicant_data_file.name.endswith('.csv'):
                    df = pd.read_csv(applicant_data_file)
                else:
                    df = pd.read_excel(applicant_data_file)
                
                essays = {}
                from PyPDF2 import PdfReader
                import io

                for uploaded_file in applicant_essay_files:
                    applicant_id = uploaded_file.name.split('.')[0]
                    try:
                        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                        text = "".join(page.extract_text() or "" for page in reader.pages)
                        essays[applicant_id] = text
                    except Exception as e:
                        st.warning(f"Could not read {uploaded_file.name}: {e}")
                
                # --- AI Evaluation Step ---
                evaluations = []
                total_applicants = len(df)
                progress_bar = st.progress(0, text="Evaluating applicants...")

                if 'ApplicantID' not in df.columns:
                    st.error("Your data must contain an 'ApplicantID' column to match essays.")
                    st.stop()
                df['ApplicantID'] = df['ApplicantID'].astype(str)

                for index, row in df.iterrows():
                    applicant_id = row['ApplicantID']
                    if applicant_id in essays:
                        evaluation_result = evaluate_applicant(row, essays[applicant_id])
                        if evaluation_result:
                            evaluations.append({**{'ApplicantID': applicant_id}, **evaluation_result})
                    else:
                        st.warning(f"No essay found for ApplicantID: {applicant_id}. Skipping.")
                    
                    progress_bar.progress((index + 1) / total_applicants, text=f"Evaluating applicant {index + 1}/{total_applicants}")

                # --- Merge and Display Final Results ---
                if evaluations:
                    eval_df = pd.DataFrame(evaluations)
                    final_df = pd.merge(df, eval_df, on='ApplicantID', how='left')
                    st.session_state['final_df'] = final_df

                    results_container.empty()
                    with results_container.container():
                        st.success("Evaluation complete!")
                        st.header("Evaluation Results")
                        st.dataframe(final_df)

                        # --- Add Download Button ---
                        @st.cache_data
                        def convert_df_to_csv(df_to_convert):
                            return df_to_convert.to_csv(index=False).encode('utf-8')

                        csv = convert_df_to_csv(final_df)
                        
                        st.download_button(
                           label="Download Results as CSV",
                           data=csv,
                           file_name='applicant_evaluations.csv',
                           mime='text/csv',
                        )
                else:
                    results_container.error("No applicants could be evaluated.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
    else:
        st.error("Please upload both applicant data and essay files before evaluating.")

# --- Results Display Container --- #
results_container = st.container()
if 'final_df' not in st.session_state:
    with results_container:
        st.info("Results will be displayed here after evaluation is complete.")

