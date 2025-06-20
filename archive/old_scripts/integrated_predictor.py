import pandas as pd
import numpy as np
import joblib
import openai
import os
import PyPDF2
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class IntegratedAdmissionsPredictor:
    """
    Integrates a trained structured data model with LLM text analysis for admissions predictions.
    """
    
    def __init__(self, 
                 model_path: str = 'models/four_tier_high_confidence_model.pkl',
                 openai_api_key: str = None,
                 essay_source_config: Optional[Dict] = None):
        
        try:
            self.model_data = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Ensure 'four_tier_classifier.py' has been run.")
            raise
            
        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names'] # These are the features the model was trained on
        self.class_names = self.model_data['class_names']
        self.scaler = self.model_data.get('scaler') # Loaded scaler from training

        if not self.scaler:
            print("Warning: Scaler not found in model file. Numerical features will not be scaled.")

        self.openai_api_key = openai_api_key
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            print("Warning: OpenAI API key not provided. LLM insights will be disabled.")

        # OpenAI Model Configuration from the guide
        self.MODEL_CONFIG = {
            'temperature': 0.2,
            'max_tokens_completion': 50,  # We expect only comma-separated values as outputs
            'max_tokens_essay_input': 4000,  # Truncate essays to this length for token limits
            'chat_model': 'gpt-4',  # Model for essay analysis; use the most capable available
            'embedding_model': 'text-embedding-3-large',  # For embedding-based similarity
        }
        self.essay_source_config = essay_source_config if essay_source_config is not None else {}
        # Example essay_source_config structure:
        # self.essay_source_config = {
        #     'personal_statement': {'type': 'excel', 'file_path': 'path/to/9. Personal Statement.xlsx', 'id_column': 'AMCAS_ID', 'text_column': 'Essay_Text'},
        #     'secondary_app': {'type': 'pdf_path_column', 'column_name': 'Secondary_PDF_Path'},
        #     'additional_essay': {'type': 'direct_column', 'column_name': 'Additional_Essay_Text'}
        # }

    def _handle_missing_data_and_ids_single(self, applicant_series: pd.Series) -> pd.Series:
        """Handles ID standardization and missing data for a single applicant Series."""
        # AMCAS ID handling is mostly for batch processing; for single predict, assume ID is passed if needed
        # For feature preparation, focus on data imputation
        data = applicant_series.copy()

        numeric_cols_to_fill_zero = [
            'Exp_Hour_Total', 'Exp_Hour_Research', 'Exp_Hour_Volunteer_Med', 
            'Exp_Hour_Shadowing', 'Comm_Service_Total_Hours', 'HealthCare_Total_Hours',
            'Age', 'Num_Dependents', 'Service Rating (Numerical)'
        ]
        for col in numeric_cols_to_fill_zero:
            data[col] = pd.to_numeric(data.get(col), errors='coerce')
            if pd.isna(data[col]):
                data[col] = 0

        gpa_trend_cols = ['Total_GPA_Trend', 'BCPM_GPA_Trend']
        for col in gpa_trend_cols:
            value = data.get(col)
            if isinstance(value, str) and value.upper() == 'NULL':
                data[col] = 0
            else:
                data[col] = pd.to_numeric(value, errors='coerce')
            if pd.isna(data[col]):
                data[col] = 0
        return data

    def _engineer_features_single(self, applicant_series: pd.Series) -> pd.Series:
        """Creates derived features for a single applicant Series."""
        data = applicant_series.copy()
        epsilon = 1e-6

        # Ensure base numeric features are numeric before operations
        exp_total = pd.to_numeric(data.get('Exp_Hour_Total', 0), errors='coerce')
        exp_research = pd.to_numeric(data.get('Exp_Hour_Research', 0), errors='coerce')
        exp_volunteer_med = pd.to_numeric(data.get('Exp_Hour_Volunteer_Med', 0), errors='coerce')
        exp_shadowing = pd.to_numeric(data.get('Exp_Hour_Shadowing', 0), errors='coerce')
        service_rating_num = pd.to_numeric(data.get('Service Rating (Numerical)', 0), errors='coerce')
        comm_service_hours = pd.to_numeric(data.get('Comm_Service_Total_Hours', 0), errors='coerce')
        total_gpa_trend = pd.to_numeric(data.get('Total_GPA_Trend', 0), errors='coerce')

        # Fill NaNs that might have resulted from coerce with 0
        exp_total = 0 if pd.isna(exp_total) else exp_total
        exp_research = 0 if pd.isna(exp_research) else exp_research
        exp_volunteer_med = 0 if pd.isna(exp_volunteer_med) else exp_volunteer_med
        exp_shadowing = 0 if pd.isna(exp_shadowing) else exp_shadowing
        service_rating_num = 0 if pd.isna(service_rating_num) else service_rating_num
        comm_service_hours = 0 if pd.isna(comm_service_hours) else comm_service_hours
        total_gpa_trend = 0 if pd.isna(total_gpa_trend) else total_gpa_trend

        data['research_intensity'] = exp_research / (exp_total + epsilon)
        data['clinical_intensity'] = (exp_volunteer_med + exp_shadowing) / (exp_total + epsilon)
        data['experience_balance'] = exp_research / (exp_volunteer_med + exp_shadowing + epsilon)
        
        comm_service_hours_positive = max(comm_service_hours, 0)
        data['service_commitment'] = service_rating_num * np.log(comm_service_hours_positive + 1)
        
        disadvantaged_ind_val = 1 if str(data.get('Disadvantanged_Ind', 'No')).lower() == 'yes' else 0
        data['adversity_overcome'] = disadvantaged_ind_val * total_gpa_trend
        
        return data

    def prepare_structured_features(self, applicant_data: pd.Series) -> np.ndarray:
        """Prepares the full feature vector for a single applicant from raw data."""
        # Process a copy of the input series
        processed_data = applicant_data.copy()

        # 1. Handle missing data for original features
        processed_data = self._handle_missing_data_and_ids_single(processed_data)

        # 2. Engineer derived features
        processed_data = self._engineer_features_single(processed_data)

        # 3. Convert binary indicator features
        binary_cols = ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']
        binary_conversion_map = {'Yes': 1, 'yes': 1}
        for col in binary_cols:
            processed_data[col] = binary_conversion_map.get(str(processed_data.get(col, 'No')).lower(), 0)

        # 4. Assemble feature vector in the correct order
        feature_vector = []
        for feature_name in self.feature_names:
            value = processed_data.get(feature_name, 0) # Default to 0 if somehow missing after processing
            feature_vector.append(pd.to_numeric(value, errors='coerce')) # Ensure numeric
        
        # Final check for NaNs in the feature vector (e.g., from failed to_numeric)
        feature_vector = [0 if pd.isna(v) else v for v in feature_vector]
        
        return np.array(feature_vector).reshape(1, -1)

    def _load_essay_text(self, applicant_data: pd.Series) -> str:
        """Loads and concatenates essay text from configured sources for a single applicant."""
        if not self.essay_source_config:
            print("Warning: essay_source_config not provided. Cannot load essays.")
            return ""

        all_essay_parts = []
        applicant_id = applicant_data.get('AMCAS_ID') # Assuming AMCAS_ID is the primary key

        for key, config in self.essay_source_config.items():
            essay_part_text = ""
            source_type = config.get('type')

            try:
                if source_type == 'excel':
                    file_path = config.get('file_path')
                    id_column = config.get('id_column')
                    text_column = config.get('text_column')
                    if not all([file_path, id_column, text_column, applicant_id]):
                        print(f"Warning: Missing configuration or applicant_id for Excel source '{key}'.")
                        continue
                    if not os.path.exists(file_path):
                        print(f"Warning: Excel file not found for source '{key}': {file_path}")
                        continue
                    
                    df_excel = pd.read_excel(file_path)
                    # Ensure ID column in Excel is treated as string for matching if applicant_id is string
                    df_excel[id_column] = df_excel[id_column].astype(str)
                    applicant_id_str = str(applicant_id)

                    row = df_excel[df_excel[id_column] == applicant_id_str]
                    if not row.empty:
                        essay_part_text = str(row.iloc[0][text_column])
                    else:
                        print(f"Info: Applicant ID {applicant_id_str} not found in Excel source '{key}' at {file_path}.")

                elif source_type == 'pdf_path_column':
                    column_name = config.get('column_name')
                    if not column_name:
                        print(f"Warning: Missing 'column_name' for pdf_path_column source '{key}'.")
                        continue
                    pdf_path = applicant_data.get(column_name)
                    if pdf_path and isinstance(pdf_path, str) and os.path.exists(pdf_path):
                        try:
                            with open(pdf_path, 'rb') as f:
                                reader = PyPDF2.PdfReader(f)
                                for page in reader.pages:
                                    essay_part_text += page.extract_text() + "\n"
                        except Exception as e:
                            print(f"Error reading PDF {pdf_path} for source '{key}': {e}")
                    elif pdf_path:
                        print(f"Warning: PDF file not found or invalid path for source '{key}': {pdf_path}")

                elif source_type == 'direct_column':
                    column_name = config.get('column_name')
                    if not column_name:
                        print(f"Warning: Missing 'column_name' for direct_column source '{key}'.")
                        continue
                    essay_part_text = str(applicant_data.get(column_name, ""))
                
                else:
                    print(f"Warning: Unknown essay source type '{source_type}' for key '{key}'.")

                if essay_part_text.strip():
                    all_essay_parts.append(essay_part_text.strip())

            except Exception as e:
                print(f"Error processing essay source '{key}': {e}")
                continue
        
        combined_essay = "\n\n---\n\n".join(all_essay_parts)
        return combined_essay[:self.MODEL_CONFIG.get('max_tokens_essay_input', 4000)]
    
    def _get_risen_prompt_template(self, evaluation_type: str) -> str:
        """Returns the RISEN prompt template for a given evaluation type."""
        # These are simplified placeholders based on the guide's description.
        # They should be expanded with the full details from the guide's RISEN prompt examples.
        
        # Define the header as simple string concatenation to avoid indentation issues
        risen_header = "ROLE:\n"
        risen_header += "You are an expert medical school admissions evaluator with years of experience on an admissions committee. "
        risen_header += "Your task is to analyze the provided essay text with exceptional attention to detail and nuance, "
        risen_header += "adhering strictly to the evaluation criteria.\n\n"
        risen_header += "INSTRUCTIONS:\n"
        risen_header += "Evaluate the essay based on the specific dimensions outlined below. "
        risen_header += "Provide scores as requested. Your evaluation should be fair, consistent, and evidence-based.\n\n"
        risen_header += "STEPS:\n"
        risen_header += "1. Carefully read the entire essay text provided.\n"
        risen_header += "2. Identify specific examples, experiences, and reflections that relate to each dimension.\n"
        risen_header += "3. Score each dimension numerically based on the provided scale and criteria.\n"
        risen_header += "4. For binary indicators, determine presence (1) or absence (0) of clear evidence.\n\n"
        risen_header += "END GOAL:\n"
        risen_header += "Your scores will contribute to a holistic review predicting the applicant's suitability for medical school "
        risen_header += "and their potential to become a compassionate and competent physician. For qualitative dimensions, "
        risen_header += "higher scores (e.g., 8-10) reflect strong positive indicators. For score predictions, "
        risen_header += "aim to match how a human reviewer would score.\n\n"
        risen_header += "NARROWING:\n"
        risen_header += "- Focus solely on the content of the essay.\n"
        risen_header += "- Avoid making assumptions beyond the text.\n"
        risen_header += "- Be objective and consistent in your application of scoring criteria.\n"
        risen_header += "- Output ONLY the numerical scores/indicators as specified, in the exact format requested."

        if evaluation_type == "comprehensive_15_point":
            # Placeholder: Expand with 10 qualitative dimensions (0-10) and 5 binary indicators (0/1)
            # Example: "Dimension 1 (0-10): ...\nDimension 2 (0-10): ...\nIndicator 1 (0/1): ...\nOUTPUT: score1,score2,...,score10,ind1,ind2,...,ind5"
            return risen_header + """SPECIFIC EVALUATION: Comprehensive 15-Point Assessment
[Detailed criteria for 10 qualitative dimensions and 5 binary indicators here...]
Predict likelihood of 21-25 reviewer score (0-10 scale).

ESSAY TEXT:
{text}

OUTPUT (16 comma-separated numbers: 10 scores, 5 indicators, 1 prediction score):"""
        elif evaluation_type == "evidence_counting":
            # Placeholder: Expand with 10 specific metrics to count
            # Example: "Metric 1 (count): Patient interaction stories...\nOUTPUT: count1,count2,...,count10"
            return risen_header + """SPECIFIC EVALUATION: Evidence Counting
[Detailed criteria for 10 specific metrics to count, e.g., patient stories, research projects, leadership...]

ESSAY TEXT:
{text}

OUTPUT (10 comma-separated numbers, representing counts for each metric):"""
        elif evaluation_type == "writing_quality":
            # Placeholder: Expand with 5 dimensions for writing quality (0-10 scale)
            # Example: "Dimension 1 (0-10): Clarity and coherence...\nOUTPUT: score1,score2,score3,score4,score5"
            return risen_header + """SPECIFIC EVALUATION: Writing Quality Assessment
[Detailed criteria for 5 dimensions: Clarity, Narrative, Specificity, Tone, Resonance...]

ESSAY TEXT:
{text}

OUTPUT (5 comma-separated numbers, 0-10 scale for each dimension):"""
        return "" # Should not happen

    def _call_openai_api(self, prompt_template: str, essay_text: str, expected_values: int) -> List[float]:
        """Helper function to call OpenAI API and parse response."""
        if not self.openai_client or not essay_text:
            return [0.0] * expected_values # Return neutral scores

        full_prompt = prompt_template.format(text=essay_text[:self.MODEL_CONFIG['max_tokens_essay_input']])
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.MODEL_CONFIG['chat_model'],
                messages=[
                    {"role": "system", "content": "You are an expert medical school admissions evaluator. Provide only numerical outputs as requested, separated by commas without any other text or explanation."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.MODEL_CONFIG['temperature'],
                max_tokens=self.MODEL_CONFIG['max_tokens_completion'] 
            )
            content = response.choices[0].message.content.strip()
            # Robust parsing: remove non-numeric/non-comma characters before splitting
            cleaned_content = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == ',', content))
            scores_str = cleaned_content.split(',')
            
            parsed_scores = []
            for s in scores_str:
                try:
                    parsed_scores.append(float(s.strip()))
                except ValueError:
                    parsed_scores.append(0.0) # Default for parsing errors
            
            # Ensure correct number of scores are returned
            if len(parsed_scores) < expected_values:
                parsed_scores.extend([0.0] * (expected_values - len(parsed_scores)))
            return parsed_scores[:expected_values]
            
        except Exception as e:
            logging.error(f"LLM extraction error: {e}")
            return [0.0] * expected_values  # Return neutral scores for errors
    
    def _get_llm_insights(self, essay_text: str) -> Dict:
        """Extracts various insights from essay text using RISEN prompts via OpenAI API."""
        insights = {
            # Comprehensive 15-point + prediction (16 values total)
            'comp_qual_dim1': 0.0, 'comp_qual_dim2': 0.0, 'comp_qual_dim3': 0.0, 'comp_qual_dim4': 0.0, 'comp_qual_dim5': 0.0,
            'comp_qual_dim6': 0.0, 'comp_qual_dim7': 0.0, 'comp_qual_dim8': 0.0, 'comp_qual_dim9': 0.0, 'comp_qual_dim10': 0.0,
            'comp_bin_ind1': 0.0, 'comp_bin_ind2': 0.0, 'comp_bin_ind3': 0.0, 'comp_bin_ind4': 0.0, 'comp_bin_ind5': 0.0,
            'comp_prediction_score': 0.0,
            # Evidence Counting (10 values)
            'ev_count_metric1': 0.0, 'ev_count_metric2': 0.0, 'ev_count_metric3': 0.0, 'ev_count_metric4': 0.0, 'ev_count_metric5': 0.0,
            'ev_count_metric6': 0.0, 'ev_count_metric7': 0.0, 'ev_count_metric8': 0.0, 'ev_count_metric9': 0.0, 'ev_count_metric10': 0.0,
            # Writing Quality (5 values)
            'wq_clarity': 0.0, 'wq_narrative': 0.0, 'wq_specificity': 0.0, 'wq_tone': 0.0, 'wq_resonance': 0.0,
            'overall_essay_quality_score': 5.0, # Aggregate placeholder
            'llm_error': False
        }

        if not self.openai_client or not essay_text.strip():
            logging.info("OpenAI client not available or essay text is empty. Using default scores.")
            insights['llm_error'] = True  # Indicate that LLM was not effectively used
            return insights

        try:
            # 1. Comprehensive 15-Point Assessment (10 qualitative, 5 binary, 1 prediction = 16 values)
            comp_prompt = self._get_risen_prompt_template("comprehensive_15_point")
            comp_scores = self._call_openai_api(comp_prompt, essay_text, expected_values=16)
            
            if len(comp_scores) == 16:
                qual_scores = comp_scores[:10]
                bin_scores = comp_scores[10:15]
                pred_score = comp_scores[15] if len(comp_scores) > 15 else 0
                
                for i, score in enumerate(qual_scores):
                    insights[f'comp_qual_dim{i+1}'] = score
                    
                for i, score in enumerate(bin_scores):
                    insights[f'comp_bin_ind{i+1}'] = score
                    
                insights['comp_prediction_score'] = pred_score
            
            # 2. Evidence Counting Assessment (10 count values)
            ev_prompt = self._get_risen_prompt_template("evidence_counting")
            ev_scores = self._call_openai_api(ev_prompt, essay_text, expected_values=10)
            
            for i, score in enumerate(ev_scores):
                insights[f'ev_count_metric{i+1}'] = score
            
            # 3. Writing Quality Assessment (5 quality dimensions)
            wq_prompt = self._get_risen_prompt_template("writing_quality")
            wq_scores = self._call_openai_api(wq_prompt, essay_text, expected_values=5)
            
            quality_dims = ['clarity', 'narrative', 'specificity', 'tone', 'resonance']
            for i, dim in enumerate(quality_dims):
                if i < len(wq_scores):
                    insights[f'wq_{dim}'] = wq_scores[i]
            
            # Calculate overall essay quality (average of writing quality scores)
            if any(wq_scores):
                insights['overall_essay_quality_score'] = sum(wq_scores) / len(wq_scores)
            
            return insights
            
        except Exception as e:
            logging.error(f"Error in LLM insights extraction: {str(e)}")
            insights['llm_error'] = True
            return insights
    def predict_with_confidence(self, 
                              applicant_data: pd.Series) -> Dict:
        """
        Make prediction combining structured features and essay analysis
        """
        
        # Get structured features
        X = self.prepare_structured_features(applicant_data)
        
        # Normalize experience hours (matching training)
        # Note: The scaler should be FIT during training and LOADED here.
        # For this example, it's re-fit, which is not ideal for production.
        experience_indices = [i for i, name in enumerate(self.feature_names) 
                            if 'Hour' in name or 'Hours' in name]
        if experience_indices:
            # Fit scaler on typical ranges (you might want to save this from training)
            # This is a placeholder. A scaler fit on the training data should be used.
            # Use the scaler loaded from training if available
            if self.scaler is not None:
                 X[:, experience_indices] = self.scaler.transform(X[:, experience_indices])
            else: # Fallback if scaler not saved with model, less ideal
                print("Warning: Experience hour scaler not found in model data. Using ad-hoc scaling.")
                typical_hours = np.array([[1000, 500, 200, 100, 300, 400]]) 
                # Ensure typical_hours matches the number of experience features
                num_exp_features = len(experience_indices)
                if typical_hours.shape[1] >= num_exp_features:
                    self.scaler.fit(typical_hours[:, :num_exp_features])
                    X[:, experience_indices] = self.scaler.transform(X[:, experience_indices])
                else:
                    print(f"Warning: Not enough typical_hours columns ({typical_hours.shape[1]}) for {num_exp_features} experience features. Skipping scaling.")

        # Scale features if scaler is available
        if self.scaler:
            # Identify columns to scale based on feature_names from the loaded model
            # This assumes the scaler was fit on a subset of features that are present in self.feature_names
            # and that their order in X matches the order they had during scaler fitting.
            # A more robust way is to save the names of scaled columns with the scaler.
            # For now, we assume the scaler was fit on all features in self.feature_names or a known subset.
            # The `HighConfidenceFourTierClassifier` scales a specific list of columns.
            # We need to find the indices of these columns in `self.feature_names`.
            
            # These are the column types scaled in HighConfidenceFourTierClassifier
            scaled_col_keywords = ['Hour', 'Hours', 'Trend', 'Age', 'Num_Dependents', 
                                   'research_intensity', 'clinical_intensity', 
                                   'experience_balance', 'service_commitment', 'adversity_overcome']
            # Exclude binary indicators that might accidentally match keywords
            binary_indicators = ['Disadvantanged_Ind', 'First_Generation_Ind', 'SES_Value']

            cols_to_scale_indices = []
            for i, fname in enumerate(self.feature_names):
                if fname not in binary_indicators and any(keyword in fname for keyword in scaled_col_keywords):
                    cols_to_scale_indices.append(i)
            
            if cols_to_scale_indices:
                # Apply scaling only to the identified columns
                # Create a copy of the sub-array to scale to avoid SettingWithCopyWarning if X were a DataFrame slice
                X_cols_to_scale = X[:, cols_to_scale_indices].copy()
                X_cols_scaled = self.scaler.transform(X_cols_to_scale)
                X[:, cols_to_scale_indices] = X_cols_scaled # Place scaled values back
                print(f"Applied scaling to {len(cols_to_scale_indices)} features.")
            else:
                print("Warning: No features identified for scaling based on keywords, or scaler not compatible.")
        else:
            print("Warning: Scaler not available. Features will not be scaled.")

        # Get base prediction from structured model
        predicted_class_label = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Create a dictionary of class probabilities
        prob_dict = {self.class_names[i]: probabilities[i] for i in range(len(self.class_names))}
        
        # Initial confidence is the probability of the predicted class
        confidence = prob_dict[predicted_class_label]

        # --- LLM Insights Integration & Confidence Adjustment (Placeholder for full logic) ---
        llm_insights = {}
        final_tier_label = predicted_class_label
        final_confidence = confidence
        recommendation = "Awaiting full LLM integration for detailed recommendation."
        essay_quality_overall = 0.0 # Placeholder
        requires_essay_review = True # Default until LLM logic is complete

        # Load essay text using the new method
        essay_text_loaded = self._load_essay_text(applicant_data)

        # LLM Insights (if enabled)
        llm_insights = {}
        if self.openai_client:
            if essay_text_loaded:
                llm_insights = self._get_llm_insights(essay_text_loaded)
        # Tier 4 (Very Likely Interview) requires >= 75% confidence
        # Otherwise, moves to adjacent tier.
        # Note: self.class_names should be sorted: ['1. Very Unlikely', '2. Potential Review', '3. Probable Interview', '4. Very Likely Interview']
        
        tier_1_label = self.class_names[0] # '1. Very Unlikely'
        tier_2_label = self.class_names[1] # '2. Potential Review'
        tier_3_label = self.class_names[2] # '3. Probable Interview'
        tier_4_label = self.class_names[3] # '4. Very Likely Interview'

        current_tier_idx = self.class_names.index(final_tier_label)

        if final_tier_label == tier_1_label and final_confidence < 0.70:
            final_tier_label = tier_2_label # Move to Potential Review
            # Confidence for the new tier might need recalculation or use prob_dict[tier_2_label]
            final_confidence = prob_dict.get(final_tier_label, 0.0) 
        elif final_tier_label == tier_4_label and final_confidence < 0.75:
            final_tier_label = tier_3_label # Move to Probable Interview
            final_confidence = prob_dict.get(final_tier_label, 0.0)

        # Recommendation placeholder - to be improved with LLM insights
        if final_tier_label == tier_4_label:
            recommendation = "Very strong candidate. Prioritize for interview."
        elif final_tier_label == tier_3_label:
            recommendation = "Likely to get interview. Review recommended."
        elif final_tier_label == tier_2_label:
            recommendation = "Potential for interview. Requires careful review."
        else: # Tier 1
            recommendation = "Unlikely to be competitive. Review if specific program needs align."

        # --- Construct Final Output --- 
        # Ensure AMCAS ID is present in the output
        amcas_id_val = applicant_data.get('AMCAS ID', applicant_data.get('Amcas_ID', 'N/A'))

        # Retrieve essay text for further analysis
        essay_text = self._load_essay_text(applicant_data)
        
        # Basics - get model prediction on structured data
        base_proba = self.model.predict_proba(X)[0]
        base_pred = np.argmax(base_proba)
        
        # Try to incorporate LLM insights if available
        if self.openai_client and essay_text:
            # Extract essay insights & scores using LLM
            llm_insights = self._get_llm_insights(essay_text)
            
            # Calculate an overall essay quality score (0-1) 
            essay_quality = llm_insights.get('overall_essay_quality_score', 5.0) / 10.0
            
            # This needs to align with how your 'class_names' are structured.
            # If class_names = ['1. Very Unlikely', '2. Potential Review', '3. Probable Interview', '4. Very Likely Interview']
            # Then '4. Very Likely Interview' would be index 3.
            
            # Simplified adjustment logic (assumes higher index = more positive outcome)
            positive_class_index = len(self.class_names) - 1 # Most positive class
            middle_class_index = len(self.class_names) // 2 # A middle class

            if base_pred == positive_class_index:  # e.g., Likely interview
                confidence_adjustment = (essay_quality - 0.5) * 0.2
            elif base_pred == middle_class_index:  # e.g., Maybe
                if essay_quality > 0.7: confidence_adjustment = 0.15
                elif essay_quality < 0.3: confidence_adjustment = -0.15
                else: confidence_adjustment = 0
            else:  # e.g., Unlikely
                confidence_adjustment = max(0, (essay_quality - 0.8) * 0.3)
            
            adjusted_proba = base_proba.copy()
            # Ensure base_pred is a valid index for adjusted_proba
            if 0 <= base_pred < len(adjusted_proba):
                adjusted_proba[base_pred] = min(1.0, max(0.0, adjusted_proba[base_pred] + confidence_adjustment))
                adjusted_proba = adjusted_proba / adjusted_proba.sum()  # Renormalize
            else:
                logging.warning(f"Base prediction index {base_pred} out of bounds for probabilities array.")
                # No adjustment if index is invalid
        else:
            llm_insights = None
            adjusted_proba = base_proba
            essay_quality = None
        
        final_pred_idx = np.argmax(adjusted_proba)
        final_confidence = adjusted_proba[final_pred_idx]
        
        decision_changed = (final_pred_idx != base_pred) if llm_insights else False
        
        # Generate recommendation text
        recommendation = self._get_recommendation(
            pred_class_idx=final_pred_idx,
            confidence=final_confidence, 
            essay_quality=essay_quality if essay_quality is not None else 0.5
        )
        
        # Format probabilities for readable output
        prob_dict = {self.class_names[i]: float(p) for i, p in enumerate(adjusted_proba)}
        
        # Create final tier label
        final_tier_label = self.class_names[final_pred_idx]
        
        # Determine if this candidate needs manual essay review
        # Logic: If LLM errored, essay quality is low, or decision changed due to essay
        essay_quality_overall = essay_quality * 10 if essay_quality is not None else 5.0
        requires_essay_review = (
            llm_insights is None or 
            llm_insights.get('llm_error', False) or
            essay_quality_overall < 4.0 or  # Low quality essays need human eyes
            decision_changed  # Decision changed by essay analysis
        )
        
        return {
            "AMCAS_ID": str(amcas_id_val),
            "prediction": final_tier_label,
            "tier_number": self.class_names.index(final_tier_label) + 1, # 1-indexed tier number
            "confidence": round(final_confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "recommendation": recommendation,
            "essay_quality": round(essay_quality_overall, 2) if essay_quality is not None else None, # From LLM, 0-10 scale
            "requires_essay_review": requires_essay_review, # Based on LLM processing success/need
            "llm_detailed_scores": llm_insights # Include all raw scores from LLM for transparency
        }

    
    def _get_recommendation(self, pred_class_idx: int, confidence: float, essay_quality: float) -> str:
        """Generate actionable recommendation based on predicted class index"""
        # This logic needs to map pred_class_idx to your specific class meanings
        # Example for a 4-tier system (indices 0, 1, 2, 3)
        # 0: Very Unlikely, 1: Potential Review, 2: Probable Interview, 3: Very Likely Interview
        
        if pred_class_idx == 3:  # '4. Very Likely Interview'
            if confidence > 0.8: return "Strong candidate - recommend direct interview invitation"
            else: return "Good candidate - recommend interview, review essays for final confirmation"
                
        elif pred_class_idx == 2:  # '3. Probable Interview'
            # Adjusting logic based on essay for 'Probable Interview' as it's also a positive outcome
            if essay_quality and essay_quality > 0.7 and confidence > 0.6:
                 return "Probable interview with strong essays - recommend interview"
            elif confidence > 0.7:
                 return "Probable interview - recommend interview"
            else:
                 return "Leaning towards interview - committee review needed"

        elif pred_class_idx == 1:  # '2. Potential Review'
            if essay_quality and essay_quality > 0.7: return "Borderline candidate with strong essays - recommend committee review with positive bias"
            elif essay_quality and essay_quality < 0.4: return "Borderline candidate with weak essays - committee review but unlikely to proceed"
            else: return "Borderline candidate - requires full committee review"
                
        else:  # 0: '1. Very Unlikely'
            if essay_quality and essay_quality > 0.8 and confidence < 0.5: # Low confidence in rejection + great essay
                return "Weak metrics but exceptional essays - consider committee review"
            else:
                return "Does not meet minimum criteria - recommend polite rejection"


# Example usage showing the complete workflow
def process_batch_with_model(applicants_df: pd.DataFrame,
                           essays_dict: Dict[str, str] = None,
                           model_path: str = 'models/four_tier_high_confidence_model.pkl',
                           api_key: str = None) -> pd.DataFrame:
    """
    Process a batch of applicants using the trained model
    """
    
    # Initialize predictor
    # Ensure the model_path points to the model saved by train_model.py
    predictor = IntegratedAdmissionsPredictor(model_path, api_key)
    
    results = []
    
    for idx, row in applicants_df.iterrows():
        amcas_id_col_names = ['Amcas_ID', 'AMCAS ID', 'application_id'] # Add other possible names
        amcas_id = ''
        for col_name in amcas_id_col_names:
            if col_name in row and pd.notna(row[col_name]):
                amcas_id = str(row[col_name])
                break
        if not amcas_id:
            print(f"Warning: AMCAS ID not found for row {idx}. Skipping.")
            continue
            
        essay = essays_dict.get(amcas_id, '')
        
        prediction = predictor.predict_with_confidence(row, essay)
        
        # Dynamically create probability columns based on class_names
        result_row = {
            'AMCAS_ID': amcas_id,
            'Predicted_Class': prediction['prediction'],
            'Confidence': prediction['confidence'],
            'Essay_Quality': prediction['essay_quality'],
            'Decision_Changed': prediction['decision_changed_by_essay'],
            'Recommendation': prediction['recommendation']
        }
        for class_name, prob in prediction['probabilities'].items():
            result_row[f'Probability_{class_name.replace(" ", "_").replace(".", "")}'] = prob
        
        results.append(result_row)
        
        if idx == 0:
            print("Example Prediction:")
            print(f"Applicant: {amcas_id}")
            print(f"Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.2%})")
            print(f"Essay Impact: {'Yes' if prediction['decision_changed_by_essay'] else 'No'}")
            print(f"Recommendation: {prediction['recommendation']}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example: Load your data
    print("Loading data...")
    # Make sure the path to '1. Applicants.xlsx' is correct or it's in the same directory
    try:
        applicants_df = pd.read_excel('1. Applicants.xlsx').head(5)  # Just 5 for example
    except FileNotFoundError:
        print("Error: '1. Applicants.xlsx' not found. Make sure it's in the correct path.")
        exit()

    # Mock essays (in production, load from your essays file)
    # Ensure Amcas_ID used here matches an ID in your sample applicants_df
    mock_amcas_id = ''
    if not applicants_df.empty:
        # Try to get a valid AMCAS ID from the loaded data
        for col_name in ['Amcas_ID', 'AMCAS ID']:
            if col_name in applicants_df.columns and pd.notna(applicants_df.iloc[0][col_name]):
                mock_amcas_id = str(applicants_df.iloc[0][col_name])
                break
    
    essays_dict = {}
    if mock_amcas_id:
        essays_dict[mock_amcas_id] = "My journey to medicine began when I witnessed a complex surgical procedure. The precision and teamwork inspired me. I volunteered extensively at a local clinic, where I learned the importance of empathy and communication in patient care. These experiences solidified my resolve to become a physician, dedicated to serving diverse communities and advancing medical knowledge through research."
    else:
        print("Warning: Could not find a valid AMCAS ID in the first 5 rows of '1. Applicants.xlsx' for mock essay.")

    # Process with trained model
    # Use the correct model file path that matches four_tier_classifier.py
    MODEL_FILE_PATH = 'models/four_tier_high_confidence_model.pkl'
    print(f"\nProcessing applicants using model: {MODEL_FILE_PATH}...")
    
    # Check if model file exists
    import os
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"Error: Model file '{MODEL_FILE_PATH}' not found. Train the model using train_model.py first.")
        exit()

    results_df = process_batch_with_model(
        applicants_df,
        essays_dict,
        model_path=MODEL_FILE_PATH, # Use the variable
        api_key='your-openai-key'  # Optional: Replace with your actual key or load from env
    )
    
    print("\nResults:")
    print(results_df)
    
    if not results_df.empty:
        print("\nSummary:")
        print(results_df['Predicted_Class'].value_counts())
        print(f"\nAverage confidence: {results_df['Confidence'].mean():.2%}")
        print(f"Essays changed decision: {results_df['Decision_Changed'].sum()} cases")
    else:
        print("\nNo results to summarize.")
