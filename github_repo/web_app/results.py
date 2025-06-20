"""
Results Viewer Module
Displays and manages application processing results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io


class ResultsViewer:
    """Display and manage application results."""
    
    def __init__(self, data):
        """Initialize with processed data."""
        self.data = data
        
    def display(self):
        """Main display method for results."""
        # Add tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "📋 Detailed Results", 
            "📈 Analytics",
            "📥 Export"
        ])
        
        with tab1:
            self._display_overview()
        
        with tab2:
            self._display_detailed_results()
        
        with tab3:
            self._display_analytics()
        
        with tab4:
            self._display_export()
    
    def _display_overview(self):
        """Display overview statistics and visualizations."""
        st.markdown("### 📊 Results Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = len(self.data)
            st.metric("Total Applications", total)
        
        with col2:
            q1_count = len(self.data[self.data['predicted_quartile'] == 'Q1'])
            q1_pct = (q1_count / total * 100) if total > 0 else 0
            st.metric("Q1 Candidates", f"{q1_count} ({q1_pct:.1f}%)")
        
        with col3:
            high_conf = len(self.data[self.data['confidence'] >= 80])
            high_conf_pct = (high_conf / total * 100) if total > 0 else 0
            st.metric("High Confidence", f"{high_conf} ({high_conf_pct:.1f}%)")
        
        with col4:
            review_needed = len(self.data[self.data['needs_review'] == True])
            review_pct = (review_needed / total * 100) if total > 0 else 0
            st.metric("Need Review", f"{review_needed} ({review_pct:.1f}%)")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Quartile distribution
            fig_pie = px.pie(
                self.data,
                names='predicted_quartile',
                title='Quartile Distribution',
                color_discrete_map={
                    'Q1': '#006747',
                    'Q2': '#4A9B7F',
                    'Q3': '#FFB500',
                    'Q4': '#FFD166'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_hist = px.histogram(
                self.data,
                x='confidence',
                nbins=20,
                title='Confidence Score Distribution',
                labels={'confidence': 'Confidence Score', 'count': 'Number of Applications'}
            )
            fig_hist.update_traces(marker_color='#006747')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Quartile breakdown
        st.markdown("### 📊 Quartile Breakdown")
        
        quartile_stats = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            q_data = self.data[self.data['predicted_quartile'] == q]
            quartile_stats.append({
                'Quartile': q,
                'Count': len(q_data),
                'Avg Confidence': f"{q_data['confidence'].mean():.1f}" if len(q_data) > 0 else "N/A",
                'Need Review': len(q_data[q_data['needs_review'] == True]),
                'Description': self._get_quartile_description(q)
            })
        
        df_quartiles = pd.DataFrame(quartile_stats)
        st.dataframe(df_quartiles, use_container_width=True)
    
    def _display_detailed_results(self):
        """Display detailed results table with filtering."""
        st.markdown("### 📋 Detailed Results")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            quartile_filter = st.multiselect(
                "Filter by Quartile",
                options=['Q1', 'Q2', 'Q3', 'Q4'],
                default=['Q1', 'Q2', 'Q3', 'Q4']
            )
        
        with col2:
            confidence_range = st.slider(
                "Confidence Range",
                min_value=0,
                max_value=100,
                value=(0, 100)
            )
        
        with col3:
            review_filter = st.selectbox(
                "Review Status",
                options=['All', 'Need Review', 'No Review Needed']
            )
        
        with col4:
            search_term = st.text_input("Search by ID or Name")
        
        # Apply filters
        filtered_data = self.data.copy()
        
        # Quartile filter
        filtered_data = filtered_data[filtered_data['predicted_quartile'].isin(quartile_filter)]
        
        # Confidence filter
        filtered_data = filtered_data[
            (filtered_data['confidence'] >= confidence_range[0]) &
            (filtered_data['confidence'] <= confidence_range[1])
        ]
        
        # Review filter
        if review_filter == 'Need Review':
            filtered_data = filtered_data[filtered_data['needs_review'] == True]
        elif review_filter == 'No Review Needed':
            filtered_data = filtered_data[filtered_data['needs_review'] == False]
        
        # Search filter
        if search_term:
            # Search in ID and name columns if they exist
            mask = filtered_data['amcas_id'].astype(str).str.contains(search_term, case=False)
            if 'name' in filtered_data.columns:
                mask |= filtered_data['name'].str.contains(search_term, case=False)
            filtered_data = filtered_data[mask]
        
        # Display count
        st.info(f"Showing {len(filtered_data)} of {len(self.data)} applications")
        
        # Prepare display columns
        display_columns = [
            'amcas_id', 'predicted_quartile', 'confidence', 'needs_review'
        ]
        
        # Add optional columns if they exist
        optional_cols = ['name', 'service_rating_numerical', 'healthcare_total_hours']
        for col in optional_cols:
            if col in filtered_data.columns:
                display_columns.append(col)
        
        # Color code confidence
        def highlight_confidence(row):
            if row['confidence'] >= 80:
                return ['background-color: #d4edda'] * len(row)
            elif row['confidence'] >= 60:
                return ['background-color: #fff3cd'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)
        
        # Display dataframe
        styled_df = filtered_data[display_columns].style.apply(highlight_confidence, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Selected applicant details
        if st.checkbox("Show detailed view for selected applicant"):
            selected_id = st.selectbox(
                "Select Applicant ID",
                options=filtered_data['amcas_id'].tolist()
            )
            
            if selected_id:
                self._display_applicant_details(selected_id)
    
    def _display_analytics(self):
        """Display analytical insights."""
        st.markdown("### 📈 Analytics & Insights")
        
        # Demographics analysis if available
        if 'gender' in self.data.columns:
            st.markdown("#### Gender Distribution by Quartile")
            
            gender_quartile = pd.crosstab(
                self.data['predicted_quartile'],
                self.data['gender'],
                normalize='index'
            ) * 100
            
            fig_gender = px.bar(
                gender_quartile,
                title="Gender Distribution Across Quartiles (%)",
                labels={'value': 'Percentage', 'index': 'Quartile'},
                color_discrete_map={
                    'Male': '#006747',
                    'Female': '#FFB500',
                    'Other': '#4A9B7F'
                }
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Confidence analysis
        st.markdown("#### Confidence Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence by quartile
            fig_box = px.box(
                self.data,
                x='predicted_quartile',
                y='confidence',
                title='Confidence Distribution by Quartile',
                color='predicted_quartile',
                color_discrete_map={
                    'Q1': '#006747',
                    'Q2': '#4A9B7F',
                    'Q3': '#FFB500',
                    'Q4': '#FFD166'
                }
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Review needs by quartile
            review_by_quartile = self.data.groupby('predicted_quartile')['needs_review'].mean() * 100
            
            fig_review = px.bar(
                x=review_by_quartile.index,
                y=review_by_quartile.values,
                title='% Needing Review by Quartile',
                labels={'x': 'Quartile', 'y': 'Percentage Needing Review'}
            )
            fig_review.update_traces(marker_color='#FFB500')
            st.plotly_chart(fig_review, use_container_width=True)
        
        # Feature importance if available
        if 'service_rating_numerical' in self.data.columns:
            st.markdown("#### Key Feature Analysis")
            
            # Average service rating by quartile
            avg_service = self.data.groupby('predicted_quartile')['service_rating_numerical'].mean()
            
            fig_service = px.bar(
                x=avg_service.index,
                y=avg_service.values,
                title='Average Service Rating by Quartile',
                labels={'x': 'Quartile', 'y': 'Average Service Rating'}
            )
            fig_service.update_traces(marker_color='#006747')
            st.plotly_chart(fig_service, use_container_width=True)
    
    def _display_export(self):
        """Display export options."""
        st.markdown("### 📥 Export Results")
        
        # Export format selection
        export_format = st.radio(
            "Select export format",
            options=["Excel (.xlsx)", "CSV (.csv)", "Summary Report (PDF)"]
        )
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            include_all_fields = st.checkbox("Include all data fields", value=False)
            include_summary = st.checkbox("Include summary statistics", value=True)
        
        with col2:
            if 'needs_review' in self.data.columns:
                export_filter = st.selectbox(
                    "Export subset",
                    options=["All Applications", "Q1 Only", "Need Review Only", "High Confidence Only"]
                )
        
        # Prepare export data
        export_data = self.data.copy()
        
        # Apply export filter
        if export_filter == "Q1 Only":
            export_data = export_data[export_data['predicted_quartile'] == 'Q1']
        elif export_filter == "Need Review Only":
            export_data = export_data[export_data['needs_review'] == True]
        elif export_filter == "High Confidence Only":
            export_data = export_data[export_data['confidence'] >= 80]
        
        # Select columns
        if not include_all_fields:
            export_columns = [
                'amcas_id', 'predicted_quartile', 'confidence', 'needs_review'
            ]
            # Add key fields if available
            for col in ['name', 'service_rating_numerical', 'healthcare_total_hours']:
                if col in export_data.columns:
                    export_columns.append(col)
            export_data = export_data[export_columns]
        
        # Export button
        if st.button("🚀 Generate Export", use_container_width=True):
            
            if export_format == "Excel (.xlsx)":
                # Create Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Main results
                    export_data.to_excel(writer, sheet_name='Results', index=False)
                    
                    # Summary statistics if requested
                    if include_summary:
                        summary_data = self._generate_summary_stats()
                        summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # Download button
                st.download_button(
                    label="📥 Download Excel File",
                    data=output.getvalue(),
                    file_name=f"rush_admissions_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            elif export_format == "CSV (.csv)":
                # Create CSV
                csv = export_data.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV File",
                    data=csv,
                    file_name=f"rush_admissions_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            else:  # PDF Report
                st.info("PDF report generation would be implemented here with charts and summary statistics.")
    
    def _display_applicant_details(self, amcas_id):
        """Display detailed view for a single applicant."""
        applicant = self.data[self.data['amcas_id'] == amcas_id].iloc[0]
        
        st.markdown(f"### Applicant Details: {amcas_id}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AI Assessment")
            st.metric("Predicted Quartile", applicant['predicted_quartile'])
            st.metric("Confidence Score", f"{applicant['confidence']:.1f}%")
            st.metric("Review Needed", "Yes" if applicant['needs_review'] else "No")
        
        with col2:
            st.markdown("#### Key Metrics")
            if 'service_rating_numerical' in applicant:
                st.metric("Service Rating", applicant['service_rating_numerical'])
            if 'healthcare_total_hours' in applicant:
                st.metric("Clinical Hours", f"{applicant['healthcare_total_hours']:.0f}")
            if 'exp_hour_research' in applicant:
                st.metric("Research Hours", f"{applicant['exp_hour_research']:.0f}")
        
        # Probability breakdown if available
        prob_cols = ['reject_probability', 'waitlist_probability', 
                    'interview_probability', 'accept_probability']
        if all(col in applicant for col in prob_cols):
            st.markdown("#### Probability Breakdown")
            
            probs = [
                applicant['reject_probability'],
                applicant['waitlist_probability'],
                applicant['interview_probability'],
                applicant['accept_probability']
            ]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Reject (Q4)', 'Waitlist (Q3)', 'Interview (Q2)', 'Accept (Q1)'],
                    y=probs,
                    marker_color=['#FFD166', '#FFB500', '#4A9B7F', '#006747']
                )
            ])
            fig.update_layout(
                title="Outcome Probabilities",
                yaxis_title="Probability",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _get_quartile_description(self, quartile):
        """Get description for each quartile."""
        descriptions = {
            'Q1': "Top 25% - Strongest candidates for interview",
            'Q2': "50-75% - Above average, consider for interview",
            'Q3': "25-50% - Below average, potential waitlist",
            'Q4': "Bottom 25% - Weakest applications"
        }
        return descriptions.get(quartile, "")
    
    def _generate_summary_stats(self):
        """Generate summary statistics for export."""
        summary = []
        
        # Overall stats
        summary.append({
            'Metric': 'Total Applications',
            'Value': len(self.data)
        })
        
        # Quartile distribution
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            count = len(self.data[self.data['predicted_quartile'] == q])
            pct = (count / len(self.data) * 100) if len(self.data) > 0 else 0
            summary.append({
                'Metric': f'{q} Count',
                'Value': f"{count} ({pct:.1f}%)"
            })
        
        # Confidence stats
        summary.append({
            'Metric': 'Average Confidence',
            'Value': f"{self.data['confidence'].mean():.1f}%"
        })
        
        summary.append({
            'Metric': 'High Confidence (≥80%)',
            'Value': len(self.data[self.data['confidence'] >= 80])
        })
        
        summary.append({
            'Metric': 'Need Review',
            'Value': len(self.data[self.data['needs_review'] == True])
        })
        
        return pd.DataFrame(summary)