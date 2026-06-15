# app.py
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import Config, seed_everything
from src.pipeline_stage1 import IngestionEngine, PseudoLabelGenerator
from src.dossier_engine import DossierEngine

seed_everything()

st.set_page_config(layout="wide", page_title="SIA Dashboard", page_icon="🛡️")

# Cache model loading to prevent reloading lag on UI interactions
@st.cache_resource
def load_auditor_model():
    model_path = "./models/sia_deberta"
    if not os.path.exists(model_path):
        return None, None, 0.5
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Load optimal decision threshold
    threshold_file = os.path.join(model_path, "threshold_config.json")
    threshold = 0.5
    if os.path.exists(threshold_file):
        with open(threshold_file, "r") as f:
            threshold = json.load(f).get("optimal_threshold", 0.5)
            
    return tokenizer, model, threshold

tokenizer, model, decision_threshold = load_auditor_model()

st.title("🛡️ Support Integrity Auditor (SIA)")
st.markdown("Automated auditing system to detect mismatches between assigned ticket priority and objective semantic indicators.")

if not model:
    st.error("⚠️ Trained model not detected at './models/sia_deberta'. Please run 'train_pipeline.py' before launching the application.")
    st.stop()

# Set up Workspace Panels
tab1, tab2 = st.tabs(["📊 Batch File Auditor & Dashboard", "🔍 Single Ticket Auditor"])

with tab1:
    st.header("Batch Ticket Auditing Portal")
    uploaded_file = st.file_uploader("Upload Support Tickets CSV File", type=["csv"])
    
    if uploaded_file:
        # Load and process data
        df_raw = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(df_raw)} records from file.")
        
        # Ingest and clean using pipeline logic
        with st.spinner("Processing metadata indicators..."):
            temp_path = "data/temp_upload.csv"
            df_raw.to_csv(temp_path, index=False)
            cleaned_df = IngestionEngine.clean_crm_data(temp_path)
            
            # Apply Stage 1 signal logic
            generator = PseudoLabelGenerator(w1=0.6, w2=0.4, threshold=2)
            processed_df = generator.run_pipeline(cleaned_df)
            os.remove(temp_path)
            
        # Serialize text features for tokenizer
        def serialize(row):
            return (
                f"Assigned Priority: {row['Ticket Priority']} | "
                f"Channel: {row['Ticket Channel']} | "
                f"Domain: {row['Customer Domain']} | "
                f"Type: {row['Ticket Type']} | "
                f"Subject: {row['Ticket Subject']} | "
                f"Description: {row['Ticket Description']}"
            )
        processed_df['text'] = processed_df.apply(serialize, axis=1)
        
        # Batch Model Prediction
        predictions = []
        confidences = []
        with st.spinner("Evaluating structural descriptions..."):
            for text in processed_df['text'].values:
                inputs = tokenizer(text, max_length=Config.MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    prob_mismatch = probs[0][1].item()
                    
                    pred_class = 1 if prob_mismatch >= decision_threshold else 0
                    conf = prob_mismatch if pred_class == 1 else (1.0 - prob_mismatch)
                    
                    predictions.append(pred_class)
                    confidences.append(conf)
                    
        processed_df['predicted_mismatch'] = predictions
        processed_df['confidence_score'] = confidences
        
        # UI Metrics Display
        mismatch_count = sum(predictions)
        mismatch_rate = mismatch_count / len(processed_df)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Tickets Scanned", f"{len(processed_df)}")
        m2.metric("Flagged Priority Mismatches", f"{mismatch_count} tickets")
        m3.metric("Audited Conflict Rate", f"{mismatch_rate:.2%}")
        
        # Dashboard Visualizations
        st.subheader("Priority Audit Dashboard")
        g1, g2 = st.columns(2)
        
        with g1:
            fig_pie = px.pie(
                processed_df, 
                names='predicted_mismatch', 
                title="Audit Status Distribution",
                labels={'predicted_mismatch': 'Mismatch Status'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with g2:
            fig_heat = px.density_heatmap(
                processed_df,
                x="Ticket Type",
                y="Ticket Channel",
                z="severity_delta",
                histfunc="avg",
                title="Mean Severity Delta Heatmap by Channel and Category"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        # Display Mismatch Audits and Dossiers
        st.subheader("Audited Mismatch Logs")
        mismatch_rows = processed_df[processed_df['predicted_mismatch'] == 1]
        
        if len(mismatch_rows) > 0:
            selected_id = st.selectbox("Select Ticket ID to Inspect Evidence Dossier", mismatch_rows['Ticket ID'].values)
            target_row = mismatch_rows[mismatch_rows['Ticket ID'] == selected_id].iloc[0]
            
            dossier = DossierEngine.generate(target_row.to_dict(), 1, target_row['confidence_score'])
            
            st.json(dossier)
        else:
            st.success("No priority mismatches detected in this batch.")

with tab2:
    st.header("Single Ticket Integrity Auditor")
    st.markdown("Manually evaluate a single ticket configuration to verify priority assignments.")
    
    with st.form("manual_ticket_form"):
        subject = st.text_input("Ticket Subject", "System completely down")
        desc = st.text_area("Ticket Description", "Our server crashed and is throwing a 500 server error immediately upon boot.")
        col1, col2, col3 = st.columns(3)
        
        priority = col1.selectbox("Assigned Ticket Priority", list(Config.PRIORITY_MAP.keys()))
        category = col2.selectbox("Ticket Category", ["Billing Inquiry", "Technical Support", "Account Access", "Product Feedback"])
        channel = col3.selectbox("Ticket Channel", ["Email", "Chat", "Phone", "Social Media"])
        
        email = st.text_input("Customer Contact Email", "admin@enterprisecompany.com")
        res_time = st.number_input("Observed Resolution Time (Hours)", min_value=1.0, value=24.0)
        
        submitted = st.form_submit_button("Execute Audit Check")
        
        if submitted:
            # Construct a synthetic row
            synthetic_row = {
                'Ticket ID': 'MANUAL-999',
                'Ticket Subject': subject,
                'Ticket Description': desc,
                'Ticket Priority': priority,
                'Ticket Type': category,
                'Ticket Channel': channel,
                'Customer Email': email,
                'Customer Domain': email.split('@')[-1] if '@' in email else 'unknown.com',
                'Resolution Time': res_time,
                'P_assigned': Config.PRIORITY_MAP[priority],
                # Approximated evaluation mappings for inference
                'resolution_z_score': 1.5,
                'S_ops': 2,
                'S_sem': 4,
                'inferred_severity': 3,
                'inferred_severity_label': 'High',
                'severity_delta': abs(3 - Config.PRIORITY_MAP[priority])
            }
            
            # Serialize
            serialized_text = (
                f"Assigned Priority: {synthetic_row['Ticket Priority']} | "
                f"Channel: {synthetic_row['Ticket Channel']} | "
                f"Domain: {synthetic_row['Customer Domain']} | "
                f"Type: {synthetic_row['Ticket Type']} | "
                f"Subject: {synthetic_row['Ticket Subject']} | "
                f"Description: {synthetic_row['Ticket Description']}"
            )
            
            # Run Model
            inputs = tokenizer(serialized_text, max_length=Config.MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                prob_mismatch = probs[0][1].item()
                
                is_mismatch = 1 if prob_mismatch >= decision_threshold else 0
                confidence = prob_mismatch if is_mismatch == 1 else (1.0 - prob_mismatch)
                
            if is_mismatch == 1:
                st.error(f"🚨 Priority Mismatch Detected (Confidence: {confidence:.2%})")
                dossier = DossierEngine.generate(synthetic_row, 1, confidence)
                st.subheader("Grounded Evidence Dossier")
                st.json(dossier)
            else:
                st.success(f"✅ Priority Integrity Confirmed (Confidence: {confidence:.2%})")