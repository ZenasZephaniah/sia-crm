# predict.py
import os
import json
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import Config, seed_everything
from src.pipeline_stage1 import IngestionEngine, PseudoLabelGenerator
from src.dossier_engine import DossierEngine

seed_everything()

def run_inference(input_path: str, output_path: str):
    # 1. Load and Standardize input CSV
    print(f"Step 1: Reading and cleaning input file: {input_path}")
    raw_df = IngestionEngine.clean_crm_data(input_path)
    
    # Run Stage 1 logic to establish baseline priority variables
    print("Step 2: Processing operational and semantic markers...")
    generator = PseudoLabelGenerator(w1=0.6, w2=0.4, threshold=2)
    labeled_df = generator.run_pipeline(raw_df)

    # 2. Serialize structural inputs
    print("Step 3: Building serialized feature inputs...")
    def serialize(row):
        return (
            f"Assigned Priority: {row['Ticket Priority']} | "
            f"Channel: {row['Ticket Channel']} | "
            f"Domain: {row['Customer Domain']} | "
            f"Type: {row['Ticket Type']} | "
            f"Subject: {row['Ticket Subject']} | "
            f"Description: {row['Ticket Description']}"
        )
    texts = labeled_df.apply(serialize, axis=1).values

    # 3. Load Model checkpoint
    model_path = "./models/sia_deberta"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Please complete training first.")
        
    print(f"Step 4: Loading fine-tuned classifier from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Read optimal threshold from saved training configurations
    threshold_file = os.path.join(model_path, "threshold_config.json")
    decision_threshold = 0.5
    if os.path.exists(threshold_file):
        with open(threshold_file, "r") as f:
            config_data = json.load(f)
            decision_threshold = config_data.get("optimal_threshold", 0.5)
    print(f"Loaded decision boundary threshold: {decision_threshold:.2f}")
    
    # 4. Perform batch prediction
    print("Step 5: Executing batch classifier inference...")
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Analyzing tickets"):
            inputs = tokenizer(
                text,
                max_length=Config.MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            prob_mismatch = probs[0][1].item()
            
            # Apply our optimized decision threshold
            pred_class = 1 if prob_mismatch >= decision_threshold else 0
            conf = prob_mismatch if pred_class == 1 else (1.0 - prob_mismatch)
            
            predictions.append(pred_class)
            confidences.append(conf)

    labeled_df['predicted_mismatch'] = predictions
    labeled_df['confidence_score'] = confidences

    # 5. Generate structured dossiers for flagged mismatches
    print("Step 6: Compiling Evidence Dossiers for audited mismatches...")
    dossiers = []
    for idx, row in labeled_df.iterrows():
        if row['predicted_mismatch'] == 1:
            dossier = DossierEngine.generate(row.to_dict(), 1, row['confidence_score'])
            dossiers.append(dossier)

    # Save output predictions to CSV
    labeled_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] CSV predictions written to: {output_path}")

    # Save dossiers to JSON
    json_output_path = output_path.replace(".csv", "_dossiers.json")
    with open(json_output_path, "w") as f:
        json.dump(dossiers, f, indent=2)
    print(f"[SUCCESS] Evidence Dossiers written to: {json_output_path}")

    # Output preview summary
    print("\n=== AUDITOR METRIC HIGHLIGHTS ===")
    print(f"Total Tickets Scanned  : {len(labeled_df)}")
    print(f"Total Flagged Mismatches: {sum(predictions)} ({sum(predictions)/len(labeled_df):.1%})")
    print("=================================")

def main():
    parser = argparse.ArgumentParser(description="Support Integrity Auditor Inference CLI")
    parser.add_argument("--input_path", type=str, default="data/customer_support_tickets.csv", help="Path to input support tickets CSV")
    parser.add_argument("--output_path", type=str, default="output/predictions.csv", help="Path to save output results")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    run_inference(args.input_path, args.output_path)

if __name__ == "__main__":
    main()