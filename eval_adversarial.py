# eval_adversarial.py
"""
Support Integrity Auditor (SIA) — Adversarial Robustness Evaluator.

This script tests the fine-tuned DeBERTa-v3-small model against 10
adversarially crafted tickets designed to break keyword-matching systems.
A score of >= 7/10 is required to secure the 10% submission bonus.
"""

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import Config, seed_everything

seed_everything()

def main():
    test_path = "data/adversarial_test.csv"
    model_path = "./models/sia_deberta"
    
    if not os.path.exists(test_path):
        print(f"Error: Missing adversarial test file at {test_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Missing trained model checkpoint at {model_path}")
        return

    # 1. Load data
    df = pd.read_csv(test_path)
    
    # Extract domain proxy
    df['Customer Domain'] = df['Customer Email'].apply(
        lambda x: str(x).split('@')[-1] if '@' in str(x) else 'unknown.com'
    )

    # 2. Serialize (Using the original Priority-First format to match your saved model)
    def serialize(row):
        return (
            f"Assigned Priority: {row['Ticket Priority']} | "
            f"Channel: {row['Ticket Channel']} | "
            f"Domain: {row['Customer Domain']} | "
            f"Type: {row['Ticket Type']} | "
            f"Subject: {row['Ticket Subject']} | "
            f"Description: {row['Ticket Description']}"
        )
    df['text'] = df.apply(serialize, axis=1)

    # 3. Load Model and Decision Threshold Config
    print("[INFO] Loading DeBERTa model and tokenizer checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    threshold_file = os.path.join(model_path, "threshold_config.json")
    decision_threshold = 0.5
    if os.path.exists(threshold_file):
        with open(threshold_file, "r") as f:
            decision_threshold = json.load(f).get("optimal_threshold", 0.5)
    print(f"[INFO] Loaded optimal decision boundary: {decision_threshold:.2f}")

    # 4. Run Batch Inference
    print("\nRunning adversarial evaluation...")
    correct_count = 0
    results = []

    for idx, row in df.iterrows():
        inputs = tokenizer(
            row['text'],
            max_length=Config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prob_mismatch = probs[0][1].item()
            
            # Apply our optimized decision threshold
            pred_class = 1 if prob_mismatch >= decision_threshold else 0
        
        is_correct = (pred_class == int(row['Expected_Mismatch']))
        if is_correct:
            correct_count += 1
            
        results.append({
            "ID": row["Ticket ID"],
            "Subject": row["Ticket Subject"][:30] + "...",
            "Priority": row["Ticket Priority"],
            "Expected": row["Expected_Mismatch"],
            "Predicted": pred_class,
            "Prob": f"{prob_mismatch:.4f}",
            "Status": "PASS ✅" if is_correct else "FAIL ❌"
        })

    # Display Results
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("                 ADVERSARIAL EVALUATION REPORT")
    print("="*80)
    print(results_df.to_string(index=False))
    print("-"*80)
    
    score = correct_count
    percentage = (score / 10.0)
    print(f"Final Robustness Score: {score}/10 ({percentage:.1%})")
    if score >= 7:
        print("Verdict: VERIFIED (Eligible for 10% submission bonus) 🎉")
    else:
        print("Verdict: FAILED (Did not meet the 7/10 threshold)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
