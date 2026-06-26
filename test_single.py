import os
import sys
import pandas as pd
from google import genai

from main import load_user_history, load_evidence_requirements, analyze_claim, load_dotenv

def test():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.")
        sys.exit(1)
        
    client = genai.Client(api_key=api_key)
    user_history = load_user_history("dataset/user_history.csv")
    evidence_reqs = load_evidence_requirements("dataset/evidence_requirements.csv")
    
    claims_df = pd.read_csv("dataset/sample_claims.csv")
    row = claims_df.iloc[0].to_dict()
    
    print("Testing analyze_claim on first sample claim...")
    print("Row data:", row)
    
    try:
        res = analyze_claim(client, row, user_history, evidence_reqs, strategy="cot", base_dir="dataset")
        print("\nSUCCESS! Result:")
        print(res)
    except Exception as e:
        print("\nFAILED! Exception:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
