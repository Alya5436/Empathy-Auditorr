import pandas as pd
import time
import os
from google import genai

# 1. Setup Gemini Client
# Ensure this is your Google AI Studio API Key (starts with AIza)
API_KEY = ""
client = genai.Client(api_key=API_KEY)

# 2. Load Data & Handle Resuming
if os.path.exists("responses.csv"):
    print("Resuming from existing responses.csv...")
    df = pd.read_csv("responses.csv")
else:
    print("Starting fresh from prompts.csv...")
    df = pd.read_csv("prompts.csv")
    if 'response' not in df.columns:
        df['response'] = None

print(f"Total prompts to process: {len(df)}")

# 3. The Robust Generation Loop
for i, row in df.iterrows():
    # Skip if we already have a successful response
    current_res = str(row.get('response', ''))
    if pd.notnull(row.get('response')) and current_res != "" and "ERROR" not in current_res and "FAILED" not in current_res:
        continue

    prompt = row["prompt"]
    success = False
    retries = 0

    # Retry logic for Rate Limits (429)
    while not success and retries < 5: # Increased retries for a "small bucket"
        try:
            # Call Gemini 2.5 Flash
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=f"System: You are an empathetic assistant.\nUser: {prompt}"
            )
            
            if response.text:
                df.at[i, 'response'] = response.text.strip()
                print(f"[{i+1}/{len(df)}] SUCCESS")
                success = True
            else:
                df.at[i, 'response'] = "[BLOCKED BY SAFETY FILTER]"
                print(f"[{i+1}/{len(df)}] BLOCKED (Safety Filter)")
                success = True 

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                retries += 1
                wait_time = 60 * retries # Wait 60s, then 120s, etc.
                print(f"[{i+1}/{len(df)}] RATE LIMIT. Waiting {wait_time}s (Attempt {retries}/5)...")
                time.sleep(wait_time)
            else:
                print(f"[{i+1}/{len(df)}] PERMANENT ERROR: {error_msg}")
                df.at[i, 'response'] = f"FAILED: {error_msg[:50]}"
                break 

    # 4. Save and "Safe Mode" Delay
    # Save every row so progress is 100% safe
    df.to_csv("responses.csv", index=False)
    
    # 20-second delay to guarantee we stay under the "New Account" limit
    if success:
        time.sleep(30)


print("\n--- ALL DONE! ---")
print("Your empathetic dataset is ready in 'responses.csv'.")