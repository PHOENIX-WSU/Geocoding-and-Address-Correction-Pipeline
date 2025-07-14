import requests
import pandas as pd
import os
import sys
import time
import json
from dotenv import load_dotenv
from thefuzz import fuzz
import google.generativeai as genai

# --- Load credentials from .env file ---
load_dotenv()

# For USPS API
CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")

# For Google AI API
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

# --- NEW: Configuration for your Local AI Model ---
# ❗ IMPORTANT: Change this URL to the actual address of your local model server.
LOCAL_API_ENDPOINT = "http://localhost:11434/v1/chat/completions"
# This is the name of the model as your local server knows it
LOCAL_MODEL_NAME = "gemma3:12b"

# --- Configuration & Validation ---
if not CONSUMER_KEY or not CONSUMER_SECRET:
    print("Error: USPS CONSUMER_KEY or CONSUMER_SECRET not found in .env file.")
    sys.exit()
if not GOOGLE_AI_API_KEY:
    print("Error: GOOGLE_AI_API_KEY not found in .env file.")
    sys.exit()

try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
except Exception as e:
    print(f"Error configuring Google AI. Please check your API key. Details: {e}")
    sys.exit()

# --- API endpoints ---
TOKEN_URL = "https://apis.usps.com/oauth2/v3/token"
ADDRESS_API_URL = "https://apis.usps.com/addresses/v3/address"

# --- Helper Functions ---
def get_access_token(client_id, client_secret):
    # This function is unchanged
    headers = {"Content-Type": "application/json"}
    data = {"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"}
    print("Requesting access token...")
    try:
        response = requests.post(TOKEN_URL, headers=headers, json=data)
        response.raise_for_status()
        token_data = response.json()
        print("Successfully received access token!")
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error getting token: {e.response.status_code} {e.response.text}")
    except Exception as e:
        print(f"An error occurred getting token: {e}")
    return None

def clean_single_address(access_token, address_info):
    # This function is unchanged
    if not access_token:
        return None, "Skipped - No Access Token"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    params = {
        "streetAddress": address_info.get("street_1"), "secondaryAddress": address_info.get("street_2"),
        "city": address_info.get("city"), "state": address_info.get("state"), "ZIPCode": address_info.get("zip_code"),
    }
    try:
        response = requests.get(ADDRESS_API_URL, headers=headers, params=params)
        response.raise_for_status()
        json_response = response.json()
        if "errors" in json_response:
            error_message = json_response["errors"][0].get("message", "Address not found")
            print(f"   - API Validation Error for '{params.get('streetAddress')}': {error_message}")
            return None, "API Validation Error"
        return json_response, "Success"
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        print(f"   - API HTTP Error for address '{params.get('streetAddress')}': {status_code}")
        return None, f"API HTTP Error: {status_code}"
    except Exception as e:
        print(f"   - An unknown error occurred cleaning address: {e}")
        return None, "Unknown Error"

# --- AI Helper Functions ---

# NEW: Simplified and more effective prompt
AI_PROMPT_TEMPLATE = """
Please correct the following US address. Fix the address please

Address to correct:
"{full_address_str}"

After you have corrected it, please format your final answer as a single, valid JSON object with the following keys: "street_address", "secondary_address", "city", "state", "zip_code".
If a value is not present (like a secondary address), use an empty string "".
Do not add any other text, explanation, or markdown formatting around the JSON object.
"""

# NEW: Upgraded function to parse JSON output
def parse_ai_response(text_response):
    """Parses the JSON output from an AI model."""
    if not text_response:
        return None
    try:
        # The model might wrap the JSON in markdown, so we extract it
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        
        data = json.loads(text_response)
        
        # Map the JSON keys to the keys our USPS function expects
        return {
            "street_1": data.get("street_address", ""),
            "street_2": data.get("secondary_address", ""),
            "city": data.get("city", ""),
            "state": data.get("state", ""),
            "zip_code": data.get("zip_code", "")
        }
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"    - AI returned non-JSON or malformed data: '{text_response}'. Error: {e}")
        return None

def correct_address_with_local_ai(original_row):
    # This function now uses the new prompt and parser
    full_address_str = f"{original_row.get('demo_address', '')} {original_row.get('demo_address2', '')}, {original_row.get('demo_city', '')}, {original_row.get('demo_state', '')} {original_row.get('demo_zip', '')}".strip()
    prompt = AI_PROMPT_TEMPLATE.format(full_address_str=full_address_str)
    headers = {"Content-Type": "application/json"}
    data = {"model": LOCAL_MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "stream": False}
    try:
        print(f"    - Attempting correction with local model '{LOCAL_MODEL_NAME}'...")
        response = requests.post(LOCAL_API_ENDPOINT, headers=headers, json=data, timeout=60) # Added timeout
        response.raise_for_status()
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        return parse_ai_response(content)
    except Exception as e:
        print(f"    - An error occurred calling local AI model: {e}")
        return None

def correct_address_with_google_ai(model, original_row):
    # This function now uses the new prompt and parser
    full_address_str = f"{original_row.get('demo_address', '')} {original_row.get('demo_address2', '')}, {original_row.get('demo_city', '')}, {original_row.get('demo_state', '')} {original_row.get('demo_zip', '')}".strip()
    prompt = AI_PROMPT_TEMPLATE.format(full_address_str=full_address_str)
    try:
        print(f"    - Attempting correction with Google AI model...")
        response = model.generate_content(prompt)
        return parse_ai_response(response.text)
    except Exception as e:
        print(f"    - An error occurred during the Google AI API call: {e}")
        return None

def populate_cleaned_columns(df, index, address_part):
    # This function is unchanged
    df.loc[index, 'cleaned_street_address'] = address_part.get('streetAddress')
    df.loc[index, 'cleaned_secondary_address'] = address_part.get('secondaryAddress')
    df.loc[index, 'cleaned_city'] = address_part.get('city')
    df.loc[index, 'cleaned_state'] = address_part.get('state')
    df.loc[index, 'cleaned_zip'] = address_part.get('ZIPCode')
    df.loc[index, 'cleaned_zip_plus_4'] = address_part.get('ZIPPlus4')


# --- Main part of the script ---
if __name__ == "__main__":
    
    input_csv_path = "Data/patientDEMOGRAPHICS-11th-July-2025.csv"
    output_csv_path = "Data/patientDEMOGRAPHICS-11th-July-2025-Output.csv"

    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at '{input_csv_path}'")
    else:
        token = get_access_token(CONSUMER_KEY, CONSUMER_SECRET)
        if token:
            df = pd.read_csv(input_csv_path, dtype={"demo_zip": str})
            original_df = df.copy()
            df = df.dropna(subset=["demo_address"])
            print(f"\nRead {len(df)} rows from '{input_csv_path}'. Starting cleaning process...")

            # The rest of the script (all correction steps) is the same.
            # It will now just use the more effective AI helpers.
            new_cols = ['cleaned_street_address', 'cleaned_secondary_address', 'cleaned_city', 'cleaned_state', 'cleaned_zip', 'cleaned_zip_plus_4', 'api_status']
            for col in new_cols:
                df[col] = None

            for index, row in df.iterrows():
                raw_zip = row.get("demo_zip", "")
                zip_for_api = raw_zip[:5] if isinstance(raw_zip, str) else ""
                address_to_clean = {
                    "street_1": row.get("demo_address"), "street_2": row.get("demo_address2"),
                    "city": row.get("demo_city"), "state": row.get("demo_state"), "zip_code": zip_for_api
                }
                cleaned_data, status = clean_single_address(token, address_to_clean)
                df.loc[index, 'api_status'] = status

                # Sleep to stay under API limit
                time.sleep(0.15) 
                
                if status == 'Success' and cleaned_data and 'address' in cleaned_data:
                    populate_cleaned_columns(df, index, cleaned_data.get("address", {}))
                    print(f"   - Cleaned address for row {index + 1}")

            failed_statuses = ['API HTTP Error: 400', 'API Validation Error', 'Failed']

            print("\nCorrection Step 1: Checking for 'PO' errors to retry...")
            retry_count_po = 0
            rows_to_retry_indices = df[df['api_status'].isin(failed_statuses)].index.tolist()
            for index in rows_to_retry_indices:
                row = df.loc[index]
                original_address = str(row.get('demo_address', ''))
                parts = original_address.split()
                if len(parts) > 1 and parts[0].upper() == 'PO' and parts[1].isdigit():
                    new_address = 'PO Box ' + ' '.join(parts[1:])
                    address_to_clean = {"street_1": new_address, "street_2": row.get("demo_address2"), "city": row.get("demo_city"), "state": row.get("demo_state"), "zip_code": str(row.get("demo_zip", ""))[:5]}
                    cleaned_data, status = clean_single_address(token, address_to_clean)

                    time.sleep(0.15)
                    
                    if status == 'Success':
                        retry_count_po += 1
                        df.loc[index, 'api_status'] = 'Success (Corrected PO Box)'
                        if cleaned_data and 'address' in cleaned_data:
                            populate_cleaned_columns(df, index, cleaned_data.get("address", {}))
            print(f"   - Corrected {retry_count_po} 'PO Box' addresses.")

            print("\nCorrection Step 2: Checking for misplaced ZIP Codes to retry...")
            retry_count_zip = 0
            rows_to_check_zip = df[df['api_status'].isin(failed_statuses)].index.tolist()
            for index in rows_to_check_zip:
                row = df.loc[index]
                zip_is_missing = pd.isna(row.get('demo_zip')) or str(row.get('demo_zip', '')).strip() == ''
                if zip_is_missing:
                    addr2_val, city_val, state_val = str(row.get('demo_address2', '')).strip(), str(row.get('demo_city', '')).strip(), str(row.get('demo_state', '')).strip()
                    misplaced_zip, source_col_name = (addr2_val, 'demo_address2') if addr2_val.isdigit() and len(addr2_val) == 5 else (city_val, 'demo_city') if city_val.isdigit() and len(city_val) == 5 else (state_val, 'demo_state') if state_val.isdigit() and len(state_val) == 5 else (None, None)
                    if misplaced_zip:
                        address_to_clean = {"street_1": row.get("demo_address"), "street_2": "" if source_col_name == 'demo_address2' else row.get("demo_address2"), "city": "" if source_col_name == 'demo_city' else row.get("demo_city"), "state": "" if source_col_name == 'demo_state' else row.get("demo_state"), "zip_code": misplaced_zip}
                        cleaned_data, status = clean_single_address(token, address_to_clean)
                        
                        time.sleep(0.15)
                        
                        if status == 'Success':
                            retry_count_zip += 1
                            df.loc[index, 'demo_zip'], df.loc[index, source_col_name] = misplaced_zip, ''
                            df.loc[index, 'api_status'] = 'Success (Corrected ZIP Location)'
                            if cleaned_data and 'address' in cleaned_data:
                                populate_cleaned_columns(df, index, cleaned_data.get("address", {}))
            print(f"   - Corrected {retry_count_zip} misplaced ZIP addresses.")

            print("\nCorrection Step 3: Matching remaining failed rows by lastname (fallback)...")
            failed_df = df[df['api_status'].isin(failed_statuses)].copy()
            success_df = df[df['api_status'].str.startswith('Success', na=False)].copy()
            match_count = 0
            for failed_index, failed_row in failed_df.iterrows():
                potential_matches = success_df[(success_df['demo_lastname'] == failed_row['demo_lastname']) & (success_df['demo_zip'] == failed_row['demo_zip']) & (success_df['demo_state'] == failed_row['demo_state']) & (success_df['demo_city'] == failed_row['demo_city'])]
                for _, match_row in potential_matches.iterrows():
                    addr1, addr2 = str(failed_row.get('demo_address', '')), str(match_row.get('demo_address', ''))
                    if fuzz.ratio(addr1, addr2) > 50:
                        for col in new_cols:
                            if col.startswith('cleaned_'):
                                df.loc[failed_index, col] = match_row[col]
                        df.loc[failed_index, 'api_status'] = 'Matched on Lastname'
                        match_count += 1
                        break
            print(f"   - Corrected {match_count} rows through lastname matching.")

            print("\nCorrection Step 4: Using AI for remaining failed addresses...")
            gemini_flash_model = genai.GenerativeModel('gemini-2.0-flash')
            ai_correction_count = 0
            final_failed_indices = df[df['api_status'].isin(failed_statuses)].index.tolist()
            for index in final_failed_indices:
                print(f"\n  - Processing final failure in row {index + 1} with AI...")
                original_row_data = original_df.loc[index]
                
                corrected_dict = correct_address_with_local_ai(original_row_data)
                if corrected_dict:
                    cleaned_data, status = clean_single_address(token, corrected_dict)
                    
                    time.sleep(0.15)
                    
                    if status == 'Success':
                        print("    - SUCCESS: Local model suggestion was validated by USPS.")
                        ai_correction_count += 1
                        df.loc[index, 'api_status'] = f'Success (AI - Local)'
                        if cleaned_data and 'address' in cleaned_data:
                            populate_cleaned_columns(df, index, cleaned_data.get("address", {}))
                        continue
                
                time.sleep(1)

                print("    - Local model suggestion failed. Trying cloud model (Gemini Flash)...")
                corrected_dict = correct_address_with_google_ai(gemini_flash_model, original_row_data)
                if corrected_dict:
                    cleaned_data, status = clean_single_address(token, corrected_dict)
                    
                    time.sleep(0.15)
                    
                    if status == 'Success':
                        print("    - SUCCESS: Gemini Flash suggestion was validated by USPS.")
                        ai_correction_count += 1
                        df.loc[index, 'api_status'] = 'Success (AI - Flash)'
                        if cleaned_data and 'address' in cleaned_data:
                           populate_cleaned_columns(df, index, cleaned_data.get("address", {}))
                        continue

                df.loc[index, 'api_status'] = 'Failed (AI Uncorrectable)'
                print("    - FAILURE: Both AI models failed to produce a valid address.")
            print(f"\n   - Corrected {ai_correction_count} rows using AI models.")

            total_rows = len(df)
            if total_rows > 0:
                total_successful = len(df[df['api_status'].str.startswith('Success', na=False)])
                success_rate = (total_successful / total_rows) * 100
                print(f"\nOverall Success Rate: {success_rate:.2f}% ({total_successful} of {total_rows} valid addresses)")
            else:
                print("\nNo addresses were processed.")

            df.to_csv(output_csv_path, index=False)
            print(f"\nAll done! Cleaned data saved to '{output_csv_path}'.")
