import requests
import pandas as pd
import os
import sys
import time
import json
from dotenv import load_dotenv
from thefuzz import fuzz
import google.generativeai as genai

# --- Load credentials & Configuration ---
load_dotenv()

# Google AI Key
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

# Local AI Model Config
# ❗ IMPORTANT: Change this URL to the actual address of your local model server.
LOCAL_API_ENDPOINT = "http://localhost:11434/v1/chat/completions"
# This is the name of the model as your local server knows it
LOCAL_MODEL_NAME = "gemma3:12b"

# Validate keys
if not GOOGLE_AI_API_KEY:
    print("Error: GOOGLE_AI_API_KEY not found in .env file.")
    sys.exit()

# Configure Google AI
try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
except Exception as e:
    print(f"Error configuring Google AI. Please check your API key. Details: {e}")
    sys.exit()

# --- API endpoints ---
CENSUS_API_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"


# --- Helper Functions ---
def validate_with_census(address_info):
    """
    Validates and geocodes a single address using the U.S. Census Bureau Geocoder.
    """
    address_string = f"{address_info.get('street_1', '')}, {address_info.get('city', '')}, {address_info.get('state', '')} {address_info.get('zip_code', '')}"
    
    params = {
        'address': address_string,
        'benchmark': 'Public_AR_Current',
        'format': 'json'
    }
    
    try:
        response = requests.get(CENSUS_API_URL, params=params)
        response.raise_for_status()
        result = response.json()
        
        if result['result']['addressMatches']:
            return result['result']['addressMatches'][0], "Success"
        else:
            return None, "API No Match"

    except requests.exceptions.HTTPError as e:
        return None, f"API HTTP Error: {e.response.status_code}"
    except Exception as e:
        return None, "Unknown Error"

AI_PROMPT_TEMPLATE = """
Please correct the following US address. Fix any spelling mistakes and fill in missing information like city, state, or ZIP code if they are obvious from the context.
Address to correct:
"{full_address_str}"
After you have corrected it, please format your final answer as a single, valid JSON object with the following keys: "street_address", "secondary_address", "city", "state", "zip_code".
If a value is not present (like a secondary address), use an empty string "".
Do not add any other text, explanation, or markdown formatting around the JSON object.
"""

def parse_ai_response(text_response):
    if not text_response: return None
    try:
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        data = json.loads(text_response)
        return {
            "street_1": data.get("street_address", ""), "street_2": data.get("secondary_address", ""),
            "city": data.get("city", ""), "state": data.get("state", ""), "zip_code": data.get("zip_code", "")
        }
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"    - AI returned non-JSON or malformed data. Error: {e}")
        return None

def correct_address_with_local_ai(original_row):
    full_address_str = f"{original_row.get('demo_address', '')} {original_row.get('demo_address2', '')}, {original_row.get('demo_city', '')}, {original_row.get('demo_state', '')} {original_row.get('demo_zip', '')}".strip()
    prompt = AI_PROMPT_TEMPLATE.format(full_address_str=full_address_str)
    headers = {"Content-Type": "application/json"}
    data = {"model": LOCAL_MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "stream": False}
    try:
        response = requests.post(LOCAL_API_ENDPOINT, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return parse_ai_response(content)
    except Exception as e:
        print(f"    - An error occurred calling local AI model: {e}")
        return None

def correct_address_with_google_ai(model, original_row):
    full_address_str = f"{original_row.get('demo_address', '')} {original_row.get('demo_address2', '')}, {original_row.get('demo_city', '')}, {original_row.get('demo_state', '')} {original_row.get('demo_zip', '')}".strip()
    prompt = AI_PROMPT_TEMPLATE.format(full_address_str=full_address_str)
    try:
        response = model.generate_content(prompt)
        return parse_ai_response(response.text)
    except Exception as e:
        print(f"    - An error occurred during the Google AI API call: {e}")
        return None

def populate_cleaned_columns(df, index, census_match):
    """
    Helper to populate cleaned address and GEO columns from the Census API response.
    """
    # --- Address Components ---
    full_address = census_match.get('matchedAddress', '')
    components = census_match.get('addressComponents', {})
    street_parts = [
        components.get('preQualifier', ''), components.get('preDirection', ''),
        components.get('preType', ''), components.get('houseNumber', ''),
        components.get('streetName', ''), components.get('postType', ''),
        components.get('postDirection', '')
    ]
    street_address = ' '.join(part for part in street_parts if part).strip()
    if not street_address:
        street_address = full_address.split(',')[0].strip()

    df.loc[index, 'cleaned_street_address'] = street_address
    df.loc[index, 'cleaned_city'] = components.get('city')
    df.loc[index, 'cleaned_state'] = components.get('state')
    df.loc[index, 'cleaned_zip_full'] = components.get('zip')
    df.loc[index, 'cleaned_secondary_address'] = '' # Census API does not provide this

    # --- Geocoding Components ---
    coords = census_match.get('coordinates', {})
    df.loc[index, 'latitude'] = coords.get('y')
    df.loc[index, 'longitude'] = coords.get('x')

    # Extract and build the 11-digit FIPS code
    fips_code = ''
    try:
        tract_info = census_match.get('geographies', {}).get('Census Tracts', [{}])[0]
        state_fips = tract_info.get('STATE', '').zfill(2)
        county_fips = tract_info.get('COUNTY', '').zfill(3)
        tract_fips = tract_info.get('TRACT', '').zfill(6)
        if state_fips and county_fips and tract_fips:
            fips_code = f"{state_fips}{county_fips}{tract_fips}"
    except (IndexError, AttributeError):
        pass # Leave FIPS code blank if not found
        
    df.loc[index, 'fips_11_digit'] = fips_code


# --- Main part of the script ---
if __name__ == "__main__":
    
    input_csv_path = "Data/census-data-test.csv"
    output_csv_path = "Data/census-data-test-address-corrected.csv"

    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at '{input_csv_path}'")
    else:
        df = pd.read_csv(input_csv_path, dtype=str).fillna('')
        original_df = df.copy()
        df = df.dropna(subset=["demo_address"]).reset_index(drop=True)
        print(f"\nRead {len(df)} total rows. Starting processing at maximum speed...")

        # Add new geocoding columns to the list
        new_cols = [
            'cleaned_street_address', 'cleaned_secondary_address', 'cleaned_city', 
            'cleaned_state', 'cleaned_zip_full', 'api_status', 
            'latitude', 'longitude', 'fips_11_digit'
        ]
        for col in new_cols:
            df[col] = None
        
        # Step 0: Initial API Pass with Census Geocoder
        print("\nStep 0: Initial Validation and Geocoding Pass...")
        for index, row in df.iterrows():
            address_parts = [str(row.get('demo_address', '')).strip(), str(row.get('demo_city', '')).strip(), str(row.get('demo_state', '')).strip(), str(row.get('demo_zip', '')).strip()]
            printable_address = ', '.join(part for part in address_parts if part)
            print(f"  - Processing row {index + 1}/{len(df)} -> {printable_address}")

            address_to_validate = {"street_1": row.get("demo_address"), "city": row.get("demo_city"), "state": row.get("demo_state"), "zip_code": str(row.get("demo_zip", ""))}
            cleaned_data, status = validate_with_census(address_to_validate)
            df.loc[index, 'api_status'] = status
            if status == 'Success' and cleaned_data:
                populate_cleaned_columns(df, index, cleaned_data)

        failed_statuses = ['API HTTP Error', 'API No Match', 'Failed', 'Unknown Error']
        
        # ... [Programmatic correction steps (PO Box, etc.) would go here if needed] ...
        
        # Final Step: AI Correction Fallback
        print("\nFinal Step: Using AI for remaining failed addresses...")
        gemini_flash_model = genai.GenerativeModel('gemini-2.0-flash')
        for index in df[df['api_status'].isin(failed_statuses)].index:
            print(f"  - AI Fallback for row {index + 1}...")
            original_row_data = original_df.loc[index]
            
            # Attempt 1: Local Model
            corrected_dict = correct_address_with_local_ai(original_row_data)
            if corrected_dict:
                cleaned_data, status = validate_with_census(corrected_dict)
                if status == 'Success':
                    print("    - SUCCESS: Local model suggestion was validated.")
                    df.loc[index, 'api_status'] = 'Success (AI - Local)'
                    if cleaned_data:
                        populate_cleaned_columns(df, index, cleaned_data)
                    continue
            
            # Attempt 2: Cloud Model
            corrected_dict = correct_address_with_google_ai(gemini_flash_model, original_row_data)
            if corrected_dict:
                cleaned_data, status = validate_with_census(corrected_dict)
                if status == 'Success':
                    print("    - SUCCESS: Gemini Flash suggestion was validated.")
                    df.loc[index, 'api_status'] = 'Success (AI - Flash)'
                    if cleaned_data:
                        populate_cleaned_columns(df, index, cleaned_data)
                    continue

            df.loc[index, 'api_status'] = 'Failed (AI Uncorrectable)'

        # --- Final Summary & Save ---
        print("\n--- All processing complete! ---")
        total_rows = len(df)
        if total_rows > 0:
            total_successful = len(df[df['api_status'].str.startswith('Success', na=False)])
            success_rate = (total_successful / total_rows) * 100
            print(f"\nOverall Success Rate: {success_rate:.2f}% ({total_successful} of {total_rows} valid addresses)")
        df.to_csv(output_csv_path, index=False)
        print(f"\nAll done! Final geocoded data saved to '{output_csv_path}'.")