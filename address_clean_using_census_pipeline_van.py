import requests
import pandas as pd
import io
import os
import sys
import json
from dotenv import load_dotenv
import google.generativeai as genai

# --- Load credentials & Configure AI ---
load_dotenv()
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

if not GOOGLE_AI_API_KEY:
    print("Error: GOOGLE_AI_API_KEY not found in .env file.")
    sys.exit()
try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
except Exception as e:
    print(f"Error configuring Google AI. Please check your API key. Details: {e}")
    sys.exit()

# --- Helper Functions ---

def geocode_single_address(address_info):
    """Geocodes a single, structured address using the Census API."""
    url = "https://geocoding.geo.census.gov/geocoder/locations/address"
    try:
        params = {
            'street': address_info.get('street', ''),
            'city': address_info.get('city', ''),
            'state': address_info.get('state', ''),
            'zip': address_info.get('zip', ''),
            'benchmark': 'Public_AR_Current',
            'format': 'json'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result['result']['addressMatches']:
            return result['result']['addressMatches'][0]
    except Exception:
        # Fail silently on single re-geocode attempt
        pass
    return None

AI_PROMPT_TEMPLATE = """
Fix this address please
"{full_address_str}"
Format your answer as a single, valid JSON object with keys: "street", "city", "state", "zip".
If a value is not present, use an empty string "". Do not add any other text around the JSON object.
"""

def parse_ai_response(text_response):
    """Safely parses the JSON output from the AI model."""
    if not text_response: return None
    try:
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        data = json.loads(text_response)
        return {
            "street": data.get("street", ""),
            "city": data.get("city", ""),
            "state": data.get("state", ""),
            "zip": data.get("zip", "")
        }
    except Exception:
        return None

def correct_address_with_google_ai(model, original_row):
    """Uses the Gemini model to correct an address."""
    full_address_str = original_row.get('Input Address', '')
    prompt = AI_PROMPT_TEMPLATE.format(full_address_str=full_address_str)
    try:
        response = model.generate_content(prompt)
        return parse_ai_response(response.text)
    except Exception as e:
        print(f"    - An error occurred during the Google AI API call: {e}")
        return None

def update_row_with_geocoded_data(results_df, index, census_match):
    """Updates a row in the results DataFrame with new geocoded data."""
    if not census_match: return
    
    # Update status and matched address
    results_df.loc[index, 'Match Status'] = 'Match (AI Corrected)'
    results_df.loc[index, 'Match Type'] = census_match.get('matchType', 'Non_Exact')
    results_df.loc[index, 'Matched Address'] = census_match.get('matchedAddress')
    
    # Update coordinates and TIGER ID
    coords = census_match.get('coordinates', {})
    results_df.loc[index, 'Coordinates'] = f"{coords.get('x')},{coords.get('y')}"
    results_df.loc[index, 'TIGER Line ID'] = census_match.get('tigerLine', {}).get('tigerLineId')
    
    # Update FIPS codes
    geographies = census_match.get('geographies', {})
    tract_info = geographies.get('Census Tracts', [{}])[0]
    results_df.loc[index, 'State FIPS'] = tract_info.get('STATE')
    results_df.loc[index, 'County FIPS'] = tract_info.get('COUNTY')
    results_df.loc[index, 'Tract'] = tract_info.get('TRACT')
    results_df.loc[index, 'Block'] = tract_info.get('BLOCK')

# --- Main Geocoding Function ---

def geocode_csv(input_filepath, output_filepath):
    """
    Geocodes a CSV file and runs an AI correction pipeline on any failures.
    """
    print(f"--- Step 1: Starting Initial Batch Geocoding for {input_filepath} ---")
    url = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
    payload = {'benchmark': 'Public_AR_Current', 'vintage': 'Current_Current'}
    
    try:
        with open(input_filepath, 'rb') as address_file:
            files = {'addressFile': (os.path.basename(input_filepath), address_file, 'text/csv')}
            print("Sending data to the US Census API. This may take a moment...")
            response = requests.post(url, files=files, data=payload, timeout=300)
            response.raise_for_status()
        
        geocoded_data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), header=None, dtype=str).fillna('')
        column_names = ['Unique ID', 'Input Address', 'Match Status', 'Match Type', 'Matched Address', 'Coordinates', 'TIGER Line ID', 'Side', 'State FIPS', 'County FIPS', 'Tract', 'Block']
        geocoded_data.columns = column_names[:len(geocoded_data.columns)]
        print("Initial batch geocoding complete.")

    except Exception as e:
        print(f"An error occurred during initial batch geocoding: {e}")
        return

    # --- Step 2: AI Correction for 'No_Match' Rows ---
    failed_rows = geocoded_data[geocoded_data['Match Status'] == 'No_Match']
    print(f"\n--- Step 2: Found {len(failed_rows)} addresses for AI Correction ---")
    
    if len(failed_rows) > 0:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        ai_corrected_count = 0
        
        for index, row in failed_rows.iterrows():
            print(f"  - AI Fallback for Unique ID: {row['Unique ID']}...")
            
            corrected_address = correct_address_with_google_ai(gemini_model, row)
            
            if corrected_address:
                new_match = geocode_single_address(corrected_address)
                if new_match:
                    print("    - SUCCESS: AI suggestion was geocoded successfully.")
                    update_row_with_geocoded_data(geocoded_data, index, new_match)
                    ai_corrected_count += 1
        print(f"AI correction step complete. Successfully corrected {ai_corrected_count} addresses.")

    # --- Final Step: Calculate Accuracy and Save ---
    total_rows = len(geocoded_data)
    if total_rows > 0:
        successful_matches = len(geocoded_data[geocoded_data['Match Status'].str.contains('Match', na=False)])
        accuracy = (successful_matches / total_rows) * 100
        print(f"\n--- Final Accuracy: {accuracy:.2f}% ({successful_matches} of {total_rows} matched) ---")

    geocoded_data.to_csv(output_filepath, index=False)
    print(f"Geocoding complete! Results saved to {output_filepath}")

if __name__ == '__main__':
    # --- How to use the script ---
    # 1. Prepare an input CSV with columns: 'Unique ID', 'street', 'city', 'state', 'zip'
    input_filename = 'Data/van-complete-data-for-geocoding.csv'

    # 2. Define the name for your output file.
    output_filename = 'geocoded_results_with_ai.csv'

    # 3. Run the script!
    geocode_csv(input_filename, output_filename)