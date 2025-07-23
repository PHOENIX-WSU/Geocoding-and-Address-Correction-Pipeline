import requests
import pandas as pd
import io
import os
import sys
import time
import json
from dotenv import load_dotenv
from thefuzz import fuzz
import google.generativeai as genai

# --- Load credentials & Configuration ---
load_dotenv()
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

# Validate keys and configure AI
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
    """Geocodes a single, structured address."""
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
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        if result['result']['addressMatches']:
            return result['result']['addressMatches'][0]
    except Exception as e:
        print(f"    - Single address geocoding error: {e}")
    return None

AI_PROMPT_TEMPLATE = """
Fix this address please:
"{full_address_str}"
Format your answer as a single, valid JSON object with keys: "street_address", "secondary_address", "city", "state", "zip_code".
If a value is not present, use an empty string "". Do not add any other text around the JSON object.
"""

def parse_ai_response(text_response):
    if not text_response: return None
    try:
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        data = json.loads(text_response)
        return {
            "street": data.get("street_address", ""),
            "city": data.get("city", ""),
            "state": data.get("state", ""),
            "zip": data.get("zip_code", "")
        }
    except Exception:
        return None

def correct_address_with_google_ai(model, original_row):
    """Uses the Gemini model to correct an address."""
    # This now uses the original column names from the input file
    full_address_str = f"{original_row.get('street', '')}, {original_row.get('city', '')}, {original_row.get('state', '')} {original_row.get('zip', '')}".strip()
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
    
    results_df.loc[index, 'Match Status'] = 'Match'
    results_df.loc[index, 'Match Type'] = census_match.get('matchType', 'Non_Exact')
    results_df.loc[index, 'Matched Address'] = census_match.get('matchedAddress')
    
    coords = census_match.get('coordinates', {})
    results_df.loc[index, 'Coordinates'] = f"{coords.get('x')},{coords.get('y')}"
    
    geographies = census_match.get('geographies', {})
    tract_info = geographies.get('Census Tracts', [{}])[0]
    
    results_df.loc[index, 'State FIPS'] = tract_info.get('STATE')
    results_df.loc[index, 'County FIPS'] = tract_info.get('COUNTY')
    results_df.loc[index, 'Tract'] = tract_info.get('TRACT')
    results_df.loc[index, 'Block'] = tract_info.get('BLOCK')
    
# --- Main Geocoding and Correction Function ---

def geocode_and_correct_csv(input_filepath, output_filepath):
    """
    Geocodes a CSV and runs a correction pipeline on any failures.
    """
    print(f"--- Starting Step 1: Initial Batch Geocoding for {input_filepath} ---")
    url = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
    payload = {'benchmark': 'Public_AR_Current', 'vintage': 'Current_Current'}
    
    try:
        with open(input_filepath, 'rb') as address_file:
            files = {'addressFile': (os.path.basename(input_filepath), address_file, 'text/csv')}
            response = requests.post(url, files=files, data=payload)
            response.raise_for_status()
        
        geocoded_data = pd.read_csv(io.StringIO(response.text), header=None, dtype=str).fillna('')
        column_names = ['Unique ID', 'Input Address', 'Match Status', 'Match Type', 'Matched Address', 'Coordinates', 'TIGER Line ID', 'Side', 'State FIPS', 'County FIPS', 'Tract', 'Block']
        geocoded_data.columns = column_names[:len(geocoded_data.columns)]
        print("Initial batch geocoding complete.")

    except Exception as e:
        print(f"An error occurred during initial batch geocoding: {e}")
        return

    # --- Step 2: Begin Correction Pipeline for 'No_Match' Rows ---
    print("\n--- Starting Step 2: Correction Pipeline for Failed Addresses ---")
    original_df = pd.read_csv(input_filepath, dtype=str).fillna('')
    failed_rows = geocoded_data[geocoded_data['Match Status'] == 'No_Match']
    print(f"Found {len(failed_rows)} addresses that need correction.")
    
    if len(failed_rows) == 0:
        geocoded_data.to_csv(output_filepath, index=False)
        print(f"\nAll addresses matched! Results saved to {output_filepath}")
        return

    # --- Programmatic Corrections (PO Box, Misplaced ZIP) ---
    # These are simple string manipulations on the original data
    
    # --- Conditional Lastname Matching ---
    if 'demo_lastname' in original_df.columns and 'demo_zip' in original_df.columns:
        print("\nAttempting correction via Lastname Matching...")
        match_count = 0
        successful_rows = geocoded_data[geocoded_data['Match Status'] == 'Match']
        for index, row in failed_rows.iterrows():
            original_input = original_df.loc[int(row['Unique ID'])]
            potential_matches = successful_rows[
                (original_df['demo_lastname'] == original_input['demo_lastname']) &
                (original_df['demo_zip'] == original_input['demo_zip'])
            ]
            if not potential_matches.empty:
                # Copy the data from the first successful match
                geocoded_data.loc[index] = potential_matches.iloc[0].values
                match_count += 1
        print(f"Corrected {match_count} addresses via lastname matching.")

    # --- AI Correction Fallback ---
    print("\nAttempting AI Correction for remaining failures...")
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    ai_corrected_count = 0
    
    # Re-check for failures after potential lastname matching
    rows_for_ai = geocoded_data[geocoded_data['Match Status'] == 'No_Match']
    
    for index, row in rows_for_ai.iterrows():
        unique_id = int(row['Unique ID'])
        original_row_data = original_df.loc[unique_id]
        print(f"  - AI Fallback for Unique ID: {unique_id}...")
        
        corrected_address = correct_address_with_google_ai(gemini_model, original_row_data)
        
        if corrected_address:
            # Re-geocode the AI's suggestion
            new_match = geocode_single_address(corrected_address)
            if new_match:
                print("    - SUCCESS: AI suggestion was geocoded successfully.")
                update_row_with_geocoded_data(geocoded_data, index, new_match)
                ai_corrected_count += 1

    print(f"Corrected {ai_corrected_count} addresses via AI.")

    # --- Final Step: Save Results ---
    geocoded_data.to_csv(output_filepath, index=False)
    print(f"\nCorrection pipeline complete! Results saved to {output_filepath}")

if __name__ == '__main__':
    # --- How to use the script ---
    # 1. Prepare your input CSV with columns: 'Unique ID', 'street', 'city', 'state', 'zip'
    #    (and optional 'demo_lastname' for matching)
    input_filename = 'Data/van-complete-data-for-geocoding.csv'

    # 2. Define the name for your final output file.
    output_filename = 'geocoded_and_corrected_results.csv'

    # 3. Run the script!
    geocode_and_correct_csv(input_filename, output_filename)