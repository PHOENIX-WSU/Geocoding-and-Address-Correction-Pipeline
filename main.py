import os
import sys
import json
import io
import argparse
import pandas as pd
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

# --- Configuration for Different Data Sources ---
# Define all unique settings for your datasets here.
# To add a new source, just create a new entry in this dictionary.
CONFIG = {
    'demographics': {
        'input_file': 'Data/patientDEMOGRAPHOGRAPHICS-11th-July-2025.csv',
        'geocoding_input': 'Data/prepared_demographics_for_geocoding.csv',
        'geocoded_output': 'geocoded_results/demographics_geocoded_raw.csv',
        'final_output': 'geocoded_results/Production/demographics_final_for_bigquery.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'demo_address',
            'city': 'demo_city',
            'state': 'demo_state',
            'zip': 'demo_zip'
        },
        'dataset_name': 'Patient Demographics (Geocoded)',
        'add_state_mi': False # This dataset already has a state column
    },
    'van': {
        'input_file': 'Production-Code-Tests/Data/van.csv',
        'geocoding_input': 'Production-Code-Tests/Data/prepared_van_for_geocoding.csv',
        'geocoded_output': 'Production-Code-Tests/geocoded_results/van_geocoded_raw.csv',
        'final_output': 'Production-Code-Tests/geocoded_results/Production/van_final_for_bigquery.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'street',
            'city': 'city',
            'state': 'state',
            'zip': 'zip'
        },
        'dataset_name': 'MHU Field Data: Van deployment addresses (Geocoded)',
        'add_state_mi': True # This dataset needs 'MI' added
    }
}

# --- AI and API Configuration ---
AI_PROMPT_TEMPLATE = """
Fix this address please: "{full_address_str}".
Format your answer as a single, valid JSON object with keys: "street", "city", "state", "zip".
If a value is not present, use an empty string "". Do not add any other text around the JSON object.
"""

def load_environment_and_models():
    """Loads environment variables and configures the AI models."""
    print("Loading environment variables and configuring AI...")
    load_dotenv()
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_AI_API_KEY not found in .env file.")
        sys.exit(1)
    try:
        genai.configure(api_key=api_key)
        models = {
            'flash': genai.GenerativeModel('gemini-1.5-flash-latest'),
            'pro': genai.GenerativeModel('gemini-1.5-pro-latest')
        }
        print("✅ Google AI configured successfully.")
        return models
    except Exception as e:
        print(f"Error configuring Google AI. Please check your API key. Details: {e}")
        sys.exit(1)

# --- Stage 1: Data Preparation ---
def prepare_for_geocoding(config):
    """
    Reads a source CSV, cleans it, and formats it for the Census Geocoder.
    """
    print(f"\n--- Stage 1: Preparing data from '{config['input_file']}' ---")
    try:
        # Read the source data, keeping all data as strings
        df = pd.read_csv(config['input_file'], dtype=str).fillna('')
        config['original_data_df'] = df.copy() # Save original for final merge
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{config['input_file']}'.")
        sys.exit(1)

    # Add 'MI' as the state if the config requires it
    if config.get('add_state_mi', False):
        df['state'] = 'MI'
        print("Hardcoded 'MI' as the state.")

    # Standardize column names based on the map in the config
    col_map = config['column_map']
    
    # Check if all required columns exist
    for key, val in col_map.items():
        if val not in df.columns:
            print(f"❌ Error: Required column '{val}' not found in the input file.")
            sys.exit(1)
            
    df_geocoder = pd.DataFrame()
    df_geocoder['Street Address'] = df[col_map['street']]
    df_geocoder['City'] = df[col_map['city']]
    df_geocoder['State'] = df[col_map['state']]
    df_geocoder['ZIP'] = df[col_map['zip']]

    # CRITICAL: Remove rows with no street address
    original_rows = len(df_geocoder)
    df_geocoder = df_geocoder[df_geocoder['Street Address'].str.strip() != ''].copy()
    print(f"Removed {original_rows - len(df_geocoder)} rows with an empty street address.")

    # Create a Unique ID for re-joining data later
    df_geocoder.insert(0, 'Unique ID', range(len(df_geocoder)))
    
    # Link the Unique ID back to the original dataframe for the final merge
    config['original_data_df'] = config['original_data_df'].iloc[df_geocoder.index].copy()
    config['original_data_df']['Unique ID'] = df_geocoder['Unique ID']


    # Save the prepared file
    output_path = config['geocoding_input']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_geocoder.to_csv(output_path, index=False)
    print(f"✅ Prepared file saved to: {output_path}")
    return df_geocoder

# --- Stage 2: Geocoding with AI Fallback ---
# --- Stage 2: Geocoding with AI Fallback ---
def geocode_dataframe(df, config, ai_models):
    """
    Geocodes a prepared DataFrame using batch processing and a two-tiered AI fallback for failures.
    """
    print(f"\n--- Stage 2: Starting Geocoding Process ---")
    # Step 2.1: Batch Geocoding with Census API
    try:
        print("Sending data for batch geocoding...")
        url = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
        with io.StringIO() as buffer:
            # The FIX: The Census API requires a CSV with NO HEADER.
            df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            files = {'addressFile': ('addresses.csv', buffer, 'text/csv')}
            
            # The FIX: Re-added the 'vintage' parameter to be more explicit.
            payload = {'benchmark': 'Public_AR_Current', 'vintage': 'Current_Current'}
            
            response = requests.post(url, files=files, data=payload, timeout=300)
            response.raise_for_status()
        
        col_names = ['Unique ID', 'Input Address', 'Match Status', 'Match Type', 'Matched Address', 
                     'Coordinates', 'TIGER Line ID', 'Side', 'State FIPS', 'County FIPS', 'Tract', 'Block']
        geocoded_df = pd.read_csv(io.StringIO(response.text), header=None, names=col_names, dtype=str).fillna('')
        print("✅ Batch geocoding complete.")

    except requests.RequestException as e:
        print(f"❌ Error during batch geocoding: {e}")
        # Add more detail for 400 errors
        if e.response is not None and e.response.status_code == 400:
            print("This 400 error often means the input file format was incorrect (e.g., contained a header) or had invalid data.")
            print(f"Server response: {e.response.text}")
        return None

    # Step 2.2: AI Correction for Failures
    failed_rows = geocoded_df[geocoded_df['Match Status'].isin(['No_Match', 'Tie'])]
    print(f"Found {len(failed_rows)} addresses requiring AI correction.")
    
    if not failed_rows.empty:
        ai_corrected_count = 0
        for index, row in failed_rows.iterrows():
            print(f"  - AI Fallback for ID {row['Unique ID']}: '{row['Input Address']}'")
            # Tier 1: Gemini Flash
            ai_address = _correct_address_with_ai(ai_models['flash'], row['Input Address'])
            new_match = _geocode_single_address(ai_address) if ai_address else None
            
            if new_match:
                print("    ✅ Success with Gemini Flash.")
                _update_row_with_match(geocoded_df, index, new_match, "Match (AI Corrected - Flash)")
                ai_corrected_count += 1
                continue

            # Tier 2: Gemini Pro
            print("    - Flash failed. Retrying with Gemini Pro...")
            ai_address = _correct_address_with_ai(ai_models['pro'], row['Input Address'])
            new_match = _geocode_single_address(ai_address) if ai_address else None

            if new_match:
                print("    ✅ Success with Gemini Pro.")
                _update_row_with_match(geocoded_df, index, new_match, "Match (AI Corrected - Pro)")
                ai_corrected_count += 1
        
        print(f"AI correction complete. Successfully corrected {ai_corrected_count} addresses.")
    
    # Save the raw geocoded results
    geocoded_df.to_csv(config['geocoded_output'], index=False)
    print(f"✅ Geocoding process finished. Raw results saved to {config['geocoded_output']}")
    return geocoded_df

# Helper functions for geocoding
def _correct_address_with_ai(model, full_address_str):
    """Sends a single address to the AI for correction."""
    prompt = AI_PROMPT_TEMPLATE.format(full_address_str=full_address_str)
    try:
        response = model.generate_content(prompt)
        text_response = response.text
        # Clean the response to ensure it's valid JSON
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        return json.loads(text_response)
    except Exception:
        return None

def _geocode_single_address(address_info):
    """Geocodes a single, structured address dictionary."""
    if not address_info: return None
    url = "https://geocoding.geo.census.gov/geocoder/locations/address"
    try:
        params = {
            'street': address_info.get('street', ''),
            'city': address_info.get('city', ''),
            'state': address_info.get('state', ''),
            'zip': address_info.get('zip', ''),
            'benchmark': 'Public_AR_Current', 'format': 'json'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result['result']['addressMatches']:
            return result['result']['addressMatches'][0]
    except Exception:
        pass
    return None

def _update_row_with_match(df, index, match, status_text):
    """Updates a DataFrame row in-place with new geocoded data."""
    df.loc[index, 'Match Status'] = status_text
    df.loc[index, 'Match Type'] = match.get('matchType', 'Non_Exact')
    df.loc[index, 'Matched Address'] = match.get('matchedAddress')
    coords = match.get('coordinates', {})
    df.loc[index, 'Coordinates'] = f"{coords.get('x')},{coords.get('y')}"
    geographies = match.get('geographies', {})
    tract_info = geographies.get('Census Tracts', [{}])[0]
    for key in ['STATE', 'COUNTY', 'TRACT', 'BLOCK']:
        df.loc[index, f'{key.title()} FIPS' if key != 'TRACT' and key != 'BLOCK' else key.title()] = tract_info.get(key)

# --- Stage 3: Final Processing and Cleanup ---
def process_final_results(geocoded_df, config):
    """
    Cleans FIPS codes, fixes any remaining missing FIPS via coordinates,
    merges with original data, and saves the final file.
    """
    print("\n--- Stage 3: Finalizing Results ---")
    
    # 3.1: Clean and format FIPS codes
    print("Cleaning and padding FIPS codes...")
    fips_cols = ['State FIPS', 'County FIPS', 'Tract', 'Block']
    for col in fips_cols:
        geocoded_df[col] = geocoded_df[col].str.split('.').str[0].fillna('')
    
    geocoded_df['State FIPS'] = geocoded_df['State FIPS'].str.zfill(2)
    geocoded_df['County FIPS'] = geocoded_df['County FIPS'].str.zfill(3)
    geocoded_df['Tract'] = geocoded_df['Tract'].str.zfill(6)
    geocoded_df['Block'] = geocoded_df['Block'].str.zfill(4)

    # 3.2: Fix rows where AI found coordinates but FIPS are still missing
    print("Checking for missing FIPS in AI-corrected rows...")
    geocoded_df = _fix_fips_from_coords(geocoded_df)

    # 3.3 Create full FIPS codes
    geocoded_df['fips_11'] = geocoded_df['State FIPS'] + geocoded_df['County FIPS'] + geocoded_df['Tract']
    geocoded_df['fips_15'] = geocoded_df['fips_11'] + geocoded_df['Block']

    # 3.4 Merge geocoded data back with original data
    print("Merging geocoded data with original source data...")
    original_df = config['original_data_df']
    
    # Ensure 'Unique ID' is the same type for merging
    geocoded_df['Unique ID'] = pd.to_numeric(geocoded_df['Unique ID'], errors='coerce').astype('Int64')
    original_df['Unique ID'] = pd.to_numeric(original_df['Unique ID'], errors='coerce').astype('Int64')

    final_df = pd.merge(original_df, geocoded_df, on='Unique ID', how='inner')

    # 3.5 Filter for successful matches only
    matched_statuses = ['Match', 'Match (AI Corrected - Flash)', 'Match (AI Corrected - Pro)']
    final_df = final_df[final_df['Match Status'].isin(matched_statuses)].copy()
    print(f"Filtered down to {len(final_df)} successfully matched rows.")
    
    # 3.6 Final column cleanup and metadata addition
    final_df = final_df.drop(columns=['Unique ID', 'Match Type'])
    
    # --- THE FIX IS HERE ---
    # First, drop the 'dataset' column if it already exists from the source file.
    if 'dataset' in final_df.columns:
        final_df = final_df.drop(columns=['dataset'])
    # Now, safely insert the new 'dataset' column at the beginning.
    final_df.insert(0, 'dataset', config['dataset_name'])
    # --- END OF FIX ---
    
    # Check if 'dateingested' exists before trying to insert after it
    if 'dateingested' in final_df.columns:
        insert_loc = final_df.columns.get_loc('dateingested') + 1
        final_df.insert(insert_loc, 'dategeocoded', datetime.now().date())
    else:
        final_df['dategeocoded'] = datetime.now().date()
        
    # Save the final file for upload
    output_path = config['final_output']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    total_rows = len(geocoded_df)
    successful_matches = len(final_df)
    accuracy = (successful_matches / total_rows * 100) if total_rows > 0 else 0
    print(f"\n--- Final Accuracy: {accuracy:.2f}% ({successful_matches} of {total_rows} matched) ---")
    print(f"✅ Success! Final file ready for BigQuery at: {output_path}")

# Helper function for FIPS fixing
def _fix_fips_from_coords(df):
    """Uses FCC API to find FIPS for rows with valid coordinates but missing FIPS."""
    ai_statuses = ['Match (AI Corrected - Flash)', 'Match (AI Corrected - Pro)']
    fips_is_missing = ~df['State FIPS'].str.match(r'[1-9]', na=False) 
    rows_to_fix = df[df['Match Status'].isin(ai_statuses) & fips_is_missing]

    if rows_to_fix.empty:
        print("No AI-corrected rows with missing FIPS codes to fix.")
        return df

    print(f"Found {len(rows_to_fix)} rows to fix using coordinate-to-FIPS lookup...")
    for index, row in rows_to_fix.iterrows():
        try:
            lon_str, lat_str = row.get('Coordinates', '').split(',')
            fips_data = _get_fips_from_fcc(float(lat_str), float(lon_str))
            if fips_data:
                df.loc[index, 'State FIPS'] = fips_data['State FIPS']
                df.loc[index, 'County FIPS'] = fips_data['County FIPS']
                df.loc[index, 'Tract'] = fips_data['Tract']
                df.loc[index, 'Block'] = fips_data['Block']
                print(f"  - Fixed FIPS for row index {index}")
        except (ValueError, IndexError):
            continue
    return df

def _get_fips_from_fcc(latitude, longitude):
    """Gets FIPS data from lat/lon using the FCC Area API."""
    url = f"https://geo.fcc.gov/api/census/block/find?latitude={latitude}&longitude={longitude}&format=json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        fips_code = response.json().get('Block', {}).get('FIPS')
        if fips_code:
            return {
                'State FIPS': fips_code[0:2], 'County FIPS': fips_code[2:5],
                'Tract': fips_code[5:11], 'Block': fips_code[11:15]
            }
    except requests.RequestException:
        pass
    return None

# --- Main Execution ---
def main():
    """Main function to orchestrate the geocoding pipeline."""
    parser = argparse.ArgumentParser(description="Full-cycle Geocoding Pipeline with AI Correction.")
    parser.add_argument('source', choices=CONFIG.keys(), help='The name of the data source to process.')
    args = parser.parse_args()
    
    source_key = args.source
    print(f"Starting geocoding pipeline for source: '{source_key}'")
    
    config = CONFIG[source_key]

    # Step 0: Load environment and AI models
    ai_models = load_environment_and_models()

    # Step 1: Prepare the data for geocoding
    prepared_df = prepare_for_geocoding(config)
    if prepared_df is None:
        return

    # Step 2: Geocode the prepared data with AI fallback
    geocoded_df = geocode_dataframe(prepared_df, config, ai_models)
    if geocoded_df is None:
        return

    # Step 3: Process the results for final output
    process_final_results(geocoded_df, config)

if __name__ == '__main__':
    main()