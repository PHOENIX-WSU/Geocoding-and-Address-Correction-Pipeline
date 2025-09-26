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
CONFIG = {
    'demographics': {
        'input_file': 'Production/Input/patientDemographics10thSep.csv',
        'geocoding_input': 'Production/Geocoding-Input/patientDemographics10thSep.csv',
        'geocoded_output': 'Production/Geocoded/patientDemographics10thSep_geocoded.csv',
        'final_output': 'Production/Geocoded/patientDemographics10thSep_geocodedupload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'ADDRESS',
            'city': 'CITY',
            'state': 'STATE',
            'zip': 'ZIP'
        },
        'dataset_name': 'Patient Demographics (Geocoded)',
        'add_state_mi': False
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
        'add_state_mi': True
    },
    'AFI': {
        'input_file': 'Production/Input/HistoricTestingAFI_to_geocode.csv',
        'geocoding_input': 'Production/Input/HistoricTestingAFI_to_geocode.csv',
        'geocoded_output': 'Production/Input/HistoricTestingAFI_to_geocode.csv',
        'final_output': 'Production/Input/HistoricTestingAFI_to_geocode_Geocoded.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'AddressLine1',
            'city': 'City',
            'state': 'state',
            'zip': 'Zip'
        },
        'dataset_name': 'HistoricTestingAFI (Geocoded)',
        'add_state_mi': True
    },
    'TestingFlat': {
        'input_file': 'Production/Input/HistoricTestingFlat(in)_cleaned.csv',
        'geocoding_input': 'Production/Geocoding-Input/HistoricTestingFlat(in).csv',
        'geocoded_output': 'Production/Geocoded/HistoricTestingFlat(in)_geocoded.csv',
        'final_output': 'Production/Geocoded/HistoricTestingFlat(in)_geocoded_upload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'ParsedStreetAddress',
            'city': 'ParsedCity',
            'state': 'ParsedState',
            'zip': 'ParsedZip'
        },
        'dataset_name': 'HistoricTestingAFI (Geocoded)',
        'add_state_mi': False
    },
    'DentalPlaces': {
        'input_file': 'Production/Input/dentalPlaces15thAug.csv',
        'geocoding_input': 'Production/Geocoding-Input/dentalPlaces15thAug.csv',
        'geocoded_output': 'Production/Geocoded/dentalPlaces15thAug_geocoded.csv',
        'final_output': 'Production/Geocoded/dentalPlaces15thAug_geocoded_upload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'street_address',
            'city': 'city',
            'state': 'state',
            'zip': 'zip'
        },
        'dataset_name': 'dentalPlaces (Geocoded)',
        'add_state_mi': False
    },
    'fqhc': {
        'input_file': 'Production/Input/fqhcCLINICS.csv',
        'geocoding_input': 'Production/Geocoding-Input/fqhcCLINICS.csv',
        'geocoded_output': 'Production/Geocoded/fqhcCLINICS_geocoded.csv',
        'final_output': 'Production/Geocoded/fqhcCLINICS_geocodedupload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'address',
            'city': 'city',
            'state': 'state',
            'zip': 'zip_code'
        },
        'dataset_name': 'dentalPlaces (Geocoded)',
        'add_state_mi': False
    },
    'rhc': {
        'input_file': 'Production/Input/rhcCLINICS.csv',
        'geocoding_input': 'Production/Geocoding-Input/rhcCLINICS.csv',
        'geocoded_output': 'Production/Geocoded/rhcCLINICS_geocoded.csv',
        'final_output': 'Production/Geocoded/rhcCLINICS_geocodedupload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'address',
            'city': 'city',
            'state': 'state',
            'zip': 'zip'
        },
        'dataset_name': 'rhcCLINICS (Geocoded)',
        'add_state_mi': False
    },
    'pharmacy': {
        'input_file': 'Production/Input/pharmacyPLACES.csv',
        'geocoding_input': 'Production/Geocoding-Input/pharmacyPLACES.csv',
        'geocoded_output': 'Production/Geocoded/pharmacyPLACES_geocoded.csv',
        'final_output': 'Production/Geocoded/pharmacyPLACES_geocodedupload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'address',
            'city': 'city',
            'state': 'state',
            'zip': 'zipcode'
        },
        'dataset_name': 'pharmacyPLACES (Geocoded)',
        'add_state_mi': False
    },
    'oralhealthPLACES': {
        'input_file': 'Production/Input/oralhealthPLACES.csv',
        'geocoding_input': 'Production/Geocoding-Input/oralhealthPLACES.csv',
        'geocoded_output': 'Production/Geocoded/oralhealthPLACES_geocoded.csv',
        'final_output': 'Production/Geocoded/oralhealthPLACES_geocodedupload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'street',
            'city': 'city',
            'state': 'state',
            'zip': 'zip'
        },
        'dataset_name': 'oralhealthPLACES (Geocoded)',
        'add_state_mi': False
    },
    'oralhealthCLINICS': {
        'input_file': 'Production/Input/oralhealthCLINICS.csv',
        'geocoding_input': 'Production/Geocoding-Input/oralhealthCLINICS.csv',
        'geocoded_output': 'Production/Geocoded/oralhealthCLINICS_geocoded.csv',
        'final_output': 'Production/Geocoded/oralhealthCLINICS_geocodedupload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'street',
            'city': 'city',
            'state': 'state',
            'zip': 'zip'
        },
        'dataset_name': 'oralhealthCLINICS (Geocoded)',
        'add_state_mi': False
    },
    'medicalPLACES': {
        'input_file': 'Production/Input/medicalPLACES.csv',
        'geocoding_input': 'Production/Geocoding-Input/medicalPLACES.csv',
        'geocoded_output': 'Production/Geocoded/medicalPLACES_geocoded.csv',
        'final_output': 'Production/Geocoded/medicalPLACES_geocodedupload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'street_address',
            'city': 'city',
            'state': 'state',
            'zip': 'zip'
        },
        'dataset_name': 'medicalPLACES (Geocoded)',
        'add_state_mi': False
    },
    'mmhcTotalCounts': {
        'input_file': 'Production/Input/mmhctotalcounts.csv',
        'geocoding_input': 'Production/Geocoding-Input/mmhctotalcounts.csv',
        'geocoded_output': 'Production/Geocoded/mmhctotalcounts_geocoded.csv',
        'final_output': 'Production/Geocoded/mmhctotalcounts_geocodedupload.csv',
        'original_data_df': None, # To be loaded later
        'column_map': {
            'street': 'Address',
            'city': 'City',
            'state': 'State',
            'zip': 'zipcode'
        },
        'dataset_name': 'MMHC Total Counts (Geocoded)',
        'add_state_mi': False
    },
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
            'flash': genai.GenerativeModel('gemini-2.5-flash'),
            'pro': genai.GenerativeModel('gemini-2.5-pro')
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
        df = pd.read_csv(config['input_file'], dtype=str).fillna('')
        config['original_data_df'] = df.copy()
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{config['input_file']}'.")
        sys.exit(1)

    if config.get('add_state_mi', False):
        df['state'] = 'MI'
        print("Hardcoded 'MI' as the state.")

    col_map = config['column_map']
    for val in col_map.values():
        if val not in df.columns:
            print(f"❌ Error: Required column '{val}' not found in the input file.")
            sys.exit(1)
            
    df_geocoder = pd.DataFrame()
    df_geocoder['Street Address'] = df[col_map['street']]
    df_geocoder['City'] = df[col_map['city']]
    df_geocoder['State'] = df[col_map['state']]
    df_geocoder['ZIP'] = df[col_map['zip']]

    original_rows = len(df_geocoder)
    df_geocoder = df_geocoder[df_geocoder['Street Address'].str.strip() != ''].copy()
    print(f"Removed {original_rows - len(df_geocoder)} rows with an empty street address.")

    df_geocoder.insert(0, 'Unique ID', range(len(df_geocoder)))
    
    config['original_data_df'] = config['original_data_df'].iloc[df_geocoder.index].copy()
    config['original_data_df']['Unique ID'] = df_geocoder['Unique ID']

    output_path = config['geocoding_input']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_geocoder.to_csv(output_path, index=False)
    print(f"✅ Prepared file saved to: {output_path}")
    return df_geocoder

# --- Stage 2: Geocoding with AI Fallback ---
def geocode_dataframe(df, config, ai_models):
    """
    Geocodes a DataFrame using batch processing and a batch AI fallback.
    """
    print("\n--- Stage 2: Starting Geocoding Process ---")

    # Step 2.1: Initial Batch Geocoding
    print("Performing initial batch geocoding...")
    geocoded_df = _perform_batch_geocoding(df)

    if geocoded_df is None:
        print("❌ Initial batch geocoding failed. Stopping pipeline.")
        return None
    print("✅ Initial batch geocoding complete.")

    # Step 2.2: AI Correction for Failures
    failed_rows = geocoded_df[geocoded_df['Match Status'].isin(['No_Match', 'Tie'])].copy()
    print(f"\nFound {len(failed_rows)} addresses requiring AI correction.")

    if not failed_rows.empty:
        # Part 1: Get AI corrections for all failed rows sequentially.
        print("Starting AI correction process... (This may take a while for large files)")
        corrected_data = []
        total_failed = len(failed_rows)
        for i, (idx, row) in enumerate(failed_rows.iterrows()):
            if (i + 1) % 100 == 0:
                print(f"  ...AI processing row {i + 1} of {total_failed}")

            input_address = row['Input Address']
            ai_address = _correct_address_with_ai(ai_models['flash'], input_address)
            model_used = 'Flash'

            if not ai_address or not ai_address.get('street'):
                ai_address = _correct_address_with_ai(ai_models['pro'], input_address)
                model_used = 'Pro'
            
            if ai_address and ai_address.get('street'):
                corrected_data.append({
                    'Unique ID': row['Unique ID'],
                    'Street Address': ai_address.get('street', ''),
                    'City': ai_address.get('city', ''),
                    'State': ai_address.get('state', ''),
                    'ZIP': ai_address.get('zip', ''),
                    'AI Model': model_used
                })
        
        print(f"✅ AI correction finished. {len(corrected_data)} addresses were parsed for re-geocoding.")

        # Part 2: Batch geocode the newly corrected addresses.
        if corrected_data:
            corrected_df = pd.DataFrame(corrected_data)
            regeocoding_input_df = corrected_df[['Unique ID', 'Street Address', 'City', 'State', 'ZIP']]
            
            print("\nSending AI-corrected addresses for a new round of batch geocoding...")
            regeocoded_results_df = _perform_batch_geocoding(regeocoding_input_df)

            if regeocoded_results_df is not None and not regeocoded_results_df.empty:
                # Part 3: Merge the successful new results back into the main DataFrame.
                print("Merging re-geocoded results...")
                regeocoded_results_df = regeocoded_results_df.merge(
                    corrected_df[['Unique ID', 'AI Model']], on='Unique ID', how='left'
                )
                
                successful_regeocodes = regeocoded_results_df[regeocoded_results_df['Match Status'] == 'Match'].copy()
                
                if not successful_regeocodes.empty:
                    print(f"Successfully re-geocoded {len(successful_regeocodes)} addresses after AI correction.")
                    
                    geocoded_df.set_index('Unique ID', inplace=True)
                    successful_regeocodes.set_index('Unique ID', inplace=True)
                    
                    successful_regeocodes['Match Status'] = 'Match (AI Corrected - ' + successful_regeocodes['AI Model'] + ')'
                    
                    columns_to_update = ['Match Status', 'Match Type', 'Matched Address', 'Coordinates', 'TIGER Line ID', 'Side', 'State FIPS', 'County FIPS', 'Tract', 'Block']
                    
                    geocoded_df.update(successful_regeocodes[columns_to_update])
                    geocoded_df.reset_index(inplace=True)
                else:
                    print("No AI-corrected addresses could be matched in the second batch.")
    
    geocoded_df.to_csv(config['geocoded_output'], index=False)
    print(f"\n✅ Geocoding process finished. Raw results saved to {config['geocoded_output']}")
    return geocoded_df

# --- Helper Functions ---
def _perform_batch_geocoding(df):
    """
    Takes a dataframe and performs geocoding in chunks. 
    Returns a combined dataframe of results or None if it fails.
    """
    chunk_size = 6000
    all_results = []
    num_chunks = (len(df) // chunk_size) + (1 if len(df) % chunk_size > 0 else 0)
    
    if num_chunks == 0:
        return pd.DataFrame()

    print(f"Data will be sent in {num_chunks} chunk(s) of up to {chunk_size} rows.")

    for i, start_index in enumerate(range(0, len(df), chunk_size)):
        end_index = start_index + chunk_size
        chunk_df = df.iloc[start_index:end_index]
        print(f"  - Sending chunk {i + 1} of {num_chunks}...")
        
        try:
            url = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
            with io.StringIO() as buffer:
                chunk_df.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                files = {'addressFile': ('addresses.csv', buffer, 'text/csv')}
                payload = {'benchmark': 'Public_AR_Current', 'vintage': 'Current_Current'}
                
                response = requests.post(url, files=files, data=payload, timeout=300)
                response.raise_for_status()
            
            col_names = ['Unique ID', 'Input Address', 'Match Status', 'Match Type', 'Matched Address', 'Coordinates', 'TIGER Line ID', 'Side', 'State FIPS', 'County FIPS', 'Tract', 'Block']
            
            geocoded_chunk_df = pd.read_csv(io.StringIO(response.text), header=None, names=col_names, dtype=str).fillna('')
            all_results.append(geocoded_chunk_df)
            print(f"    ✅ Chunk {i + 1} processed.")

        except requests.RequestException as e:
            print(f"    ❌ Error on chunk {i + 1}: {e}")
            if e.response is not None:
                print(f"    Server response: {e.response.text[:200]}...") # Print first 200 chars of error
            continue
    
    if not all_results:
        return None
        
    return pd.concat(all_results, ignore_index=True)

def _correct_address_with_ai(model, full_address_str):
    """Sends a single address to the AI for correction."""
    prompt = AI_PROMPT_TEMPLATE.format(full_address_str=full_address_str)
    try:
        response = model.generate_content(prompt)
        text_response = response.text
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        return json.loads(text_response)
    except Exception:
        return None

# --- Stage 3: Final Processing and Cleanup ---
def process_final_results(geocoded_df, config):
    """
    Cleans FIPS codes, fixes any remaining missing FIPS via coordinates,
    merges with original data, and saves the final file.
    """
    print("\n--- Stage 3: Finalizing Results ---")
    
    print("Cleaning and padding FIPS codes...")
    fips_cols = ['State FIPS', 'County FIPS', 'Tract', 'Block']
    for col in fips_cols:
        geocoded_df[col] = geocoded_df[col].str.split('.').str[0].fillna('')
    
    geocoded_df['State FIPS'] = geocoded_df['State FIPS'].str.zfill(2)
    geocoded_df['County FIPS'] = geocoded_df['County FIPS'].str.zfill(3)
    geocoded_df['Tract'] = geocoded_df['Tract'].str.zfill(6)
    geocoded_df['Block'] = geocoded_df['Block'].str.zfill(4)

    print("Checking for missing FIPS in AI-corrected rows...")
    geocoded_df = _fix_fips_from_coords(geocoded_df)

    geocoded_df['fips_11'] = geocoded_df['State FIPS'] + geocoded_df['County FIPS'] + geocoded_df['Tract']
    geocoded_df['fips_15'] = geocoded_df['fips_11'] + geocoded_df['Block']

    print("Merging geocoded data with original source data...")
    original_df = config['original_data_df']
    
    geocoded_df['Unique ID'] = pd.to_numeric(geocoded_df['Unique ID'], errors='coerce').astype('Int64')
    original_df['Unique ID'] = pd.to_numeric(original_df['Unique ID'], errors='coerce').astype('Int64')

    final_df = pd.merge(original_df, geocoded_df, on='Unique ID', how='inner')

    matched_statuses = ['Match', 'Match (AI Corrected - Flash)', 'Match (AI Corrected - Pro)']
    final_df = final_df[final_df['Match Status'].isin(matched_statuses)].copy()
    print(f"Filtered down to {len(final_df)} successfully matched rows.")
    
    final_df = final_df.drop(columns=['Unique ID', 'Match Type'])
    
    if 'dataset' in final_df.columns:
        final_df = final_df.drop(columns=['dataset'])
    final_df.insert(0, 'dataset', config['dataset_name'])
    
    if 'dateingested' in final_df.columns:
        insert_loc = final_df.columns.get_loc('dateingested') + 1
        final_df.insert(insert_loc, 'dategeocoded', datetime.now().date())
    else:
        final_df['dategeocoded'] = datetime.now().date()
        
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
    fips_is_missing = ~df['State FIPS'].str.match(r'\d', na=False) 
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
        if fips_code and len(fips_code) == 15:
            return {
                'State FIPS': fips_code[0:2], 'County FIPS': fips_code[2:5],
                'Tract': fips_code[5:11], 'Block': fips_code[11:15]
            }
    except requests.RequestException:
        pass
    return None

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Full-cycle Geocoding Pipeline with AI Correction.")
    parser.add_argument('source', choices=CONFIG.keys(), help='The name of the data source to process.')
    args = parser.parse_args()
    
    config = CONFIG[args.source]
    print(f"Starting geocoding pipeline for source: '{args.source}'")

    ai_models = load_environment_and_models()
    prepared_df = prepare_for_geocoding(config)
    if prepared_df is None or prepared_df.empty:
        print("Preparation stage resulted in an empty dataframe. Halting.")
        return

    geocoded_df = geocode_dataframe(prepared_df, config, ai_models)
    if geocoded_df is None or geocoded_df.empty:
        print("Geocoding stage resulted in an empty dataframe. Halting.")
        return

    process_final_results(geocoded_df, config)

if __name__ == '__main__':
    main()