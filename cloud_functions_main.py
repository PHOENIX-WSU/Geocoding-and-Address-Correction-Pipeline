import os
import json
import io
import pandas as pd
import requests
import google.generativeai as genai
from datetime import datetime

# Import Google Cloud client libraries
from google.cloud import storage, bigquery, secretmanager

# --- Initialize Google Cloud Clients ---
# These are initialized globally to be reused across function invocations
# for better performance.
storage_client = storage.Client()
bigquery_client = bigquery.Client()

def get_secret(secret_id, project_id, version_id="latest"):
    """Fetches a secret from Google Cloud Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# --- Configuration ---
# You can still use a config dictionary, but the file paths will be handled dynamically.
# BigQuery destination is added here.
CONFIG = {
    'demographics': {
        'column_map': {
            'street': 'ADDRESS', 'city': 'CITY', 'state': 'STATE', 'zip': 'ZIP'
        },
        'dataset_name': 'Patient Demographics (Geocoded)',
        'add_state_mi': False,
        'bq_project': 'your-gcp-project-id',  # <-- TODO: Update
        'bq_dataset': 'your_bigquery_dataset', # <-- TODO: Update
        'bq_table': 'your_geocoded_table'      # <-- TODO: Update
    },
    'van': {
        'column_map': {
            'street': 'street', 'city': 'city', 'state': 'state', 'zip': 'zip'
        },
        'dataset_name': 'MHU Field Data: Van deployment addresses (Geocoded)',
        'add_state_mi': True,
        'bq_project': 'your-gcp-project-id',  # <-- TODO: Update
        'bq_dataset': 'your_bigquery_dataset', # <-- TODO: Update
        'bq_table': 'your_geocoded_table'      # <-- TODO: Update
    }
}

# --- Cloud Function Entry Point ---
def geocode_gcs_file(event, context):
    """
    Triggered by a file upload to a GCS bucket. This is the main entry point.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    bucket_name = event['bucket']
    file_name = event['name']
    
    print(f"Processing file: {file_name} from bucket: {bucket_name}.")

    # Determine which configuration to use based on the file path in GCS
    # Example: if file is in 'demographics/input/some_file.csv', use 'demographics' config
    source_type = file_name.split('/')[0] 
    if source_type not in CONFIG:
        print(f"Error: Unknown source type '{source_type}' from file path. Halting.")
        return

    config = CONFIG[source_type]
    
    # --- Execute the Pipeline ---
    ai_models = load_environment_and_models()
    if not ai_models:
        return # Stop if AI setup fails

    prepared_df, original_df = prepare_for_geocoding(bucket_name, file_name, config)
    if prepared_df is None or prepared_df.empty:
        print("Preparation stage resulted in an empty dataframe. Halting.")
        return

    geocoded_df = geocode_dataframe(prepared_df, config, ai_models)
    if geocoded_df is None or geocoded_df.empty:
        print("Geocoding stage resulted in an empty dataframe. Halting.")
        return

    final_df = process_final_results(geocoded_df, original_df, config)

    # NEW: Load final results to BigQuery instead of saving to CSV
    if final_df is not None and not final_df.empty:
        load_to_bigquery(final_df, config)
    else:
        print("No data to load to BigQuery.")

# --- AI and API Configuration ---
def load_environment_and_models():
    """Loads environment variables and configures the AI models."""
    print("Loading environment variables and configuring AI...")
    try:
        # MODIFIED: Fetch API key securely from Secret Manager
        project_id = os.environ.get('GCP_PROJECT')
        api_key = get_secret("GOOGLE_AI_API_KEY", project_id) # Assumes secret is named "GOOGLE_AI_API_KEY"
        
        genai.configure(api_key=api_key)
        models = {
            'flash': genai.GenerativeModel('gemini-2.5-flash'), # Updated to modern model
            'pro': genai.GenerativeModel('gemini-2.5-pro')      # Updated to modern model
        }
        print("✅ Google AI configured successfully.")
        return models
    except Exception as e:
        print(f"Error configuring Google AI. Check secret & permissions. Details: {e}")
        return None

# --- Stage 1: Data Preparation ---
def prepare_for_geocoding(bucket_name, file_name, config):
    """
    MODIFIED: Reads a source CSV from GCS, cleans it, and formats it.
    """
    print(f"\n--- Stage 1: Preparing data from 'gs://{bucket_name}/{file_name}' ---")
    try:
        # MODIFIED: Read directly from GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        csv_data = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(csv_data), dtype=str).fillna('')
        original_data_df = df.copy()
    except Exception as e:
        print(f"❌ Error: Could not read file from GCS. Details: {e}")
        return None, None

    if config.get('add_state_mi', False):
        df['state'] = 'MI'

    col_map = config['column_map']
    df_geocoder = pd.DataFrame()
    df_geocoder['Street Address'] = df[col_map['street']]
    df_geocoder['City'] = df[col_map['city']]
    df_geocoder['State'] = df[col_map['state']]
    df_geocoder['ZIP'] = df[col_map['zip']]

    original_rows = len(df_geocoder)
    df_geocoder = df_geocoder[df_geocoder['Street Address'].str.strip() != ''].copy()
    print(f"Removed {original_rows - len(df_geocoder)} rows with an empty street address.")

    df_geocoder.insert(0, 'Unique ID', range(len(df_geocoder)))
    
    original_data_df = original_data_df.iloc[df_geocoder.index].copy()
    original_data_df['Unique ID'] = df_geocoder['Unique ID']

    print("✅ Preparation complete.")
    return df_geocoder, original_data_df

# --- Stage 2: Geocoding (No changes needed in logic) ---
def geocode_dataframe(df, config, ai_models):
    """
    Geocodes a DataFrame using batch processing and a batch AI fallback.
    This function's internal logic remains the same.
    """
    print("\n--- Stage 2: Starting Geocoding Process ---")
    geocoded_df = _perform_batch_geocoding(df)
    if geocoded_df is None:
        print("❌ Initial batch geocoding failed. Stopping pipeline.")
        return None
        
    # (The rest of your geocoding logic is copied here, unchanged)
    # ...
    # Step 2.2: AI Correction for Failures
    failed_rows = geocoded_df[geocoded_df['Match Status'].isin(['No_Match', 'Tie'])].copy()
    print(f"\nFound {len(failed_rows)} addresses requiring AI correction.")

    if not failed_rows.empty:
        print("Starting AI correction process...")
        corrected_data = []
        for i, (idx, row) in enumerate(failed_rows.iterrows()):
            input_address = row['Input Address']
            ai_address = _correct_address_with_ai(ai_models['flash'], input_address)
            model_used = 'Flash'
            if not ai_address or not ai_address.get('street'):
                ai_address = _correct_address_with_ai(ai_models['pro'], input_address)
                model_used = 'Pro'
            
            if ai_address and ai_address.get('street'):
                corrected_data.append({
                    'Unique ID': row['Unique ID'],
                    'Street Address': ai_address.get('street', ''), 'City': ai_address.get('city', ''),
                    'State': ai_address.get('state', ''), 'ZIP': ai_address.get('zip', ''),
                    'AI Model': model_used
                })
        
        if corrected_data:
            corrected_df = pd.DataFrame(corrected_data)
            regeocoding_input_df = corrected_df[['Unique ID', 'Street Address', 'City', 'State', 'ZIP']]
            print("\nSending AI-corrected addresses for a new round of batch geocoding...")
            regeocoded_results_df = _perform_batch_geocoding(regeocoding_input_df)

            if regeocoded_results_df is not None and not regeocoded_results_df.empty:
                print("Merging re-geocoded results...")
                regeocoded_results_df = regeocoded_results_df.merge(corrected_df[['Unique ID', 'AI Model']], on='Unique ID', how='left')
                successful_regeocodes = regeocoded_results_df[regeocoded_results_df['Match Status'] == 'Match'].copy()
                
                if not successful_regeocodes.empty:
                    print(f"Successfully re-geocoded {len(successful_regeocodes)} addresses after AI correction.")
                    geocoded_df.set_index('Unique ID', inplace=True)
                    successful_regeocodes.set_index('Unique ID', inplace=True)
                    successful_regeocodes['Match Status'] = 'Match (AI Corrected - ' + successful_regeocodes['AI Model'] + ')'
                    columns_to_update = ['Match Status', 'Match Type', 'Matched Address', 'Coordinates', 'TIGER Line ID', 'Side', 'State FIPS', 'County FIPS', 'Tract', 'Block']
                    geocoded_df.update(successful_regeocodes[columns_to_update])
                    geocoded_df.reset_index(inplace=True)
    
    print(f"\n✅ Geocoding process finished.")
    return geocoded_df

# --- Stage 3: Final Processing (Modified to accept original_df) ---
def process_final_results(geocoded_df, original_df, config):
    """
    MODIFIED: Merges with the original_df passed as an argument.
    """
    print("\n--- Stage 3: Finalizing Results ---")
    fips_cols = ['State FIPS', 'County FIPS', 'Tract', 'Block']
    for col in fips_cols:
        geocoded_df[col] = geocoded_df[col].str.split('.').str[0].fillna('')
    geocoded_df['State FIPS'] = geocoded_df['State FIPS'].str.zfill(2)
    geocoded_df['County FIPS'] = geocoded_df['County FIPS'].str.zfill(3)
    geocoded_df['Tract'] = geocoded_df['Tract'].str.zfill(6)
    geocoded_df['Block'] = geocoded_df['Block'].str.zfill(4)

    geocoded_df = _fix_fips_from_coords(geocoded_df)
    geocoded_df['fips_11'] = geocoded_df['State FIPS'] + geocoded_df['County FIPS'] + geocoded_df['Tract']
    geocoded_df['fips_15'] = geocoded_df['fips_11'] + geocoded_df['Block']

    print("Merging geocoded data with original source data...")
    geocoded_df['Unique ID'] = pd.to_numeric(geocoded_df['Unique ID'], errors='coerce').astype('Int64')
    original_df['Unique ID'] = pd.to_numeric(original_df['Unique ID'], errors='coerce').astype('Int64')

    final_df = pd.merge(original_df, geocoded_df, on='Unique ID', how='inner')
    matched_statuses = ['Match', 'Match (AI Corrected - Flash)', 'Match (AI Corrected - Pro)']
    final_df = final_df[final_df['Match Status'].isin(matched_statuses)].copy()
    
    final_df = final_df.drop(columns=['Unique ID', 'Match Type'])
    final_df.insert(0, 'dataset', config['dataset_name'])
    final_df['dategeocoded'] = datetime.now().date().isoformat()

    return final_df

# --- New Stage 4: Load to BigQuery ---
def load_to_bigquery(df, config):
    """Appends a pandas DataFrame to the specified BigQuery table."""
    
    destination_table = f"{config['bq_project']}.{config['bq_dataset']}.{config['bq_table']}"
    print(f"\n--- Stage 4: Loading {len(df)} rows to BigQuery table: {destination_table} ---")

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        # To append data to an existing table
        write_disposition="WRITE_APPEND", 
        # If the schema of the DataFrame doesn't match the table, BigQuery can try to update it.
        schema_update_options=[
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
        ],
        # Handles date columns correctly
        autodetect=True, 
    )

    try:
        job = bigquery_client.load_table_from_dataframe(
            df, destination_table, job_config=job_config
        )
        job.result()  # Wait for the job to complete
        print(f"✅ Successfully appended {len(df)} rows to {destination_table}.")
    except Exception as e:
        print(f"❌ Failed to load data to BigQuery. Errors: {e}")


# --- Helper Functions (Copied from your original script, no changes needed) ---
# Your AI_PROMPT_TEMPLATE, _perform_batch_geocoding, _correct_address_with_ai,
# _fix_fips_from_coords, and _get_fips_from_fcc functions go here, unchanged.

AI_PROMPT_TEMPLATE = """
Fix this address please: "{full_address_str}".
Format your answer as a single, valid JSON object with keys: "street", "city", "state", "zip".
If a value is not present, use an empty string "". Do not add any other text around the JSON object.
"""

def _perform_batch_geocoding(df):
    chunk_size = 6000
    all_results = []
    # ... (rest of the function is the same)
    num_chunks = (len(df) // chunk_size) + (1 if len(df) % chunk_size > 0 else 0)    
    if num_chunks == 0:
        return pd.DataFrame()
    for i, start_index in enumerate(range(0, len(df), chunk_size)):
        end_index = start_index + chunk_size
        chunk_df = df.iloc[start_index:end_index]        
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
        except requests.RequestException as e:
            print(f" ❌ Error on chunk {i + 1}: {e}")
            continue
    if not all_results:
        return None
    return pd.concat(all_results, ignore_index=True)

def _correct_address_with_ai(model, full_address_str):
    prompt = AI_PROMPT_TEMPLATE.format(full_address_str=full_address_str)
    try:
        response = model.generate_content(prompt)
        text_response = response.text
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        return json.loads(text_response)
    except Exception:
        return None

def _fix_fips_from_coords(df):
    ai_statuses = ['Match (AI Corrected - Flash)', 'Match (AI Corrected - Pro)']
    fips_is_missing = ~df['State FIPS'].str.match(r'\d', na=False) 
    rows_to_fix = df[df['Match Status'].isin(ai_statuses) & fips_is_missing]
    if rows_to_fix.empty:
        return df
    for index, row in rows_to_fix.iterrows():
        try:
            lon_str, lat_str = row.get('Coordinates', '').split(',')
            fips_data = _get_fips_from_fcc(float(lat_str), float(lon_str))
            if fips_data:
                df.loc[index, 'State FIPS'] = fips_data['State FIPS']
                df.loc[index, 'County FIPS'] = fips_data['County FIPS']
                df.loc[index, 'Tract'] = fips_data['Tract']
                df.loc[index, 'Block'] = fips_data['Block']
        except (ValueError, IndexError):
            continue
    return df

def _get_fips_from_fcc(latitude, longitude):
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