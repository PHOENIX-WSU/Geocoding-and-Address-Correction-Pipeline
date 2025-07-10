import requests
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from thefuzz import fuzz

# --- Load credentials from .env file ---
load_dotenv()

CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")

# Check if the keys exist and exit if they don't
if not CONSUMER_KEY:
    print("Error: CONSUMER_KEY not found in .env file.")
    sys.exit()

if not CONSUMER_SECRET:
    print("Error: CONSUMER_SECRET not found in .env file.")
    sys.exit()

# API endpoints
TOKEN_URL = "https://apis.usps.com/oauth2/v3/token"
ADDRESS_API_URL = "https://apis.usps.com/addresses/v3/address"


def get_access_token(client_id, client_secret):
    """Gets an OAuth 2.0 access token from the USPS API."""
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
    """
    Validates a single address.
    Checks for both HTTP errors and validation errors inside the API response.
    """
    if not access_token:
        return None, "Skipped - No Access Token"

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    params = {
        "streetAddress": address_info.get("street_1"),
        "secondaryAddress": address_info.get("street_2"),
        "city": address_info.get("city"),
        "state": address_info.get("state"),
        "ZIPCode": address_info.get("zip_code"),
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


# --- Main part of the script ---
if __name__ == "__main__":
    
    input_csv_path = "Data\\patientDemographicstest.csv"
    output_csv_path = "Data\\patientDemographicstestOutputAdvanced.csv"

    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at '{input_csv_path}'")
    else:
        token = get_access_token(CONSUMER_KEY, CONSUMER_SECRET)

        if token:
            df = pd.read_csv(input_csv_path, dtype={"demo_zip": str})
            df = df.dropna(subset=["demo_address"])
            print(f"\nRead {len(df)} rows from '{input_csv_path}'. Starting cleaning process...")

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

                if status == 'Success' and cleaned_data and 'address' in cleaned_data:
                    address_part = cleaned_data.get("address", {})
                    df.loc[index, 'cleaned_street_address'] = address_part.get('streetAddress')
                    df.loc[index, 'cleaned_secondary_address'] = address_part.get('secondaryAddress')
                    df.loc[index, 'cleaned_city'] = address_part.get('city')
                    df.loc[index, 'cleaned_state'] = address_part.get('state')
                    df.loc[index, 'cleaned_zip'] = address_part.get('ZIPCode')
                    df.loc[index, 'cleaned_zip_plus_4'] = address_part.get('ZIPPlus4')
                    print(f"   - Cleaned address for row {index + 1}")

            failed_statuses = ['API HTTP Error: 400', 'API Validation Error']

            print("\nCorrection Step 1: Checking for 'PO' errors to retry...")
            retry_count_po = 0
            rows_to_retry_indices = df[df['api_status'].isin(failed_statuses)].index
            for index in rows_to_retry_indices:
                row = df.loc[index]
                original_address = str(row.get('demo_address', ''))
                parts = original_address.split()
                if len(parts) > 1 and parts[0].upper() == 'PO' and parts[1].isdigit():
                    new_address = 'PO Box ' + ' '.join(parts[1:])
                    print(f"   - Retrying row {index + 1} with new address: '{new_address}'")
                    address_to_clean = {"street_1": new_address, "street_2": row.get("demo_address2"), "city": row.get("demo_city"), "state": row.get("demo_state"), "zip_code": str(row.get("demo_zip", ""))[:5]}
                    cleaned_data, status = clean_single_address(token, address_to_clean)
                    if status == 'Success':
                        retry_count_po += 1
                        df.loc[index, 'api_status'] = 'Success (Corrected PO Box)'
                        if cleaned_data and 'address' in cleaned_data:
                            address_part = cleaned_data.get("address", {})
                            df.loc[index, 'cleaned_street_address'] = address_part.get('streetAddress')
                            df.loc[index, 'cleaned_secondary_address'] = address_part.get('secondaryAddress')
                            df.loc[index, 'cleaned_city'] = address_part.get('city')
                            df.loc[index, 'cleaned_state'] = address_part.get('state')
                            df.loc[index, 'cleaned_zip'] = address_part.get('ZIPCode')
                            df.loc[index, 'cleaned_zip_plus_4'] = address_part.get('ZIPPlus4')
            print(f"   - Corrected {retry_count_po} 'PO Box' addresses.")

            print("\nCorrection Step 2: Checking for misplaced ZIP Codes to retry...")
            retry_count_zip = 0
            rows_to_check_zip = df[df['api_status'].isin(failed_statuses)].index
            for index in rows_to_check_zip:
                row = df.loc[index]
                
                # --- THIS IS THE FIX ---
                # A more robust check for empty/missing ZIP values like '' or NaN
                zip_is_missing = pd.isna(row.get('demo_zip')) or str(row.get('demo_zip', '')).strip() == ''
                
                if zip_is_missing:
                    addr2_val = str(row.get('demo_address2', '')).strip()
                    city_val = str(row.get('demo_city', '')).strip()
                    state_val = str(row.get('demo_state', '')).strip()
                    misplaced_zip = None
                    source_col_name = None
                    if addr2_val.isdigit() and len(addr2_val) == 5:
                        misplaced_zip = addr2_val
                        source_col_name = 'demo_address2'
                    elif city_val.isdigit() and len(city_val) == 5:
                        misplaced_zip = city_val
                        source_col_name = 'demo_city'
                    elif state_val.isdigit() and len(state_val) == 5:
                        misplaced_zip = state_val
                        source_col_name = 'demo_state'
                    if misplaced_zip:
                        print(f"   - Found misplaced ZIP in '{source_col_name}' for row {index + 1}. Moving '{misplaced_zip}' and retrying.")
                        address_to_clean = {"street_1": row.get("demo_address"), "street_2": "" if source_col_name == 'demo_address2' else row.get("demo_address2"), "city": "" if source_col_name == 'demo_city' else row.get("demo_city"), "state": "" if source_col_name == 'demo_state' else row.get("demo_state"), "zip_code": misplaced_zip}
                        cleaned_data, status = clean_single_address(token, address_to_clean)
                        if status == 'Success':
                            retry_count_zip += 1
                            df.loc[index, 'demo_zip'] = misplaced_zip
                            df.loc[index, source_col_name] = ''
                            df.loc[index, 'api_status'] = 'Success (Corrected ZIP Location)'
                            if cleaned_data and 'address' in cleaned_data:
                                address_part = cleaned_data.get("address", {})
                                df.loc[index, 'cleaned_street_address'] = address_part.get('streetAddress')
                                df.loc[index, 'cleaned_secondary_address'] = address_part.get('secondaryAddress')
                                df.loc[index, 'cleaned_city'] = address_part.get('city')
                                df.loc[index, 'cleaned_state'] = address_part.get('state')
                                df.loc[index, 'cleaned_zip'] = address_part.get('ZIPCode')
                                df.loc[index, 'cleaned_zip_plus_4'] = address_part.get('ZIPPlus4')
            print(f"   - Corrected {retry_count_zip} misplaced ZIP addresses.")

            print("\nCorrection Step 3: Matching remaining failed rows by lastname (fallback)...")
            failed_df = df[df['api_status'].isin(failed_statuses)].copy()
            success_df = df[df['api_status'].str.startswith('Success', na=False)].copy()
            match_count = 0
            for failed_index, failed_row in failed_df.iterrows():
                potential_matches = success_df[
                    (success_df['demo_lastname'] == failed_row['demo_lastname']) &
                    (success_df['demo_zip'] == failed_row['demo_zip']) &
                    (success_df['demo_state'] == failed_row['demo_state']) &
                    (success_df['demo_city'] == failed_row['demo_city'])
                ]
                for _, match_row in potential_matches.iterrows():
                    addr1 = str(failed_row.get('demo_address', ''))
                    addr2 = str(match_row.get('demo_address', ''))
                    similarity_ratio = fuzz.ratio(addr1, addr2)
                    if similarity_ratio > 50:
                        for col in new_cols:
                            if col.startswith('cleaned_'):
                                df.loc[failed_index, col] = match_row[col]
                        df.loc[failed_index, 'api_status'] = 'Matched on Lastname'
                        match_count += 1
                        print(f"   - Found a match for row {failed_index + 1}")
                        break
            print(f"   - Corrected {match_count} rows through lastname matching.")

            df.to_csv(output_csv_path, index=False)
            print(f"\nAll done! Cleaned data saved to '{output_csv_path}'.")