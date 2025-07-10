import requests
import pandas as pd
import os
import sys
from dotenv import load_dotenv
import re
import string

# --- Load credentials from .env file ---
load_dotenv()

CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")

# Check if the keys exist and exit if they don't
if not CONSUMER_KEY:
    print("❌ Error: CONSUMER_KEY not found in .env file.")
    sys.exit()

if not CONSUMER_SECRET:
    print("❌ Error: CONSUMER_SECRET not found in .env file.")
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
        print("✅ Successfully received access token!")
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error getting token: {e.response.status_code} {e.response.text}")
    except Exception as e:
        print(f"❌ An error occurred getting token: {e}")
    return None


def clean_single_address(access_token, address_info):
    """
    Validates a single address.
    Raises requests.exceptions.HTTPError for API-level errors (e.g., 400, 401, 500).
    """
    if not access_token:
        raise ValueError("Access token is missing.")

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
        # Raise an exception for any HTTP error status codes (4xx or 5xx)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError:
        # Re-raise the exception to be handled by the caller
        raise
    except Exception as e:
        # For other errors like network issues, wrap them in a generic exception
        raise Exception(f"An unknown error occurred during API call: {e}") from e

# --- Main part of the script ---
if __name__ == "__main__":
    
    input_csv_path = "Data\patientDemographicstest.csv"
    output_csv_path = "Data\patientDemographicstestOutputAdvanced.csv"

    if not os.path.exists(input_csv_path):
        print(f"❌ Error: Input file not found at '{input_csv_path}'")
    else:
        # 1. Get the access token
        token = get_access_token(CONSUMER_KEY, CONSUMER_SECRET)

        if token:
            # 2. Read and prepare the CSV
            df = pd.read_csv(input_csv_path)
            df = df.dropna(subset=["demo_address"])
            print(f"\nRead {len(df)} rows from '{input_csv_path}'. Starting cleaning process...")

            # 3. Create new columns for the output
            new_cols = ['cleaned_street_address', 'cleaned_secondary_address', 'cleaned_city', 'cleaned_state', 'cleaned_zip', 'cleaned_zip_plus_4', 'api_status']
            for col in new_cols:
                df[col] = ''

            # 4. Loop through each row
            for index, row in df.iterrows():
                # Get original address data from the row
                address_to_clean = {
                    "street_1": row.get("demo_address", ""),
                    "street_2": row.get("demo_address2", ""),
                    "city": row.get("demo_city", ""),
                    "state": row.get("demo_state", ""),
                    "zip_code": str(row.get("demo_zip", ""))[:5] # Ensure 5-digit zip
                }
                
                cleaned_data = None
                
                try:
                    # First attempt to validate the address
                    print(f"Processing row {index + 2}...")
                    cleaned_data = clean_single_address(token, address_to_clean)
                    df.loc[index, 'api_status'] = 'Success'
                    print(f"   - ✅ Success on first attempt.")

                except requests.exceptions.HTTPError as e:
                    # --- MODIFICATION START: Handle API errors, specifically 400 ---
                    if e.response and e.response.status_code == 400:
                        print(f"   - ⚠️ Address failed with 400. Attempting to clean and retry...")
                        
                        # Create a copy of the address to modify
                        retry_address = address_to_clean.copy()
                        original_address_for_comparison = address_to_clean.copy()

                        # --- CLEANING LOGIC ---
                        
                        # 1. Sanitize characters: Remove quotes and other non-standard characters
                        # Allowed characters: letters, numbers, space, period, hyphen, pound, slash
                        allowed_chars = set(string.ascii_letters + string.digits + ' .-#/')
                        def sanitize_string(s):
                            return "".join(filter(lambda char: char in allowed_chars, s)).strip()

                        retry_address['street_1'] = sanitize_string(retry_address['street_1'])
                        retry_address['street_2'] = sanitize_string(retry_address['street_2'])
                        retry_address['city'] = sanitize_string(retry_address['city'])

                        # 2. Fix "PO" to "PO BOX" if needed
                        if ' PO ' in retry_address['street_1'].upper() and 'PO BOX' not in retry_address['street_1'].upper():
                            # Use regex for a safe, case-insensitive replacement of whole word "PO"
                            retry_address['street_1'] = re.sub(r'\bPO\b', 'PO BOX', retry_address['street_1'], flags=re.IGNORECASE)

                        # 3. Check for a ZIP code in the secondary address line
                        if retry_address['street_2']:
                            zip_pattern = re.compile(r'\b(\d{5}(?:-\d{4})?)\b')
                            match = zip_pattern.search(retry_address['street_2'])
                            if match:
                                found_zip = match.group(1)
                                # Only move the ZIP if the main zip field is empty
                                if not retry_address.get('zip_code'):
                                    retry_address['zip_code'] = found_zip[:5] # Use first 5 digits
                                # Remove the found zip from the secondary address line
                                retry_address['street_2'] = retry_address['street_2'].replace(found_zip, '').strip()

                        # --- RETRY LOGIC ---
                        # Only retry if the cleaning logic actually changed the address
                        if retry_address != original_address_for_comparison:
                            print(f"   - Retrying with cleaned data...")
                            try:
                                cleaned_data = clean_single_address(token, retry_address)
                                df.loc[index, 'api_status'] = 'Success after cleaning'
                                print(f"   - ✅ Success on second attempt!")
                            except Exception as e2:
                                df.loc[index, 'api_status'] = 'Failed after cleaning'
                                print(f"   - ❌ Retry failed: {e2}")
                        else:
                            df.loc[index, 'api_status'] = 'Failed (no change)'
                            print("   - ❌ No automatic cleaning possible, skipping.")
                    else:
                        # Handle other non-400 HTTP errors (e.g., 401 Unauthorized, 500 Server Error)
                        df.loc[index, 'api_status'] = f'Failed ({e.response.status_code})'
                        print(f"   - ❌ API Error: {e.response.status_code} {e.response.reason}")
                except Exception as e:
                    df.loc[index, 'api_status'] = 'Failed (Unknown Error)'
                    print(f"   - ❌ An unexpected error occurred: {e}")
                # --- MODIFICATION END ---
                
                # 5. Populate the new columns if validation was successful
                if cleaned_data and 'address' in cleaned_data:
                    address_part = cleaned_data.get("address", {})
                    df.loc[index, 'cleaned_street_address'] = address_part.get('streetAddress')
                    df.loc[index, 'cleaned_secondary_address'] = address_part.get('secondaryAddress')
                    df.loc[index, 'cleaned_city'] = address_part.get('city')
                    df.loc[index, 'cleaned_state'] = address_part.get('state')
                    df.loc[index, 'cleaned_zip'] = address_part.get('ZIPCode')
                    df.loc[index, 'cleaned_zip_plus_4'] = address_part.get('ZIPPlus4')

            # 6. Save the results to a new CSV file
            df.to_csv(output_csv_path, index=False)
            print(f"\n✅ All done! Cleaned data saved to '{output_csv_path}'.")