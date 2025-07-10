import requests
import pandas as pd
import os
import sys
from dotenv import load_dotenv

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
    """Validates a single address and returns the parsed JSON response."""
    if not access_token:
        return None

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
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"   - API Error for address '{params.get('streetAddress')}': {e.response.status_code}")
        return None
    except Exception as e:
        print(f"   - An unknown error occurred cleaning address: {e}")
        return None


# --- Main part of the script ---
if __name__ == "__main__":
    
    input_csv_path = "Data\patientDemographicstest.csv"
    output_csv_path = "Data\patientDemographicstestOutputAdvanced.csv"

    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at '{input_csv_path}'")
    else:
        # 1. Get the access token using credentials from .env
        token = get_access_token(CONSUMER_KEY, CONSUMER_SECRET)

        if token:
            # 2. Read the CSV file
            df = pd.read_csv(input_csv_path)
            df = df.dropna(subset=["demo_address"])
            print(f"\nRead {len(df)} rows from '{input_csv_path}'. Starting cleaning process...")

            # 3. Create new columns
            new_cols = ['cleaned_street_address', 'cleaned_secondary_address', 'cleaned_city', 'cleaned_state', 'cleaned_zip', 'cleaned_zip_plus_4', 'api_status']
            for col in new_cols:
                df[col] = None

            # 4. Loop through each row
            for index, row in df.iterrows():
                
                # --- MODIFICATION START ---
                # Get the zip code from the row and convert it to a string.
                # This prevents errors if pandas reads it as a number (e.g., float).
                raw_zip = str(row.get("demo_zip", ""))
                
                # The API requires a 5-digit ZIP. We take the first 5 characters
                # to handle formats like '123456789' or '12345-6789'.
                zip_for_api = raw_zip[:5]
                # --- MODIFICATION END ---
                
                address_to_clean = {
                    "street_1": row.get("demo_address"),
                    "street_2": row.get("demo_address2"),
                    "city": row.get("demo_city"),
                    "state": row.get("demo_state"),
                    "zip_code": zip_for_api # Use the cleaned 5-digit zip
                }
                
                cleaned_data = clean_single_address(token, address_to_clean)

                # 5. Populate the new columns
                if cleaned_data and 'address' in cleaned_data:
                    address_part = cleaned_data.get("address", {})
                    df.loc[index, 'cleaned_street_address'] = address_part.get('streetAddress')
                    df.loc[index, 'cleaned_secondary_address'] = address_part.get('secondaryAddress')
                    df.loc[index, 'cleaned_city'] = address_part.get('city')
                    df.loc[index, 'cleaned_state'] = address_part.get('state')
                    df.loc[index, 'cleaned_zip'] = address_part.get('ZIPCode')
                    df.loc[index, 'cleaned_zip_plus_4'] = address_part.get('ZIPPlus4')
                    df.loc[index, 'api_status'] = 'Success'
                    print(f"   - Cleaned address for row {index + 1}")
                else:
                    df.loc[index, 'api_status'] = 'Failed'

            # 6. Save the results
            df.to_csv(output_csv_path, index=False)
            print(f"\nAll done! Cleaned data saved to '{output_csv_path}'.")