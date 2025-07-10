import requests
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from thefuzz import fuzz # <-- ADDED: For fuzzy string matching

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
    
    MODIFIED: Returns a tuple (json_response, status_message).
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
        return response.json(), "Success" # Return JSON and a success message
    except requests.exceptions.HTTPError as e:
        # Return None and a detailed error message
        status_code = e.response.status_code
        print(f"   - API Error for address '{params.get('streetAddress')}': {status_code}")
        return None, f"API Error: {status_code}"
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
        # 1. Get the access token
        token = get_access_token(CONSUMER_KEY, CONSUMER_SECRET)

        if token:
            # 2. Read and prepare the CSV file
            df = pd.read_csv(input_csv_path, dtype={"demo_zip": str}) # Read zip as string
            df = df.dropna(subset=["demo_address"])
            print(f"\nRead {len(df)} rows from '{input_csv_path}'. Starting cleaning process...")

            # 3. Create new columns
            new_cols = ['cleaned_street_address', 'cleaned_secondary_address', 'cleaned_city', 'cleaned_state', 'cleaned_zip', 'cleaned_zip_plus_4', 'api_status']
            for col in new_cols:
                df[col] = None

            # 4. Loop through each row to call the API
            for index, row in df.iterrows():
                
                raw_zip = row.get("demo_zip", "")
                zip_for_api = raw_zip[:5] if isinstance(raw_zip, str) else ""
                
                address_to_clean = {
                    "street_1": row.get("demo_address"),
                    "street_2": row.get("demo_address2"),
                    "city": row.get("demo_city"),
                    "state": row.get("demo_state"),
                    "zip_code": zip_for_api
                }
                
                # MODIFIED: Unpack the data and the status message
                cleaned_data, status = clean_single_address(token, address_to_clean)

                df.loc[index, 'api_status'] = status # Store the detailed status

                # 5. Populate the new columns on success
                if status == 'Success' and cleaned_data and 'address' in cleaned_data:
                    address_part = cleaned_data.get("address", {})
                    df.loc[index, 'cleaned_street_address'] = address_part.get('streetAddress')
                    df.loc[index, 'cleaned_secondary_address'] = address_part.get('secondaryAddress')
                    df.loc[index, 'cleaned_city'] = address_part.get('city')
                    df.loc[index, 'cleaned_state'] = address_part.get('state')
                    df.loc[index, 'cleaned_zip'] = address_part.get('ZIPCode')
                    df.loc[index, 'cleaned_zip_plus_4'] = address_part.get('ZIPPlus4')
                    print(f"   - Cleaned address for row {index + 1}")

            # --- NEW SECTION: POST-PROCESSING TO MATCH FAILED ROWS ---
            print("\nStarting post-processing to match failed rows...")
            
            # Isolate rows that failed with a 400 error and those that succeeded
            failed_df = df[df['api_status'] == 'API Error: 400'].copy()
            success_df = df[df['api_status'] == 'Success'].copy()
            
            match_count = 0

            # Loop through each failed row
            for failed_index, failed_row in failed_df.iterrows():
                
                # Find successful rows with the same last name, zip, state, and city
                potential_matches = success_df[
                    (success_df['demo_lastname'] == failed_row['demo_lastname']) &
                    (success_df['demo_zip'] == failed_row['demo_zip']) &
                    (success_df['demo_state'] == failed_row['demo_state']) &
                    (success_df['demo_city'] == failed_row['demo_city'])
                ]

                # Now check for address similarity
                for _, match_row in potential_matches.iterrows():
                    
                    # Ensure addresses are strings before comparing
                    addr1 = str(failed_row.get('demo_address', ''))
                    addr2 = str(match_row.get('demo_address', ''))
                    
                    # Calculate similarity ratio (0-100)
                    similarity_ratio = fuzz.ratio(addr1, addr2)
                    
                    if similarity_ratio > 50:
                        # If a good match is found, copy the cleaned data
                        for col in new_cols:
                            if col.startswith('cleaned_'):
                                df.loc[failed_index, col] = match_row[col]
                        
                        # Update the status
                        df.loc[failed_index, 'api_status'] = 'Matched on Lastname'
                        match_count += 1
                        print(f"   - Found a match for row {failed_index + 1}")
                        
                        # Stop searching for this failed row and move to the next
                        break 
            
            print(f"Found and corrected {match_count} failed rows through matching.")
            # --- END OF NEW SECTION ---

            # 6. Save the final results
            df.to_csv(output_csv_path, index=False)
            print(f"\nAll done! Cleaned data saved to '{output_csv_path}'.")