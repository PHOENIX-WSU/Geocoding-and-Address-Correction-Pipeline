import requests
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")

if not CONSUMER_KEY:
    print("Please enter the CONSUMER_KEY in .env file")
    sys.exit()

if not CONSUMER_SECRET:
    print("Please enter the CONSUMER_SECRET in .env file")
    sys.exit()

TOKEN_URL = "https://apis.usps.com/oauth2/v3/token"
ADDRESS_API_URL = "https://apis.usps.com/addresses/v3/address"


def get_access_token(client_id, client_secret):
    """
    Gets an OAuth 2.0 access token from the USPS API.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    
    print("Requesting access token...")
    try:
        response = requests.post(TOKEN_URL, headers=headers, json=data)
        # Raise an exception if the request was not successful
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
    Validates a single address using the USPS Address v3 API.
    """
    if not access_token:
        return "Cannot validate address without an access token."

    # Set up the headers with the bearer token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    # Map our dictionary to the API's expected query parameters
    params = {
        "streetAddress": address_info.get("street_1"),
        "city": address_info.get("city"),
        "state": address_info.get("state"),
        "ZIPCode": address_info.get("zip_code"),
    }
    
    try:
        response = requests.get(ADDRESS_API_URL, headers=headers, params=params)
        response.raise_for_status()
        
        cleaned_data = response.json()
        
        # The main address data is inside the 'address' key
        address_part = cleaned_data.get("address", {})
        
        # Format the address for display
        formatted_address = (
            f"{address_part.get('streetAddress', '')}\n"
            f"{address_part.get('city', '')}, {address_part.get('state', '')} "
            f"{address_part.get('ZIPCode', '')}-{address_part.get('ZIPPlus4', '')}"
        )
        return formatted_address

    except requests.exceptions.HTTPError as e:
        # The API might return a 404 for a bad address, which is useful info
        if e.response.status_code == 404:
            return f"Address not found or invalid. API Response: {e.response.json().get('error')}"
        return f"HTTP Error cleaning address: {e.response.status_code} {e.response.text}"
    except Exception as e:
        return f"An error occurred cleaning address: {e}"


# --- Main part of the script ---
if __name__ == "__main__":
    # 1. Get the access token
    token = get_access_token(CONSUMER_KEY, CONSUMER_SECRET)

    # 2. Proceed only if we have a token
    if token:
        # Your list of messy Michigan addresses
        messy_addresses = [
            {
                "street_1": "1174 SHILOH CHURCH RD",  # Misspelled Street
                "city": "ROCK HILL",
                "state": "SC",
                "zip": "29734"
            },
            {
                "street_1": "6135 Woodwar",  # Misspelled and incomplete Street
                "city": "Detroit",
                "state": "MI",
                "zip": "48202"                
            }
        ]

        print("\n--- Starting Address Cleaning Process ---\n")

        # Loop through each messy address and clean it
        for i, addr in enumerate(messy_addresses):
            print(f"Original Address #{i+1}: {addr}")
            
            # Call our cleaning function
            cleaned = clean_single_address(token, addr)
            
            print(f"Cleaned Address #{i+1}:\n{cleaned}")
            print("-" * 20)