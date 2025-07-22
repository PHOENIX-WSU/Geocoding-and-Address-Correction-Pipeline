import requests
import pandas as pd
import io

def geocode_csv(input_filepath, output_filepath):
    """
    Geocodes a CSV file using the US Census Bureau's Batch Geocoding API.

    Args:
        input_filepath (str): The path to the input CSV file.
        output_filepath (str): The path where the geocoded CSV file will be saved.
    """
    print(f"Starting the geocoding process for {input_filepath}...")

    # The URL for the Census Bureau's batch geocoding service
    url = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"

    # Define the parameters for the API request
    # 'Public_AR_Current' and 'Current_Current' match the web interface options
    payload = {
        'benchmark': 'Public_AR_Current',
        'vintage': 'Current_Current'
    }

    try:
        # Open the input CSV file in binary mode for uploading
        with open(input_filepath, 'rb') as address_file:
            files = {'addressFile': (input_filepath, address_file, 'text/csv')}

            # Make the POST request to the API
            print("Sending data to the US Census API. This may take a moment...")
            response = requests.post(url, files=files, data=payload)

            # Raise an exception if the request was unsuccessful
            response.raise_for_status()

            # The API returns the geocoded data as the response content
            # We can use pandas to easily read this into a DataFrame
            # The response is a string, so we use io.StringIO to treat it like a file
            # The first line of the response can sometimes be problematic, so we check for it.
            decoded_content = response.content.decode('utf-8')
            if "Unsupported Browser" in decoded_content:
                 print("Error: The API is indicating an unsupported browser. This can sometimes be ignored.")
                 # You might need to handle or clean the output if this error persists.

            geocoded_data = pd.read_csv(io.StringIO(decoded_content), header=None)

            # Define the column names for the output file based on Census API documentation
            column_names = [
                'Unique ID', 'Input Address', 'Match Status', 'Match Type',
                'Matched Address', 'Coordinates', 'TIGER Line ID', 'Side',
                'State FIPS', 'County FIPS', 'Tract', 'Block'
            ]

            # The number of columns in the response can vary slightly.
            # We will assign names to the columns we have.
            geocoded_data.columns = column_names[:len(geocoded_data.columns)]


            # Save the geocoded data to a new CSV file
            geocoded_data.to_csv(output_filepath, index=False)

            print(f"Geocoding complete! Results saved to {output_filepath}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the API request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- How to use the script ---

    # 1. Replace 'your_input_file.csv' with the name of your CSV file.
    #    Make sure this file is in the same directory as the script,
    #    or provide the full path to the file.
    input_filename = 'Data/van-complete-data-for-geocoding.csv'

    # 2. Define the name for your output file.
    output_filename = 'geocoded_results.csv'

    # 3. Run the script!
    geocode_csv(input_filename, output_filename)