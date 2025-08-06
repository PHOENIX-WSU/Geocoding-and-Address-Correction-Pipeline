import pandas as pd
import requests
import os

def get_fips_from_coords_fcc(latitude, longitude):
    """
    Gets FIPS data from a latitude/longitude pair using the FCC Area API.
    Returns a dictionary with FIPS components if successful, otherwise None.
    """
    if pd.isna(latitude) or pd.isna(longitude):
        return None
    
    # This is the FCC's block lookup API endpoint
    url = "https://geo.fcc.gov/api/census/block/find"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        result = response.json()
        
        # The FCC API returns the full 15-digit FIPS block code directly
        if result['Block']['FIPS']:
            full_fips = result['Block']['FIPS']
            return {
                'State FIPS': full_fips[0:2],
                'County FIPS': full_fips[2:5],
                'Tract': full_fips[5:11],
                'Block': full_fips[11:15]
            }
    except Exception as e:
        print(f"      - Coordinate-to-FIPS lookup failed for ({latitude}, {longitude}): {e}")
    return None

def fix_missing_fips(input_filepath, output_filepath):
    """
    Reads a geocoded file, finds rows with missing FIPS data, 
    and attempts to fix them using existing coordinates via the FCC API.
    """
    print(f"Reading data from: {input_filepath}")
    try:
        # Read the file, ensuring all columns are treated as strings to be safe
        df = pd.read_csv(input_filepath, dtype=str).fillna('')
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
        return

    ai_statuses = ['Match (AI Corrected - Flash)', 'Match (AI Corrected - Pro)']
    is_ai_corrected = df['Match Status'].isin(ai_statuses)
    # A blank or '00' state FIPS indicates a problem
    fips_is_missing = ~df['State FIPS'].str.match(r'[1-9]', na=False) 
    
    rows_to_fix = df[is_ai_corrected & fips_is_missing]

    if rows_to_fix.empty:
        print("No AI-corrected rows with missing FIPS codes found. Exiting.")
        return

    print(f"Found {len(rows_to_fix)} rows with missing FIPS codes. Starting correction process...")

    for index, row in rows_to_fix.iterrows():
        coords_str = row.get('Coordinates', '')
        try:
            lon_str, lat_str = coords_str.split(',')
            latitude, longitude = float(lat_str), float(lon_str)
            
            print(f"  - Fixing row index {index} using coordinates...")
            
            # Make the targeted API call to get FIPS from the coordinates
            fips_data = get_fips_from_coords_fcc(latitude, longitude)
            
            if fips_data:
                # If data is returned, update the DataFrame in place
                df.loc[index, 'State FIPS'] = fips_data['State FIPS']
                df.loc[index, 'County FIPS'] = fips_data['County FIPS']
                df.loc[index, 'Tract'] = fips_data['Tract']
                df.loc[index, 'Block'] = fips_data['Block']
                
                fips_11 = f"{fips_data['State FIPS']}{fips_data['County FIPS']}{fips_data['Tract']}"
                df.loc[index, 'fips_11'] = fips_11
                df.loc[index, 'fips_15'] = f"{fips_11}{fips_data['Block']}"
                print(f"    - Success! Found FIPS code: {fips_11}")
            else:
                print(f"    - Failed to find FIPS for coordinates: {coords_str}")

        except (ValueError, IndexError):
            print(f"  - Could not parse coordinates for row index {index}. Skipping.")
            continue
            
    # Save the final, fully corrected file
    df.to_csv(output_filepath, index=False)
    print(f"\nProcess complete! ✨\nUpdated file saved to: {output_filepath}")


if __name__ == '__main__':
    # 1. Set the path to your file that has the AI-corrected addresses with missing FIPS
    input_filename = 'geocoded_results/Production/van_30th_July_geocoded_bigquery_upload_reordered.csv'

    # 2. Define the name for your final, fully corrected output file
    output_filename = 'geocoded_results/Production/van_30th_July_geocoded_bigquery_upload_reordered_fips_fixed.csv'

    # 3. Run the script!
    fix_missing_fips(input_filename, output_filename)