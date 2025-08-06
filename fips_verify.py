import requests
import json

def get_coords_from_fips(fips_code):
    """
    Looks up an 11-digit (Tract) or 15-digit (Block) FIPS code and returns its
    central latitude and longitude using the Census TIGERweb API.
    """
    fips_code = str(fips_code).strip()
    layer_index = None

    # Determine if it's a Tract or Block FIPS code based on length
    if len(fips_code) == 11:
        layer_index = 0 # Layer index for Census Tracts
    elif len(fips_code) == 15:
        layer_index = 1 # Layer index for Census Blocks
    else:
        print("Error: FIPS code must be 11 (Tract) or 15 (Block) digits long.")
        return None

    # Construct the API URL
    url = (
        f"https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/{layer_index}/query"
        f"?where=GEOID%3D'{fips_code}'"
        "&outFields=CENTLON,CENTLAT"
        "&f=json"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract the coordinates from the JSON response
        if data.get('features'):
            attributes = data['features'][0]['attributes']
            longitude = attributes.get('CENTLON')
            latitude = attributes.get('CENTLAT')
            return latitude, longitude
        else:
            print("FIPS code not found.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An API error occurred: {e}")
        return None

if __name__ == '__main__':
    # --- How to use ---
    # 1. Replace this with the FIPS code you want to check.
    #    It can be an 11-digit Tract code or a 15-digit Block code.
    fips_code_to_check = "26163525900"  # Example: A census tract in Wayne County, MI

    print(f"Looking up FIPS code: {fips_code_to_check}...")
    
    coordinates = get_coords_from_fips(fips_code_to_check)
    
    if coordinates:
        lat, lon = coordinates
        print(f"\nSuccess! ✨")
        print(f"  Latitude: {lat}")
        print(f"  Longitude: {lon}")
        # You can paste these coordinates into Google Maps to verify the location