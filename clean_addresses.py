# Importing necessary packages
from usps import USPSApi, Address

def clean_address(usps_client, address_info):
    """
    Takes a USPS API client and address components, validates the address,
    and returns the cleaned version.

    Args:
        usps_client (USPSApi): The instance of the USPSApi client.
        address_info (dict): A dictionary containing address components.

    Returns:
        str: A formatted string of the cleaned address or an error message.
    """
    try:
        # Create an Address object using the provided information
        # The library is smart enough to handle missing pieces.
        address = Address(
            street_1=address_info.get("street_1"),
            city=address_info.get("city"),
            state=address_info.get("state"),
            zip_code=address_info.get("zip_code")
        )

        # Call the USPS API to validate and standardize the address
        validation_result = usps_client.validate_address(address)

        # The cleaned address is in the 'result' attribute
        cleaned_address = validation_result.result['Address']

        # Format the cleaned address into a readable string
        formatted_address = (
            f"{cleaned_address['Street1']} "
            f"{cleaned_address.get('Street2', '')}\n"
            f"{cleaned_address['City']}, {cleaned_address['State']} "
            f"{cleaned_address['Zip5']}-{cleaned_address['Zip4']}"
        )
        return formatted_address

    except Exception as e:
        # Handle cases where the address might be invalid or an API error occurs
        return f"Could not validate address. Error: {e}"

# --- Main part of the script ---
if __name__ == "__main__":
    USPS_USER_ID = "NaitikiBioPhoenix"

    # Initialize the USPS API client with your User ID
    client = USPSApi(USPS_USER_ID)

    # Here is your list of messy Michigan addresses.
    # Add your addresses to this list to process them.
    messy_addresses = [
        {
            "street_1": "123 N Main Stret", # Misspelled "Street"
            "city": "Ann Arbor",
            "state": "MI"
        },
        {
            "street_1": "4400 vernor hwy", # Lowercase, no street type
            "city": "detroit",
            "state": "MI",
            "zip_code": "48209"
        },
        {
            "street_1": "Comerica Park", # A landmark name
            "city": "Detroit",
            "state": "MI"
        },
        {
            "street_1": "25 Roomis Street", # Misspelled "Loomis"
            "zip_code": "49507", # Zip code provided, but no city
            "state": "MI"
        },
        {
            "street_1": "999 Fake Address Ln", # An address that likely doesn't exist
            "city": "Nowhere",
            "state": "MI"
        }
    ]

    print("--- Starting Address Cleaning Process ---\n")

    # Loop through each messy address and clean it
    for i, addr in enumerate(messy_addresses):
        print(f"Original Address #{i+1}: {addr}")
        
        # Call our cleaning function
        cleaned = clean_address(client, addr)
        
        print(f"Cleaned Address #{i+1}: \n{cleaned}")
        print("-" * 20)