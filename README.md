# Address Correction Tool

A Python utility that validates and corrects US addresses using the USPS API (v3). This tool helps ensure addresses are properly formatted and valid according to USPS standards.

TODO:
- Test more examples
- Automate the script for multiple address (i.e. read from the files)

## Prerequisites

- Python 3.8 or higher
- USPS API credentials (Consumer Key and Consumer Secret)

## How to get USPS API Key
1. Go to the USPS website and make an account: https://developer.usps.com/
2. Go to the USPS Web Tools APIs: https://www.usps.com/business/web-tools-apis/welcome.htm?msockid=33097f2acda666c801436a2fccb567f9
4. Click Sign Up for USPS APIs
5. You will get a link in your email to complete email validation
6. You will get an email with the code that you need to enter to complete the login request
7. Click on the Apps tab and then click "Add App"
8. Enter the name for the app and enable Public Access I (no need for callback URL or description)
9. Click on the app name and scroll down to view the consumer key and consumer secret in the Credentials section

## Installation

1. Clone this repository:
   ```powershell
   git clone https://github.com/PHOENIX-WSU/Address-Validation-Tool
   cd Address-Validation-Tool
   ```

2. Set up a Python virtual environment using `venv`:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

   Alternatively, you can use `uv` (a faster alternative to venv):
   ```powershell
   pip install uv
   uv venv --python 3.11
   .\venv\Scripts\Activate
   ```

3. Install dependencies:
   ```powershell
   # If using pip
   pip install requests python-dotenv pandas thefuzz python-Levenshtein

   # If using uv
   uv pip install requests python-dotenv pandas thefuzz python-Levenshtein
   ```

## Configuration

1. Create a `.env` file in the project root:
   ```
   CONSUMER_KEY=your_usps_consumer_key
   CONSUMER_SECRET=your_usps_consumer_secret
   ```

2. Replace `your_usps_consumer_key` and `your_usps_consumer_secret` with your actual USPS API credentials.

## Usage

Run the script with a single address:
```powershell
python clean_addresses.py
```

The script will:
1. Load your USPS API credentials from the `.env` file
2. Get an OAuth access token
3. Validate and clean the provided address
4. Return the standardized address format

## Features

- OAuth 2.0 authentication with USPS API
- Address validation and standardization
- Proper error handling for invalid addresses
- Environment variable support for secure credential management

## Error Handling

The script handles various error cases:
- Missing API credentials
- Invalid addresses
- API connection issues
- Rate limiting

## Contributing

Feel free to submit issues and enhancement requests!
