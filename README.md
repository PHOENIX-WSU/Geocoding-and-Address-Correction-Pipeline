# Address Correction Tool

A Python utility that validates and corrects US addresses using the USPS API (v3). This tool helps ensure addresses are properly formatted and valid according to USPS standards.

TODO:
- Test more examples
- Automate the script for multiple address (i.e. read from the files)

## Prerequisites

- Python 3.8 or higher
- USPS API credentials (Consumer Key and Consumer Secret)

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
   uv venv
   .\venv\Scripts\Activate
   ```

3. Install dependencies:
   ```powershell
   # If using pip
   pip install requests python-dotenv

   # If using uv
   uv pip install requests python-dotenv
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
