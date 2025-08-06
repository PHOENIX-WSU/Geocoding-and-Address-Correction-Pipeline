# AI-Enhanced Geocoding Pipeline

This script provides a robust, multi-stage pipeline to geocode address data from CSV files. It uses the U.S. Census Bureau's free batch geocoder and features a powerful, two-tiered AI fallback system using Google's Gemini models to correct and re-process addresses that initially fail.

The final output is a clean CSV file enriched with precise geographic data, including coordinates and FIPS codes, ready for analysis or for import into a database like Google BigQuery.

## ✨ Features

- **Batch Geocoding**: Processes thousands of addresses efficiently using the U.S. Census Bureau API.
- **Smart AI Fallback**: Automatically uses Google AI (Gemini Flash & Pro) to fix addresses that the Census Geocoder can't match.
- **FIPS Code Correction**: Uses a secondary API (FCC) to find FIPS codes for addresses that were successfully geocoded by AI but are missing census tract data.
- **Configurable**: Easily add new data sources by editing a single Python dictionary—no need to change the core logic.
- **Detailed Output**: Generates clean, final files with original data, coordinates, FIPS codes, and metadata.
- **Command-Line Driven**: Simple to run for different datasets with a single command.

## ⚙️ How It Works

The pipeline operates in three main stages:

### Stage 1: Preparation

1. Reads your source CSV file.
2. Standardizes the address columns (street, city, state, zip) based on your configuration.
3. Removes any rows that are missing a street address.
4. Assigns a Unique ID to each row to reliably merge the data back together later.
5. Saves a temporary, clean CSV file formatted specifically for the Census Geocoder API.

### Stage 2: Geocoding & AI Correction

1. The prepared CSV is sent to the U.S. Census Geocoder for batch processing.
2. The script analyzes the results. Any addresses that could not be matched (No_Match or Tie) are sent to the AI fallback system.
3. **Tier 1 (Gemini Flash)**: The fast and efficient Gemini Flash model attempts to correct the faulty address. If successful, the corrected address is geocoded individually.
4. **Tier 2 (Gemini Pro)**: If Gemini Flash fails, the more powerful Gemini Pro model is used for a second attempt at correction and geocoding.
5. All raw results from this stage are saved for debugging and review.

### Stage 3: Finalization & Cleanup

1. The script takes the geocoded data and cleans the FIPS codes to ensure they are in a standard format.
2. For any AI-corrected addresses that have coordinates but are still missing FIPS codes, it makes a final call to the FCC Area API to fill in the gaps.
3. The geocoded data is merged back with your original input data using the Unique ID.
4. Metadata columns like dataset (from your config) and dategeocoded (the current date) are added.
5. The final, production-ready CSV is saved to the specified output directory.

## 🚀 Getting Started

Follow these steps to set up and run the geocoding pipeline.

### Prerequisites

- Python 3.7+
- A Google AI API Key. You can get one from Google AI Studio.
### 1. File & Folder Structure

Before running the script, make sure your project directory is set up as follows. The script expects this structure to read input files and save results.

```plaintext
your-project-folder/
├── geocoding_script.py         # Your main Python script
├── .env                        # You will create this file for your API key
├── requirements.txt            # You will create this file for dependencies
│
├── Data/                       # Folder for your INPUT data
│   └── patientDEMOGRAPHOGRAPHICS-11th-July-2025.csv
│
└── geocoded_results/           # Folder for your OUTPUT data
    └── Production/             # Subfolder for the final, clean files
```

### 2. Install Dependencies

The script relies on several Python packages.

First, create a file named `requirements.txt` in your project folder and add the following lines:

```plaintext
pandas
requests
python-dotenv
google-generativeai
```

Now, open your terminal or command prompt, navigate to your project folder, and run this command to install them:

```bash
pip install -r requirements.txt
```

### 3. Configure Your API Key

Your Google AI API Key should be kept secret. The script uses a `.env` file to load it securely.

Create a file named `.env` in the root of your project folder.

Add your API key to the file like this:

```
GOOGLE_AI_API_KEY="YOUR_API_KEY_HERE"
```

Replace `YOUR_API_KEY_HERE` with your actual key.

## 🔧 Configuring Data Sources

You can easily process new datasets by adding a configuration entry to the CONFIG dictionary at the top of the script.

Here is a template for adding a new source. Just copy and paste this into the CONFIG dictionary and modify the values.

```python
'new_source_name': {
    # Path to the raw CSV file you want to process.
    'input_file': 'Data/my_new_data.csv',

    # Path to save the intermediate file prepared for the geocoder.
    'geocoding_input': 'Data/prepared_new_source_for_geocoding.csv',

    # Path to save the raw results from the geocoder (including failures).
    'geocoded_output': 'geocoded_results/new_source_geocoded_raw.csv',

    # Path to save the final, clean, production-ready file.
    'final_output': 'geocoded_results/Production/new_source_final_for_bigquery.csv',

    # Maps your CSV's column names to the standard names the script needs.
    # Change the values (e.g., 'AddressLine1') to match the column headers in your CSV.
    'column_map': {
        'street': 'AddressLine1',
        'city': 'CityName',
        'state': 'StateAbbr',
        'zip': 'PostalCode'
    },

    # A descriptive name that will be added to the 'dataset' column in the final output file.
    'dataset_name': 'My New Dataset (Geocoded)',

    # Set to `True` if your data is missing a state column and should default to 'MI'.
    # Set to `False` if your data already has a state column.
    'add_state_mi': False
}
```

## ▶️ How to Run the Script

The script is run from your terminal. You must specify which data source from the CONFIG dictionary you want to process.

### Syntax
```bash
python geocoding_script.py <source_name>
```

Replace `geocoding_script.py` with the name of your script file and `<source_name>` with the key you defined in the CONFIG dictionary.

### Examples
To process the demographics dataset:
```bash
python geocoding_script.py demographics
```

To process the van dataset:
```bash
python geocoding_script.py van
```

The script will print its progress to the console, showing you each stage as it completes, and will notify you of the final accuracy and where the output file is saved.

## Contributing

Feel free to submit issues and enhancement requests!
