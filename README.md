# Geocoding and Address Correction Pipeline

This guide provides instructions on how to set up your local environment and run the geocoding pipeline.

## 🚀 Getting Started

Follow these steps to set up and run the geocoding pipeline on your machine.

### Prerequisites

- Python 3.7+
- A Google AI API Key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 1. Setup and Installation

First, clone the repository to your local machine and navigate into the project directory.

```powershell
git clone https://github.com/naitikiBio/Address-Validation-Tool.git
cd Address-Validation-Tool
```

Next, set up a virtual environment and install the required Python packages using `uv`. If you don't have `uv`, install it first with `pip install uv`.

```powershell
# Create a virtual environment
uv venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies from requirements.txt
uv pip install -r requirements.txt
```

### 2. Configure Your API Key

The script requires a Google AI API key to function. Create a file named `.env` in the root of the project folder.

Add your API key to the `.env` file as follows:

```
GOOGLE_AI_API_KEY="YOUR_API_KEY_HERE"
```

Replace `YOUR_API_KEY_HERE` with your actual Google AI API key.

### 3. How to Run the Script

The script is run from the command line. You need to specify which data source you want to process.

#### Command Syntax
```bash
python main.py <source_name>
```

Replace `<source_name>` with one of the available data source keys defined in the script.

#### Available Data Sources
You can use any of the following keys as the `source_name` or add yours in the main.py file:
- `demographics`
- `van`
- `AFI`
- `TestingFlat`
- `DentalPlaces`
- `fqhc`
- `rhc`
- `pharmacy`
- `oralhealthPLACES`
- `oralhealthCLINICS`
- `medicalPLACES`
- `mmhcTotalCounts`

#### Example
To process the `demographics` dataset, run:
```bash
python main.py demographics
```

The script will display its progress in the terminal and save the final output file in the `Production/Geocoded/` directory upon completion.

#### Trailing Zeros
If any of the final output column values have trailing zeros in them, modify the contents of the `remove_trailing_zeros.py` file, reflecting your actual file name and column names

#### Potential Gotchas
Make sure when running the code, to not modify the batch sizes for either Census or Gemini or the script might potentially get stuck (Census has a limit of 10k records per batch, better to be well below the limit)

