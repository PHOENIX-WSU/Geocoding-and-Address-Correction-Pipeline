import csv
import os

def fix_multiline_csv(input_file, output_file):
    """
    Reads a CSV file with improperly handled multiline quoted fields and
    writes a new, correctly formatted CSV file.

    Args:
        input_file (str): The path to the problematic source CSV file.
        output_file (str): The path where the clean CSV will be saved.
    """
    print(f"Attempting to fix '{input_file}'...")
    
    try:
        # Read the entire file content at once.
        # This is necessary because we can't process line-by-line when
        # a single logical row might be split across multiple physical lines.
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # The core of the solution: Use the csv module to parse the entire content.
        # The reader will correctly interpret newlines inside of quoted fields.
        reader = csv.reader(content.splitlines(), quotechar='"', skipinitialspace=True)

        # Write the cleaned data to the output file.
        # QUOTE_ALL ensures every field is wrapped in quotes for maximum compatibility.
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            
            processed_rows = 0
            for row in reader:
                writer.writerow(row)
                processed_rows += 1
        
        print("-" * 50)
        print(f"Successfully processed {processed_rows} logical rows.")
        print(f"Clean file has been saved as: '{output_file}'")
        print("-" * 50)

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("The file might have a severe corruption issue beyond just newlines.")

# --- How to use the script ---
if __name__ == "__main__":
    # 1. Set the name of your file that has the errors.
    #    Make sure it's in the same folder as this script, or provide the full path.
    source_csv = 'Data/van-complete-data.csv'

    # 2. Set the name for your new, fixed CSV file.
    fixed_csv = 'van-complete-data-line-fixed.csv'

    # 3. Run the cleaning function.
    fix_multiline_csv(source_csv, fixed_csv)