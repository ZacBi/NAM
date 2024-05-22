import json
import os
from tqdm import tqdm

def process_json_files(directory, output_file):
    # Check if the output file exists, and delete it if it does
    if os.path.exists(output_file):
        os.remove(output_file)

    # Create and write the column headers
    with open(output_file, 'a') as out_file:
        out_file.write('caption\timage\n')

    # Iterate over all files in the specified directory
    for filename in tqdm(os.listdir(directory),desc="Parsing dataset..."):
        if filename.endswith('.json'):
            # Construct the full file path
            filepath = os.path.join(directory, filename)

            # Open and read the JSON file
            with open(filepath, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    continue

                # Check if the status field is 'success'
                if data.get('status') == 'success':
                    # Extract the values of the caption and key fields
                    caption = data.get('caption', '')
                    key = data.get('key', '') + '.jpg'


                    with open(output_file, 'a') as out_file:
                        out_file.write(f"{caption}\t{key}\n")


process_json_files('/data/little-nine/GILL/datasets/cc3m/validation', '/data/little-nine/GILL/datasets/cc3m_val.tsv')