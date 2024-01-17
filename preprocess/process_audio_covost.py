import csv
import os

# Access the COVOST_ROOT environment variable
covost_root = os.environ.get('COVOST_ROOT')
if not covost_root:
    raise EnvironmentError("COVOST_ROOT environment variable is not set")

# Construct the file path
file_path = os.path.join(covost_root, 'covost_v2.en_de.test.tsv')

# Read and parse the TSV file
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t', fieldnames=['audio_path', 'english_sentence', 'german_translation', 'client_id'])
    for row in reader:
        data.append(row)

# Now 'data' contains your data as a list of dictionaries
print(data)
