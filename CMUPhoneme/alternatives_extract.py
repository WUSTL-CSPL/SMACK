# This script extracts words with alternative pronunciations from the CMU Pronouncing Dictionary 'cmudict.dict'
# and saves them to a CSV file 'alternative_pronunciations.csv'

import re
import csv
from collections import defaultdict

input_file_path = 'cmudict.dict'
output_file_path = 'alternative_pronunciations.csv'

# Regular expression to match words with alternative pronunciations
regex = re.compile(r'\(\d+\)')

alternative_pronunciations = defaultdict(list)

try:
    with open(input_file_path, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.startswith(';;;'):
                continue
            
            word, pronunciation = line.split(' ', 1)

            # Remove comments from the pronunciation
            pronunciation = pronunciation.split('#')[0].strip()

            # Extract the base word without the (number)
            base_word = re.sub(regex, '', word)

            alternative_pronunciations[base_word].append(pronunciation)
                
except FileNotFoundError:
    print(f'Error: File {input_file_path} not found.')
except Exception as e:
    print(f'An error occurred: {str(e)}')

try:
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Word", "Pronunciations"])
        for base_word, pronunciations in alternative_pronunciations.items():
            if len(pronunciations) > 1 or regex.search(f'{base_word}({pronunciations[0]})'):
                writer.writerow([base_word, ', '.join(pronunciations)])
except Exception as e:
    print(f'An error occurred while saving the output file: {str(e)}')

print(f'Words with alternative pronunciations have been saved to {output_file_path}')
