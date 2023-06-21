# This script uses alternative_pronunciations.csv to calculate the phoneme similarity.

import csv
from collections import defaultdict
from .needleman_wunsch import needleman_wunsch

input_file_path = 'alternative_pronunciations.csv'
output_file_path = 'phonemic_similarities.csv'

alternative_pronunciations = defaultdict(list)

try:
    with open(input_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            word, pronunciations = row
            alternative_pronunciations[word].extend([p.split() for p in pronunciations.split(', ')])
except FileNotFoundError:
    print(f'Error: File {input_file_path} not found.')
except Exception as e:
    print(f'An error occurred: {str(e)}')


phoneme_counts = defaultdict(int)
phoneme_pair_counts = defaultdict(int)

# Loop through words and their alternative pronunciations
for word, pronunciations in alternative_pronunciations.items():
    for i in range(len(pronunciations)):
        for j in range(i + 1, len(pronunciations)):
            aligned_pron1, aligned_pron2 = needleman_wunsch(pronunciations[i], pronunciations[j])

            # Count the phonemes and phoneme substitutions
            for p1, p2 in zip(aligned_pron1, aligned_pron2):
                if p1 != '-':
                    phoneme_counts[p1] += 1
                if p2 != '-':
                    phoneme_counts[p2] += 1
                if p1 != '-' and p2 != '-':
                    phoneme_pair_counts[(p1, p2)] += 1

# Calculate the phonemic similarity S(a, b) for each phoneme pair
phonemic_similarity = defaultdict(float)

for (a, b), pair_count in phoneme_pair_counts.items():
    # Sort the phoneme pair so that (a, b) and (b, a) are treated as the same pair
    sorted_pair = tuple(sorted([a, b]))
    phonemic_similarity[sorted_pair] += pair_count

for (a, b), pair_sum in phonemic_similarity.items():
    phonemic_similarity[(a, b)] = pair_sum / (phoneme_counts[a] + phoneme_counts[b])

# Save the results to a CSV file
try:
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Phoneme Pair', 'Similarity Score'])
        for (a, b), similarity in phonemic_similarity.items():
            # Skip the phoneme pairs that are the same
            if a != b:
                writer.writerow([f'{a},{b}', similarity])
except Exception as e:
    print(f'An error occurred while saving the file: {str(e)}')