# This script uses alternative_pronunciations.csv to calculate the phonemic similarity between two strings
# For testing purposes, you can run this script directly and input two strings

import os
import csv
from .needleman_wunsch import needleman_wunsch
from g2p_en import G2p

# Load phonemic similarity scores
phonemic_similarity = {}
CMU_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phonemic_similarities.csv')
with open(CMU_file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        phonemes, similarity = row
        phoneme_a, phoneme_b = phonemes.split(',')
        phonemic_similarity[(phoneme_a, phoneme_b)] = float(similarity)


# Function to convert string to phoneme sequence using g2p_en
def string_to_phonemes(input_string):
    g2p = G2p()
    return g2p(input_string)


# Function to calculate phoneme similarity
def CMU_similarity(string1, string2):
    # Convert strings to phoneme sequences
    phonemes1 = string_to_phonemes(string1)
    phonemes2 = string_to_phonemes(string2)
    
    norm_phonemes = min(len(phonemes1), len(phonemes2))

    # Align the phoneme sequences
    aligned_phonemes1, aligned_phonemes2 = needleman_wunsch(phonemes1, phonemes2)

    # Calculate the similarity score
    similarity_score = 0

    for p1, p2 in zip(aligned_phonemes1, aligned_phonemes2):
        if p1 != '-' and p2 != '-':
            if p1 == p2:
                similarity_score += 1
            else:
                pair = tuple(sorted([p1, p2]))
                similarity_score += phonemic_similarity.get(pair, 0)

    # Calculate the average score
    average_score = similarity_score / norm_phonemes

    return average_score


# For testing purposes
if __name__ == "__main__":

    string1 = "hello world"
    string2 = "hola mundo"

    similarity_score = CMU_similarity(string1, string2)
    print(f'Similarity Score: {similarity_score}')