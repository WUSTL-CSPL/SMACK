import pronouncing
from .phonemic_dissimilarities import D
from .needleman_wunsch import needleman_wunsch


def get_phonemes(word):
    """
    This function uses the pronouncing library to get the phonemes of a word,
    and it removes stress indicators.
    """
    phones = pronouncing.phones_for_word(word)
    if not phones:
        return []
    # Removing numbers (stress indicators)
    return [''.join(filter(str.isalpha, ph)) for ph in phones[0].split()]


def ALINE_dissimilarity(string1, string2):
    words1 = string1.split()
    words2 = string2.split()

    phonemes1 = [ph for word in words1 for ph in (get_phonemes(word) + [' '])]
    phonemes2 = [ph for word in words2 for ph in (get_phonemes(word) + [' '])]
    
    norm_phonemes = min(len(phonemes1), len(phonemes2))

    # Remove trailing spaces
    phonemes1 = phonemes1[:-1]
    phonemes2 = phonemes2[:-1]

    # Align phonemes
    aligned_phonemes1, aligned_phonemes2 = needleman_wunsch(phonemes1, phonemes2)

    # Calculate phonemic similarities for aligned sequences
    total_similarity = 0
    
    for p1, p2 in zip(aligned_phonemes1, aligned_phonemes2):
        # Skip gap characters and spaces
        if p1 != '-' and p2 != '-' and p1 != ' ' and p2 != ' ':
            total_similarity += D(p1, p2)

    if norm_phonemes == 0:
        return 0
    
    return total_similarity / norm_phonemes

# For testing purposes
if __name__ == "__main__":
    
    string1 = "hola mundo"
    string2 = "hello world"
    
    similarity_score = ALINE_dissimilarity(string1, string2)
    print(f'Dissimilarity Score: {similarity_score}')