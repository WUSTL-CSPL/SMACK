# https://sites.google.com/site/similaritymatrices/computing-the-matrices?authuser=0
# https://sites.google.com/site/similaritymatrices/phonological-features/english-feature-specifications?authuser=0

# Define the phonological features for English phonemes
phoneme_features = {
    # Add your phoneme features here
    # Example: 'a': {'V': 1, 'Syl': 1, 'Voi': 1, 'Nas': 0, 'Lat': 0, 'Vic': 'a', 'High': 0, 'Back': 1, 'Round': 0, 'Long': 0},
    # ...
}

# Constants
Csub = 3500
Cvwl = 1000
Cskip = 1000

# Feature sets
Rc = ['Place', 'Manner', 'Syllabic', 'Voice', 'Nasal', 'Lateral', 'Aspirated']
Rv = ['High', 'Back', 'Round', 'Long']

def diff(p, q, f):
    return abs(phoneme_features[p][f] - phoneme_features[q][f])

def V(p):
    return phoneme_features[p]['V']

def sigma_sub(p, q):
    if V(p) + V(q) == 2:
        R = Rv
    else:
        R = Rc
    return Csub - Cvwl * V(p) * V(q) - sum(diff(p, q, f) for f in R)

def sigma_skip():
    return Cskip / 100

# Test
print(sigma_sub('a', 'b'))
print(sigma_skip())
