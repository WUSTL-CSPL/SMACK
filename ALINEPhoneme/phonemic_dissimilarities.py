# This script is used to calculate phonemic similarity based on ALINE phoneme set.

'''Define the phonological features for English phonemes
Raw data from https://sites.google.com/site/similaritymatrices/phonological-features/english-feature-specifications'''
phoneme_features = {
    'AA': {'Place': 60, 'Manner': 0, 'V': 0, 'Syl': 1, 'Voi': 100, 'Nas': 100, 'Lat': 0, 'Asp': 0, 'High': 0, 'Back': 50, 'Rou': 0, 'Lon': 100},
    'AE': {'Place': 70, 'Manner': 0, 'V': 0, 'Syl': 1, 'Voi': 100, 'Nas': 100, 'Lat': 0, 'Asp': 0, 'High': 0, 'Back': 100, 'Rou': 0, 'Lon': 0},
    'AH': {'Place': 60, 'Manner': 20, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': 50, 'Back': 50, 'Rou': 0, 'Lon': 0},
    'AO': {'Place': 60, 'Manner': 0, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': 0, 'Back': 50, 'Rou': 100, 'Lon': 100},
    'AW': {'Place': 62, 'Manner': 16, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 40, 'Back': 30, 'Rou': 40, 'Lon': 100},
    'AY': {'Place': 70, 'Manner': 16, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 40, 'Back': 70, 'Rou': 0, 'Lon': 100},
    'B': {'Place': 100, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'CH': {'Place': 75, 'Manner': 90, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'D': {'Place': 85, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'DH': {'Place': 90, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'EH': {'Place': 70, 'Manner': 20, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 50, 'Back': 100, 'Rou': 0, 'Lon': 0},
    'ER': {'Place': 60, 'Manner': 20, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 50, 'Back': 50, 'Rou': 0, 'Lon': 100},
    'EY': {'Place': 70, 'Manner': 28, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 70, 'Back': 100, 'Rou': 0, 'Lon': 100},
    'F': {'Place': 95, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'G': {'Place': 60, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'HH': {'Place': 10, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'IH': {'Place': 70, 'Manner': 20, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 50, 'Back': 100, 'Rou': 0, 'Lon': 0},
    'IY': {'Place': 70, 'Manner': 40, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 100, 'Back': 100, 'Rou': 0, 'Lon': 100},
    'JH': {'Place': 75, 'Manner': 90, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'K': {'Place': 60, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'L': {'Place': 85, 'Manner': 60, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 100, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'M': {'Place': 100, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 100, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'N': {'Place': 85, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 100, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'NG': {'Place': 60, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 100, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'OW': {'Place': 62, 'Manner': 28, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 70, 'Back': 0, 'Rou': 100, 'Lon': 100},
    'OY': {'Place': 64, 'Manner': 28, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 70, 'Back': 40, 'Rou': 0, 'Lon': 0},
    'P': {'Place': 100, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'R': {'Place': 85, 'Manner': 60, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'S': {'Place': 85, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'SH': {'Place': 75, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'T': {'Place': 85, 'Manner': 100, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'TH': {'Place': 90, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'UH': {'Place': 60, 'Manner': 40, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 100, 'Back': 0, 'Rou': 0, 'Lon': 0},
    'UW': {'Place': 60, 'Manner': 40, 'V': 1, 'Syl': 100, 'Voi': 100, 'Nas': 0, 'Lat': None, 'Asp': None, 'High': 100, 'Back': 0, 'Rou': 100, 'Lon': 100},
    'V': {'Place': 95, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'W': {'Place': 80, 'Manner': 60, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'Y': {'Place': 60, 'Manner': 60, 'V': 0, 'Syl': 0, 'Voi': 0, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'Z': {'Place': 85, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None},
    'ZH': {'Place': 75, 'Manner': 80, 'V': 0, 'Syl': 0, 'Voi': 100, 'Nas': 0, 'Lat': 0, 'Asp': 0, 'High': None, 'Back': None, 'Rou': None, 'Lon': None}
}

''' Constants
Values are set heuristicly based on data from https://hal.science/hal-01099239/document'''
v_cst = 0  # consonants
v_vwl = 1000  # vowels

# Feature sets
Rc = ['Place', 'Manner', 'Syl', 'Voi', 'Nas', 'Lat', 'Asp']
Rv = ['High', 'Back', 'Rou', 'Lon']

'''Salience of each feature
Raw data from https://sites.google.com/site/similaritymatrices/phonological-features'''
salience = {
    'Place': 40,
    'Manner': 50,
    'Syl': 5,
    'Voi': 10,
    'Nas': 10,
    'Lat': 10,
    'Asp': 5,
    'High': 5,
    'Back': 5,
    'Rou': 5,
    'Lon': 1,
}

def diff(p, q, f):
    p_val = phoneme_features.get(p, {}).get(f, 0)
    q_val = phoneme_features.get(q, {}).get(f, 0)
    if p_val is None or q_val is None:
        return 0
    return abs(p_val - q_val) * salience.get(f, 0)

def V(p):
    return v_cst if phoneme_features[p]['V'] == 0 else v_vwl

def D(a, b):
    if V(a) + V(b) == 2:
        R = Rv
    else:
        R = Rc
    return sum(diff(a, b, f) for f in R) + abs(V(a) - V(b))

# Test
if __name__ == "__main__":
    
    print(D('AA', 'AE'))