import os

def levenshteinDistance(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def unique_wav_path(wav_path):

    directory, full_name = os.path.split(wav_path)
    name, extension = os.path.splitext(full_name)
    
    if not os.path.exists(wav_path):
        return wav_path

    counter = 1

    new_name = f"{name}_{counter}{extension}"
    new_path = os.path.join(directory, new_name)

    while os.path.exists(new_path):
        counter += 1
        new_name = f"{name}_{counter}{extension}"
        new_path = os.path.join(directory, new_name)

    return new_path