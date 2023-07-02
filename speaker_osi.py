import numpy as np
import os
import sys
from FAKEBOB.ivector_PLDA_OSI import iv_OSI
from FAKEBOB.gmm_ubm_OSI import gmm_OSI
from scipy.io.wavfile import read
import pickle

debug = False
n_jobs = 8

def get_spk_list():
    spk_model_dir = 'FAKEBOB/model'
    spk_id_list = []
    for spk_id in os.listdir(spk_model_dir):
        if spk_id.endswith('.iv'):
            spk_id_short = spk_id.split('.')[0]
            if spk_id_short not in spk_id_list:
                spk_id_list.append(spk_id_short)
    return spk_id_list
                
def iv_plda_osi(probe_wav, benign_wavs_dir, target_label):
    spk_id_list = get_spk_list()
    iv_model_paths = [os.path.join('FAKEBOB/model', spk_id + ".iv") for spk_id in spk_id_list]
    iv_model_list = []
    for path in iv_model_paths:
        with open(path, "rb") as reader:
            model = pickle.load(reader)
            model[2] = "FAKEBOB" + model[2].partition("FAKEBOB")[2]
            iv_model_list.append(model)
    
    group_id = "test-iv-OSI"
    iv_osi_model = iv_OSI(group_id, iv_model_list, pre_model_dir='FAKEBOB/pre-models')
    spk_ids = np.array(iv_osi_model.spk_ids) # ['librispeech_p1089' 'librispeech_p1188' 'speechaccent_14' 'timit_MDAW1' 'vctk_p227']

    probe_audio_list = []
    path = os.path.join(probe_wav)
    _, audio = read(path)
    probe_audio_list.append(audio)
    
    _, scores = iv_osi_model.make_decisions(probe_audio_list, debug=debug, n_jobs=n_jobs)
    target_scores = scores
    # print(target_scores)

    # construction thresholds by running benign wavs against their own models
    thresholds = []
    for i, spk_id in enumerate(spk_ids):
        benign_wavs_paths = []
        benign_speaker_dir = os.path.join(benign_wavs_dir, spk_id)
        for file in os.listdir(benign_speaker_dir):
            _, audio = read(os.path.join(benign_speaker_dir, file))
            benign_wavs_paths.append(audio)
        _, scores = iv_osi_model.make_decisions(benign_wavs_paths, debug=debug, n_jobs=n_jobs)
        benign_scores = scores[:, i]
        thresholds.append(min(benign_scores))
    thresholds = np.array(thresholds)
    # print(thresholds)
    
    # generate decision, and the largest score 
    candidate_speaker_index = np.where(target_scores > thresholds)[0]
    # print(candidate_speaker_index)
    
    target_label_index = spk_ids.tolist().index(target_label)
    target_class_score = target_scores[target_label_index]
    
    if candidate_speaker_index.size == 0:
        decision = None
        max_score = max(target_scores)
        max_score_speaker = spk_ids[target_scores.tolist().index(max_score)]
    else:
        max_score_speaker_index = None
        max_score = -float('inf')
        for i in candidate_speaker_index:
            if target_scores[i] >= max_score:
                max_score = target_scores[i]
                max_score_speaker_index = i
        decision = spk_ids[max_score_speaker_index]
        max_score_speaker = decision
    
    results_dict = {}
    for i in range(0, len(target_scores)):
        results_dict[spk_ids[i]] = [target_scores[i], thresholds[i], 1 if target_scores[i] >= thresholds[i] else 0]

    return max_score, max_score_speaker, target_class_score, min(thresholds), decision, results_dict

def gmm_ubm_osi(probe_wav, benign_wavs_dir, target_label):
    spk_id_list = get_spk_list()
    gmm_model_list = []
    gmm_model_paths = [os.path.join('FAKEBOB/model', spk_id + ".gmm") for spk_id in spk_id_list]
    for path in gmm_model_paths:
        with open(path, "rb") as reader:
            model = pickle.load(reader)
            model[2] = "FAKEBOB" + model[2].partition("FAKEBOB")[2]
            gmm_model_list.append(model)
    
    ubm = os.path.join('FAKEBOB/pre-models', "final.dubm")
            
    group_id = "test-gmm-OSI"
    gmm_osi_model = gmm_OSI(group_id, gmm_model_list, ubm, pre_model_dir='FAKEBOB/pre-models')
    spk_ids = np.array(gmm_osi_model.spk_ids) # ['librispeech_p1089' 'librispeech_p1188' 'speechaccent_14' 'timit_MDAW1' 'vctk_p227']

    probe_audio_list = []
    path = os.path.join(probe_wav)
    _, audio = read(path)
    probe_audio_list.append(audio)
    
    _, scores = gmm_osi_model.make_decisions(probe_audio_list, debug=debug, n_jobs=n_jobs)
    target_scores = scores
    # print(target_scores)

    # construction thresholds by running benign wavs against their own models
    thresholds = []
    for i, spk_id in enumerate(spk_ids):
        benign_wavs_paths = []
        benign_speaker_dir = os.path.join(benign_wavs_dir, spk_id)
        for file in os.listdir(benign_speaker_dir):
            _, audio = read(os.path.join(benign_speaker_dir, file))
            benign_wavs_paths.append(audio)
        _, scores = gmm_osi_model.make_decisions(benign_wavs_paths, debug=debug, n_jobs=n_jobs)
        benign_scores = scores[:, i]
        thresholds.append(min(benign_scores))
    thresholds = np.array(thresholds)
    # print(thresholds)
    
    # generate decision, and the largest score 
    candidate_speaker_index = np.where(target_scores > thresholds)[0]
    # print(candidate_speaker_index)
    
    target_label_index = spk_ids.tolist().index(target_label)
    target_class_score = target_scores[target_label_index]
    
    if candidate_speaker_index.size == 0:
        decision = None
        max_score = max(target_scores)
        max_score_speaker = spk_ids[target_scores.tolist().index(max_score)]
    else:
        max_score_speaker_index = None
        max_score = -float('inf')
        for i in candidate_speaker_index:
            if target_scores[i] >= max_score:
                max_score = target_scores[i]
                max_score_speaker_index = i
        decision = spk_ids[max_score_speaker_index]
        max_score_speaker = decision
    
    results_dict = {}
    for i in range(0, len(target_scores)):
        results_dict[spk_ids[i]] = [target_scores[i], thresholds[i], 1 if target_scores[i] >= thresholds[i] else 0]


    return max_score, max_score_speaker, target_class_score, min(thresholds), decision, results_dict

if __name__ == "__main__":
    
    probe_wav = sys.argv[1]
    model = sys.argv[2]
    target_label = sys.argv[3]
    
    benign_wavs_dir = 'FAKEBOB/data/z-norm'
    
    if model == "ivectorOSI":
        max_score, max_score_label, target_label_score, min_threshold, decision, results_dict = iv_plda_osi(probe_wav, benign_wavs_dir, target_label)
        print(max_score, max_score_label, target_label_score, min_threshold, decision, results_dict)
    
    elif model == "gmmOSI":
        max_score, max_score_label, target_label_score, min_threshold, decision, results_dict = gmm_ubm_osi(probe_wav, benign_wavs_dir, target_label)
        print(max_score, max_score_label, target_label_score, min_threshold, decision, results_dict)
