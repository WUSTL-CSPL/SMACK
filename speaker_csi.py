import numpy as np
import os
import sys
from FAKEBOB.ivector_PLDA_CSI import iv_CSI
from FAKEBOB.gmm_ubm_CSI import gmm_CSI
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
                
def iv_plda_csi(probe_wav, target_label):
    spk_id_list = get_spk_list()
    iv_model_paths = [os.path.join('FAKEBOB/model', spk_id + ".iv") for spk_id in spk_id_list]
    iv_model_list = []
    for path in iv_model_paths:
        with open(path, "rb") as reader:
            model = pickle.load(reader)
            model[2] = "FAKEBOB" + model[2].partition("FAKEBOB")[2]
            iv_model_list.append(model)
    
    group_id = "test-iv-CSI"
    iv_csi_model = iv_CSI(group_id, iv_model_list, pre_model_dir='FAKEBOB/pre-models')
    spk_ids = np.array(iv_csi_model.spk_ids) # ['librispeech_p1089' 'librispeech_p1188' 'speechaccent_14' 'timit_MDAW1' 'vctk_p227']

    probe_audio_list = []
    path = os.path.join(probe_wav)
    _, audio = read(path)
    probe_audio_list.append(audio)
    
    _, scores = iv_csi_model.make_decisions(probe_audio_list, debug=debug, n_jobs=n_jobs)
    target_scores = scores.tolist()
    # print(target_scores)
    
    max_score = max(target_scores)
    decision = spk_ids[target_scores.index(max_score)]
    
    target_label_index = spk_ids.tolist().index(target_label)
    target_class_score = target_scores[target_label_index]
    
    results_dict = {}
    for i in range(0, len(target_scores)):
        results_dict[spk_ids[i]] = [target_scores[i], 1 if target_scores[i] == max_score else 0]

    return max_score, target_class_score, decision, results_dict

def gmm_ubm_csi(probe_wav, target_label):
    spk_id_list = get_spk_list()
    gmm_model_paths = [os.path.join('FAKEBOB/model', spk_id + ".gmm") for spk_id in spk_id_list]
    gmm_model_list = []
    for path in gmm_model_paths:
        with open(path, "rb") as reader:
            model = pickle.load(reader)
            model[2] = "FAKEBOB" + model[2].partition("FAKEBOB")[2]
            gmm_model_list.append(model)
            
    group_id = "test-gmm-CSI"
    gmm_csi_model = gmm_CSI(group_id, gmm_model_list, pre_model_dir='FAKEBOB/pre-models')
    spk_ids = np.array(gmm_csi_model.spk_ids) # ['librispeech_p1089' 'librispeech_p1188' 'speechaccent_14' 'timit_MDAW1' 'vctk_p227']

    probe_audio_list = []
    path = os.path.join(probe_wav)
    _, audio = read(path)
    probe_audio_list.append(audio)
    
    _, scores = gmm_csi_model.make_decisions(probe_audio_list, debug=debug, n_jobs=n_jobs)
    target_scores = scores.tolist()
    # print(target_scores)
    
    max_score = max(target_scores)
    decision = spk_ids[target_scores.index(max_score)]
    
    target_label_index = spk_ids.tolist().index(target_label)
    target_class_score = target_scores[target_label_index]
    
    results_dict = {}
    for i in range(0, len(target_scores)):
        results_dict[spk_ids[i]] = [target_scores[i], 1 if target_scores[i] == max_score else 0]

    return max_score, target_class_score, decision, results_dict

if __name__ == "__main__":
    
    probe_wav = sys.argv[1]
    model = sys.argv[2]
    target = sys.argv[3]
    
    if model == "ivectorCSI":
        max_score, target_label_score, decision, results_dict = iv_plda_csi(probe_wav, target)
        print(max_score, target_label_score, decision, results_dict)
    
    elif model == "gmmCSI":
        max_score, target_label_score, decision, results_dict = gmm_ubm_csi(probe_wav, target)
        print(max_score, target_label_score, decision, results_dict)
