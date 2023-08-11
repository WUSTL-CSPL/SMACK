
import numpy as np
import os
from FAKEBOB.ivector_PLDA_SV import iv_SV
from FAKEBOB.gmm_ubm_SV import gmm_SV
from scipy.io.wavfile import read
import pickle
import sys

debug = False
n_jobs = 8

def iv_plda_sv(iv_model, probe_wav, benign_wavs_dir):
    ''' Test for ivector-PLDA-based SV
    '''
    probe_wav_list = []
    _, audio = read(probe_wav)
    probe_wav_list.append(audio)

    spk_id = iv_model[0]
    spk_id_extra = "FAKEBOB/test-iv-SV-" + spk_id
    iv_sv_model = iv_SV(spk_id_extra, iv_model, pre_model_dir="FAKEBOB/pre-models")

    benign_audio_list = []
    for audio_name in os.listdir(benign_wavs_dir):
        path = os.path.join(benign_wavs_dir, audio_name)
        _, audio = read(path)
        benign_audio_list.append(audio)

    _, scores = iv_sv_model.make_decisions(benign_audio_list, n_jobs=n_jobs, debug=debug)
    threshold = min(scores)

    _, score = iv_sv_model.make_decisions(probe_wav_list, n_jobs=n_jobs, debug=debug)
    final_score = score
    decision = 1 if final_score >= threshold else 0

    return decision, threshold, final_score


def gmm_ubm_sv(gmm_model, ubm, probe_wav, benign_wavs_dir):
    ''' Test for gmm-ubm-based SV
    '''
    probe_wav_list = []
    _, audio = read(probe_wav)
    probe_wav_list.append(audio)

    spk_id = gmm_model[0]
    spk_id_extra = "FAKEBOB/test-gmm-SV-" + spk_id
    gmm_sv_model = gmm_SV(spk_id_extra, gmm_model, ubm, pre_model_dir="FAKEBOB/pre-models")

    benign_audio_list = []
    for audio_name in os.listdir(benign_wavs_dir):
        path = os.path.join(benign_wavs_dir, audio_name)
        _, audio = read(path)
        benign_audio_list.append(audio)

    _, scores = gmm_sv_model.make_decisions(benign_audio_list, n_jobs=n_jobs, debug=debug)
    threshold = min(scores)

    _, score = gmm_sv_model.make_decisions(probe_wav_list, n_jobs=n_jobs, debug=debug)
    final_score = score
    decision = 1 if final_score >= threshold else 0

    return decision, threshold, final_score

def speaker_verification_iv(probe_wav, probe_label, benign_wavs_dir):
    
    iv_model_path = os.path.join("FAKEBOB/model", probe_label + ".iv") 
    if not os.path.isfile(iv_model_path):
        print("Unable to locate the ivector model file at", iv_model_path)
        
    with open(iv_model_path, "rb") as reader:
        iv_model = pickle.load(reader)
        iv_model[2] = os.path.abspath("FAKEBOB") + iv_model[2].partition("FAKEBOB")[2]
    
    decision, threshold, score = iv_plda_sv(iv_model, probe_wav, benign_wavs_dir)
    return decision, threshold, score

def speaker_verification_gmm(probe_wav, probe_label, benign_wavs_dir):
    
    gmm_model_path = os.path.join("FAKEBOB/model", probe_label + ".gmm")
    if not os.path.isfile(gmm_model_path):
        print("Unable to locate the gmm model file at", gmm_model_path)
    
    with open(gmm_model_path, "rb") as reader:
        gmm_model = pickle.load(reader)
        gmm_model[2] = os.path.abspath("FAKEBOB") + model[2].partition("FAKEBOB")[2]
    
    ubm = os.path.join("FAKEBOB/pre-models", "final.dubm")

    decision, threshold, score = gmm_ubm_sv(gmm_model, ubm, probe_wav, benign_wavs_dir)
    return decision, threshold, score

if __name__ == "__main__":
    
    probe_wav = sys.argv[1]
    model = sys.argv[2]
    target = sys.argv[3]
    
    benign_wavs_rootdir = 'FAKEBOB/data/test-set/'
    benign_wavs_dir = os.path.join(benign_wavs_rootdir, target)
    
    if model == "ivectorSV":
        decision, threshold, score = speaker_verification_iv(probe_wav, target, benign_wavs_dir)
        print(f"Tests on ivector-PLDA models passed! The decision is {decision}, acceptance threshold is {threshold}, and score is {score}.")
    
    elif model == "gmmSV":
        decision, threshold, score = speaker_verification_gmm(probe_wav, target, benign_wavs_dir)
        print(f"Tests on GMM-UBM models passed! The decision is {decision}, acceptance threshold is {threshold}, and score is {score}.")
