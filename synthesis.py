from ETTS.tester import ETTSInferenceModel
import shutil
import librosa
import torch
import sys
import numpy as np

import warnings
warnings.filterwarnings("ignore")


model_syn = ETTSInferenceModel(text_embed_dim=256,
                           emo_embed_dim=768,
                           nmels=80,
                           maxlength=1000,
                           ngst=64,
                           nlst=32,
                           model_dim=256,
                           model_hidden_size=512*2,
                           nlayers=5,
                           nheads=2,
                           vocoder_ckpt_path='./waveglow_256channels_universal_v5.pt',
                           etts_checkpoint='./LJ.ckpt',
                           sampledir='./SampleDir')


def audio_synthesis(l_emo_numpy, reference_audio, reference_text):
    
    device = torch.device("cuda:0")
    
    # Set global style references
    global_audio = reference_audio
    global_audio, sr_global = librosa.load(global_audio, sr=None)
    assert sr_global == 16000

    l_emo = torch.from_numpy(l_emo_numpy).float()
    l_emo = l_emo.to(device)

    with torch.no_grad():
        audio_tensor_syn = model_syn.synthesize_with_sample_lemo(global_audio, l_emo, reference_text, f'synthesis.wav')
        
        audio_numpy = audio_tensor_syn.squeeze(0).cpu().detach().numpy().astype('int16')
        
    return audio_numpy

# For testing purposes
if __name__ == '__main__':
    
    exp_p0_tmp = np.exp(np.random.randn(8, 32) * 10)
    softmax_p0_tmp = exp_p0_tmp / np.sum(exp_p0_tmp, axis=-1, keepdims=True)
    p_0 = softmax_p0_tmp * 0.25
    
    reference_audio = sys.argv[1]
    reference_text = sys.argv[2]
    
    audio_numpy = audio_synthesis(p_0, reference_audio, reference_text)
    