import os
import time
import numpy as np
import soundfile as sf

from utils import levenshteinDistance, unique_wav_path
from CMUPhoneme.string_similarity import CMU_similarity
from ALINEPhoneme.string_dissimilarity import ALINE_dissimilarity
from NISQA.predict import NISQA_score
from synthesis import audio_synthesis
from google_ASR import google_ASR
from iflytek_ASR import iflytek_ASR
from speaker_sv import speaker_verification_gmm, speaker_verification_iv
from speaker_csi import gmm_ubm_csi, iv_plda_csi
from speaker_osi import gmm_ubm_osi, iv_plda_osi


class GradientEstimation:
    def __init__(self, reference_audio, reference_text, target_model, target, threshold_range, sigma, learning_rate, K):
        """
        :param sigma: Scaling factor for noise.
        :param learning_rate: Learning rate for updating the prosody vector.
        :param K: Number of noise vectors used for gradient approximation.
        """
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.target_model = target_model
        self.target = target
        self.threshold_range = threshold_range
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.K = K
    
    def _estimate_threshold(self, conf_score, is_accepted, epsilon=0.1):
        """
        Estimates the SV threshold by iteratively narrowing its range based on the maximum confidence score
        and the acceptance status.
        """
        # Retrieve the current threshold range
        inf, sup = self.threshold_range

        # Calculate current threshold estimate as the average of inf and sup
        theta = (inf + sup) / 2

        # Update the infimum and supremum based on the query results
        if is_accepted and conf_score < theta:
            # Theta is over-estimated
            sup = conf_score
        elif not is_accepted and conf_score > theta:
            # Theta is under-estimated
            inf = conf_score

        # Update the threshold range
        self.threshold_range = (inf, sup)

        # Check if the range is smaller than epsilon
        if sup - inf < epsilon:
            return True, theta  # Converged
        else:
            return False, theta  # Not converged
    
    
    def _calculate_loss(self, p_i):
        """ Calculates the loss of a given noise vector """

        l_emo_numpy = p_i.reshape(-1, 32)

        audio_numpy = audio_synthesis(l_emo_numpy, self.reference_audio, self.reference_text)
        tmp_audio_file = './SampleDir/synthesis.wav'
        
        audio_quality = NISQA_score(tmp_audio_file)
        
        if 'ASR' in self.target_model:

            if self.target_model == 'googleASR':
                transcription = google_ASR(tmp_audio_file)
                
            if self.target_model == 'iflytekASR':
                transcription = iflytek_ASR(tmp_audio_file)
                
            transcriped_file_name = self.target_model + '_' + transcription + '.wav'
            transcriped_file_path = unique_wav_path(os.path.join('./SampleDir', transcriped_file_name))
            sf.write(transcriped_file_path, audio_numpy, 22050)

            if levenshteinDistance(transcription, self.target) < 4:
                success_file_name = 'success_' + self.target_model + '_' + transcription + '.wav'
                success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                sf.write(success_file_path, audio_numpy, 22050)

            if transcription == 'NA':
                loss_levenshtein = 100
                loss_CMU = 0
                loss_ALINE = 10000
            else:
                # Divided by the average length of the target sentence and the transcribed sentence
                loss_levenshtein = levenshteinDistance(transcription, self.target) / ((len(transcription) + len(self.target)) / 2)
                loss_CMU = CMU_similarity(transcription, self.target)
                loss_ALINE = ALINE_dissimilarity(transcription, self.target)
                    
            # loss_levenshtein: [0, 1]; loss_CMU: [0, 1]; loss_ALINE: [0, 1000]; audio_quality: [0, 5]
            loss = 10*loss_levenshtein - 0.1*loss_CMU + 0.0001*loss_ALINE - 0.05*audio_quality
        
            print(f'loss:{loss}, loss_levenshtein: {10*loss_levenshtein}, loss_CMU: {-0.1*loss_CMU}, loss_ALINE: {0.0001*loss_ALINE}, audio_quality: {-0.05*audio_quality} \n')
            
        elif 'SV' in self.target_model:
                
            benign_wavs_rootdir = 'FAKEBOB/data/test-set/'
            benign_wavs_dir = os.path.join(benign_wavs_rootdir, self.target)
            
            if self.target_model == 'gmmSV':
                is_accepted, threshold, conf_score = speaker_verification_gmm(tmp_audio_file, self.target, benign_wavs_dir)
                    
            if self.target_model == 'ivectorSV':
                is_accepted, threshold, conf_score = speaker_verification_iv(tmp_audio_file, self.target, benign_wavs_dir)
                
            if is_accepted:
                success_file_name = 'success_' + self.target_model + '_' + self.target + '.wav'
                success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                sf.write(success_file_path, audio_numpy, 22050)
                
            converged, theta = self._estimate_threshold(conf_score, is_accepted)
            if converged:
                print(f"Threshold is now estimated at {theta}, the actual threshold is {threshold}")
            else:
                print(f"Threshold is now estimated at {theta}, the actual threshold is {threshold} \n")
                
            loss_adv = max(theta, conf_score) - conf_score
            
            loss = loss_adv - 0.02*audio_quality    
        
            print(f'loss:{loss}, conf_score:{conf_score}, loss_adv: {loss_adv}, audio_quality: {-0.02*audio_quality} \n')
            
        elif 'CSI' in self.target_model:
            
            if self.target_model == "ivectorCSI":
                max_score, target_label_score, decision, results_dict = iv_plda_csi(tmp_audio_file, self.target)
            
            elif self.target_model == "gmmCSI":
                max_score, target_label_score, decision, results_dict = gmm_ubm_csi(tmp_audio_file, self.target)
                
            print(f"The recognized speaker is {decision}, the attack success is {(decision == self.target)}")
            
            if decision == self.target:
                success_file_name = 'success_' + self.target_model + '_' + self.target + '.wav'
                success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                sf.write(success_file_path, audio_numpy, 22050)
                
            # This term has a maximum value of 0
            loss_adv = max_score - target_label_score
            
            loss = loss_adv - 0.02*audio_quality    
        
            print(f'loss:{loss}, conf_score:{target_label_score}, loss_adv: {loss_adv}, audio_quality: {-0.02*audio_quality} \n')
            
        elif 'OSI' in self.target_model:
            
            benign_wavs_dir = 'FAKEBOB/data/z-norm'
            
            if self.target_model == "ivectorOSI":
                max_score, max_score_label, target_label_score, min_threshold, decision, results_dict = iv_plda_osi(tmp_audio_file, benign_wavs_dir, self.target)
            
            elif self.target_model == "gmmOSI":
                max_score, max_score_label, target_label_score, min_threshold, decision, results_dict = gmm_ubm_osi(tmp_audio_file, benign_wavs_dir, self.target)
                
            target_score = results_dict[self.target][0]
            target_threshold = results_dict[self.target][1]
            is_accepted = results_dict[self.target][-1]
            
            if is_accepted:
                success_file_name = 'success_' + self.target_model + '_' + self.target + '.wav'
                success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                sf.write(success_file_path, audio_numpy, 22050)
            
            print(f"The recognition for the speaker {self.target} passed!")
                
            converged, theta = self._estimate_threshold(target_score, is_accepted)
            if converged:
                print(f"Threshold is now estimated at {theta}, the actual threshold is {target_threshold} \n")
            else:
                print(f"Threshold is now estimated at {theta}, the actual threshold is {target_threshold} \n")
                
            loss_adv = max(theta, target_score) - target_score
            
            loss = loss_adv - 0.02*audio_quality    
        
            print(f'loss:{loss}, conf_score:{target_score}, loss_adv: {loss_adv}, audio_quality: {-0.02*audio_quality} \n')
    
        return loss
    
    def _estimate_gradient(self, p_i):
        """      
        :param p_i: The prosody vector at iteration i.
        :return: Estimated gradient.
        """
        gradient = 0
        for k in range(self.K):
            # Create Gaussian noise, and update the prosody vector
            u_k = np.random.normal(0, 1, size=p_i.shape)
            loss = self._calculate_loss(p_i + self.sigma * u_k)
            gradient += loss * u_k
        
        gradient = gradient / (self.sigma * self.K)
        
        return gradient
    
    def refine_prosody_vector(self, p_i, num_iterations):
        """
        Refines an initially optimized prosody vector p_i through gradient estimation.
        
        :param num_iterations: Number of iterations to run the gradient estimation.
        :return: Refined prosody vector.
        """

        for _ in range(num_iterations):
            gradient = self._estimate_gradient(p_i)
            p_i = p_i + self.learning_rate * np.sign(gradient)
            
        return p_i

# For testing purposes
if __name__ == '__main__':
    
    reference_audio = './Original_MyVoiceIsThePassword.wav'
    reference_text = "My voice is the password"
    # target_model can be 'googleASR' or 'iflytekASR' or 'gmmSV' or 'ivectorSV'
    target_model = 'gmmSV'
    # target can be speaker id or the target transcription
    target = "librispeech_p1089"
    # Run a small number of iterations
    gradient_iterations = 20
    threshold_range = (-10, 10)
    
    # Initialize the GradientEstimation
    gradient_estimator = GradientEstimation(reference_audio, reference_text, target_model, target, threshold_range, sigma=0.1, learning_rate=0.01, K=20)

    # Initialize a prosody vector for testing
    exp_p0_tmp = np.exp(np.random.randn(8, 32) * 1)
    softmax_p0_tmp = exp_p0_tmp / np.sum(exp_p0_tmp, axis=-1, keepdims=True)
    p_0 = softmax_p0_tmp * 0.25

    p_refined = gradient_estimator.refine_prosody_vector(p_0, gradient_iterations)