# SMACK

This repository hosts the source code for paper "SMACK: Semantically Meaningful Adversarial Audio Attack". The paper has been accepted by [32nd USENIX Security Symposium, 2023](https://www.usenix.org/conference/usenixsecurity23).

SMACK is the abbreviation for **S**emantically **M**eaningful Adversarial **A**udio Atta**CK**. As suggested by its name, SMACK is a type of adversarial audio attack against automatic speech recognition (ASR) and speaker recognition (SR) systems. Different from traditional attacks in the field, SMACK additionally aims to preserve the semantic information of human speech instead of restricting perturbations within Lp norms to achieve naturalness in the resulted adversarial examples. Details regarding this work can be found in the [Sec'23 paper](https://www.usenix.org/system/files/sec23summer_371-yu_zhiyuan-prepub.pdf) and project website at [https://semanticaudioattack.github.io/](https://semanticaudioattack.github.io/) (in built).

# Hardware and Software Dependencies

The attack framework can run on a machine with a moderate CPU, at least 16GB of available RAM, and a GPU with at least 8GB VRAM. The system was tested stable with AMD Ryzen 9 3900X 12-Core Processor accompanying RTX 3070Ti and 32GB memory. No other specific hardware is required, but the variance in hardware can lead to differences in run-time. 

SMACK was implemented in Python, and the environment was set up using Miniconda 4.12.0 on Ubuntu 22.0.4. The used machine learning framework is Pytorch, and other associated dependencies are encapsulated in the YAML file. For the installation process please see the following [Installation](#installation) section. 

# Installation

The setup of this repo mainly consists of two components, (1) attack framework and (2) speaker recognition system. 

The dependencies required by the attack framework are encapsulated in the *smack.yml* file. To setup the environment using conda, please use the following commands:
```sh
$ cd <the_path_to_the_folder>
$ conda env create -f smack.yml
$ conda activate smack
```

Please find the supplementary files [here](https://drive.google.com/file/d/12vUxRaIRDaD_prg8F-vpb5oUvWMOPqsl/view?usp=sharing). Please place the pre-trained model files *LJ.ckpt* and *waveglow_256channels_universal_v5.pt* under *SMACK*. Other files are needed by FAKEBOB setup as detailed in the following.

For the setup of speaker recognition systems, we follow the existing work *FAKEBOB* and use the Kaldi toolkit. Notably, this process can be time-consuming and requires modification of the shell configuration file. Therefore, we wrote a dedicated tutorial detailing all steps in the *setup_SR.md* file. During the installation of *FAKEBOB*, customized scripts within *FAKEBOB_File_Replace.zip* are needed to replace the ones in *FAKEBOB*, please refer to *setup_SR.md* for the specifics. Alternatively, you can also refer to the original *FAKEBOB* [GitHub repo](https://github.com/FAKEBOB-adversarial-attack/FAKEBOB) for setting up the SR models. 

# Repository Overview

The repository contains both programs/files for the generative model and those for the SMACK attack framework. 

The *ETTS*, *wav2vec2*, and *waveglow* folders contain files necessary for the generative model. The *synthesis.py* file serves as a functional call to the generative model without having to expose every detail externally. 

The *genetic.py* contains the implementation of the Adapted Genetic Algorithm, whose design is described in Section 5.2 in the paper. A key enabler of this algorithm is the InsDel operator, which is implemented in the *_insdel()* function defined in *class GeneticAlgorithm()*. The *gradient.py* contains the implementation of the gradient estimation algorithm, whose design is given in Equations (1) and (2) in Section 5.2 of the paper. Combining these two, *attack.py* takes configurations of attack parameters and is used to launch the attack.

Deeper into the attack details, the *CMUPhoneme* folder contains all the raw data and programs used to construct phoneme similarity based on CMU’s Pronouncing Dictionary. Regarding raw data, the *cmudict.dict* file is the official release of CMU’s Pronouncing Dictionary, which was downloaded from the [official GitHub repo](https://github.com/cmusphinx/cmudict). With this raw data, we use the script *alternatives_extract.py* to collect alternative pronunciations of words documented in the dictionary. The *needleman_wunsch.py* contains the implementation of the Needleman-Wunsch algorithm that aligns the phoneme sequences. Then we use *phonemic_similarities.py* to calculate similarity scores between phonemes and store the raw data to *phonemic_similarities.csv*. At last, we implement calculating the phonological similarity of phoneme sequences in *string_similarity.py*. The *ALINEPhoneme* folder contains files for calculating the phonological similarity based on Kondrak’s ALINE cognate alignment system. The files inside follow similar structures and naming strategies. The sources of the raw data are also documented in the programs. Both of the two designs are reflected in Section 6.2.

The *google_ASR.py* and *iflytek_ASR.py* files are implemented queries to commercial ASR systems. They also embed testing functionalities with main functions. *speaker_csi.py*, *speaker_osi.py*, and *speaker_sv.py* are the scripts that call Kaldi speaker recognition functions. The *utils.py* file contains some utility functions such as Levenshtein Distance calculation.

# Basic Tests

The SMACK attack is enabled by the optimization framework consisting of adapted genetic algorithm (AGA) and gradient estimation scheme (ES). Each component can be individually tested for adversarial optimization.

1. The command for basic tests of the AGA is:
```sh
$ python3 genetic.py 
```

You are expected to see printed outputs like the following, comprising the unique ID of individuals in the population and their fitness value breakdown:
```
Threshold is now estimated at 0.0, the actual threshold is 0.13009999999999877 

[Individual 011611ae-b06c-481a-a085-aa1f98a4428f Fitness: 0.08]
[Individual 011611ae-b06c-481a-a085-aa1f98a4428f conf_score: 0.06]
[Individual 011611ae-b06c-481a-a085-aa1f98a4428f Adv: -0.00]
[Individual 011611ae-b06c-481a-a085-aa1f98a4428f NISQA: 0.08]
```

2. Similarly, the basic functionality of the gradient estimation part can be tested with:
```sh
$ python3 gradient.py
```
And the expected ouput should look like this:
```
Threshold is now estimated at 0.0, the actual threshold is 0.13009999999999877 

loss:-0.0680399227142334, conf_score:0.03159999999999741, loss_adv: 0.0, audio_quality: -0.0680399227142334
```

Note that the above two tests are based on speaker recognition systems, so they examine the setup and functionality of both attack algorithms and SR models. If the SR models are not properly setup prior to tests, the basic tests would fail automatically. You can also choose to modify the main function and test with ASR models alternatively.

3. The normal functionality of target models can be tested by running corresponding scripts. For ASR, we can use an adversarial example to test *iflytekASR* model:
```sh
$ python3 iflytek_ASR.py "SMACK_Examples/iflytekASR_THEY DID NOT HAVE A LIGHT.wav"
```
And you are expected to see the output as:
```sh
iflytek ASR Result after 1 connection retries: THEY DID NOT HAVE A LIGHT
```
Similarly, the command and expected output for Google speech-to-text model is:
```sh 
$ python3 google_ASR.py "SMACK_Examples/success_gmmSV_librispeech_p1089.wav"
Google ASR Result after 0 retries: MY VOICE IS THE PASSWORD
```

4. For SR models, the testing command with the provided adversarial example is:
```sh
$ python3 speaker_sv.py "SMACK_Examples/success_gmmSV_librispeech_p1089.wav" gmmSV librispeech_p1089
```
And you are expected to see the output as:
```sh
Tests on GMM-UBM models passed! The decision is 1, acceptance threshold is 0.13009999999999877, and score is 0.16089999999999804.
```

# Usage

To run the attack, please use the *attack.py* with the specified parameters including the target model ('googleASR' or 'iflytekASR' or 'gmmSV' or 'ivectorSV'), content text (referred to as t0 in the paper), and target (transcription or speaker label). For further details regarding parameter tuning to adapt SMACK to diverse targets, please see [Other Notes](#other-notes).

## Attack ASR

In the attack against ASR systems, we provide two real-world speech recognition services as the target models, iFlytek and Google. To run the attack, please use the following command:
```sh
$ python3 attack.py --audio "./Original_TheyDenyTheyLied.wav" --model iflytekASR --content "They deny they lied" --target "They did not have a light"
```

You are expected the see printed outputs similar to the basic tests. An example output produced by this attack is recorded in the *SMACK_Examples/iflytekASR_THEY DID NOT HAVE A LIGHT.txt* file along with the resulted adversarial example *SMACK_Examples/iflytekASR_THEY DID NOT HAVE A LIGHT.wav*. The example is named as its transcription by the target model, and its success in misleading transcription can be validated in the basic test 3. This attack against iFlyTek generally takes 4 compute-hour on our machine, which can vary depending on your hardware.

Similarly, the attack can be launched against Google speech-to-text service using the command:
```sh
$ python3 attack.py --audio "./Original_SamiGotAngry.wav" --model googleASR --content "Sami got angry" --target "Send me that"
```
And the sample adversarial audio and associated outputs are recorded in *SMACK_Examples/googleASR_SEND ME THAT.txt*. This attack against Google speech-to-text takes 1 compute-hour on our machine. 

## Attack SR

In the attack against SR systems, we provide two types of well-established models deployed in the Kaldi toolkit, GMM-UBM and ivector-PLDA. The installation guide can be found in the *setup_SR.md*. The command for running the attack against GMM-based model is as follows:
```sh
$ python3 attack.py --audio "./Original_MyVoiceIsThePassword.wav" --model gmmSV --content "My voice is the password" --target librispeech_p1089
```

By using this command, we conduct an inter-gender attack, that is, the reference audio is uttered by a woman while the target speaker *librispeech_p1089* is a man. The original benign speech samples of *librispeech_p1089* can be found in the *FAKEBOB/data/test-set/librispeech_p1089* folder. An adversarial example generated by the attack is provided in *SMACK_Examples/success_gmmSV_librispeech_p1089.wav*, and its success can be validated in the basic test 4. The associated printed output throughout the optimization process is also provided in *SMACK_Examples/success_gmmSV_librispeech_p1089.txt*. Each query to SR models is much more time-consuming than ASR models; therefore this attack against GMM-based models can take about 8 compute-hour.

The command used to launch the attack against ivector-based SR model is similar:
```sh
$ python3 attack.py --audio "./Original_MyVoiceIsThePassword.wav" --model ivectorCSI --content "My voice is the password" --target librispeech_p1089
```
And the sample audio and intermediate results are provided in *SMACK_Examples/success_ivectorCSI_librispeech_p1089.wav*, *SMACK_Examples/success_ivectorCSI_librispeech_p1089_1.wav*, and *SMACK_Examples/success_ivectorCSI_librispeech_p1089.txt* respectively. Similar to attacks against GMM-based models, this attack against ivector-based models can take 8 compute-hour.

(Optional for Extended Testing) Besides our provided pre-trained models, you can also choose to create your own by enrolling new speakers. You will need to manually place speech samples of the speaker to be enrolled under the folder *FAKEBOB/data/enrollment-set*. Then simply run the command as follows:
```sh
$ python3 speaker_enroll.py
```

# Other Notes

The attack against commercial APIs can sometimes fail due to unstable connections. In the cases that APIs constantly return meaningless responses, please first try basic test 3 and see if the queries work well.

There are two potential reasons if the attack results do not reach the specified target (transcription or speaker label). First, SMACK involves genetic search and gradient estimation, which therefore introduces uncertainties in the optimization process. Results can be different even with the same parameters. Second, the parameter settings need to be adjusted for the target. Considering the time and resource consumption, the provided parameters were selected very conservatively. For instance, the adapted genetic algorithm will only run for 10 epochs with a population size of 20, and the gradient estimation will only run for 5 iterations. As such, the entire attack process adds up to only 300 queries to the target model. To adapt the attack to the target, please consider enlarging the maximum iterations and adjusting other parameters used in the attack algorithm. Besides, the limitations of SMACK regarding arbitrary targets can be found in our paper.

# Citation

If you find the platform useful, please cite our work with the following reference:
```
@inproceedings{yu2023smack,
  title={SMACK: Semantically Meaningful Adversarial Audio Attack},
  author={Yu, Zhiyuan and Chang, Yuanhaur and Zhang, Ning and Xiao, Chaowei},
  booktitle={32nd USENIX Security Symposium 2023},
  year={2023}
}
```
