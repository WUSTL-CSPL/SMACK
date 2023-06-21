# 1. Clone the repositories

FAKEBOB supports UNIX-like systems, and the following instructions are tested on Ubuntu 20.04.
Clone FAKEBOB to the local directory:
```
git clone 'https://github.com/FAKEBOB-adversarial-attack/FAKEBOB'
```

Clone Kaldi inside FAKEBOB:
```
cd FAKEBOB
git clone 'https://github.com/kaldi-asr/kaldi'
```

# 2 Add paths to FAKEBOB and Kaldi

Inside "FAKEBOB/path_cmd.sh", change "KALDI_ROOT" and "FAKEBOB_PATH" to the global paths of Kaldi and Fakebob respectively.
For example:
```
export KALDI_ROOT=".../FAKEBOB/kaldi" # change to your own KAIDI_ROOT
FAKEBOB_PATH=".../FAKEBOB" # change to your FAKEBOB PATH
```

# 3 Move gmm-global-est-map.cc to Kaldi, add it to Makefile

Move file "gmm-global-est-map.cc" under Kaldi withe the following command:
```
mv gmm-global-est-map.cc kaldi/src/gmmbin
```

Open kaldi/src/gmmbin/Makefile, add "gmm-global-est-map" (the file just moved, without extension ".cc") to be compiled later:
For example:
```
BINFILES = ...
...
gmm-global-gselect-to-post gmm-global-est-lvtln-trans gmm-init-biphone gmm-global-est-map
```

# 4 Compile Kaldi/tools

Go to kaldi/tools, run the following dependency checker shell code:
```
cd FAKEBOB/kaldi/tools
./extras/check_dependencies.sh
```
The checker should output missing dependencies, which may include:
*make automake autoconf patch grep*
*bzip2 gzip unzip wget git sox*
*zlib gfortran libtoolize svn awk*
Install these dependencies with sudo apt-get, or follow the output steps to install themm manually.
To support the compile, g++ >= 4.8 is also required.

Once all dependencies are installed, compile them by running "make" under kaldi/tools:
```
make -j 8
```
This may take up to 5 minutes depending on the system spec. Set the "-j" flag to the number of cores to run in parallel. If all dependencies are installed correctly, the program should output:
```
All done OK.
```

# 5 Compile Kaldi/src

Go to kaldi/src, run the following command to compile:
```
cd FAKEBOB/kaldi/src
./configure --shared
make -j clean depend; make -j 8
```
This may take up to 10 minutes depending on the system spec. Set the "-j" flag to the number of cores to run in parallel.

# 6 Download Pre-trained Models

Download pretrained models at https://drive.google.com/open?id=1T_hx9Pqopk-rlmiSrBWdXjl825wjBQVF
(MD5 Checksum: a61fd0f2a470eb6f9bdba11f5b71a123)
After moving it under FAKEBOB/, unzip with the following command:
```
cd FAKEBOB
tar -xzvf pre-models.tgz
rm pre-models.tgz
```

# 7 Copy paths to ~/.bashrc

Open the ~/.bashrc (or ~/.zshrc depending on the shell used), and copy the entire content of ...FAKEBOB/path_cmd.sh to the very bottom.
Make sure that "KALDI_ROOT" and "FAKEBOB_PATH" varaibles are set to the correct global paths, see step 2.
Finally, run the following command:
```
source ~/.bashrc
```
If a conda environment is used, activate it again. 

# 8 Load data and custom scripts

Create the directory structure for speaker verification with the following command:
```
cd ...FAKEBOB/
mkdir data
mkdir data/enrollment-set 
mkdir data/z-norm
mkdir data/test-set
```

The data/enrollment-set is used to place wavs to be enrolled. The wav audio file placed here should be follow the naming format "<speaker_name>-<audio_name>.wav", lead with speaker_name and separate with the first '-'.
For example, the following structure would enroll model for label "speaker_1" and "speaker_2":
-->...FAKEBOB/data/enrollment-set/
------> speaker_1-audio_001.wav
------> speaker_2-audio_001.wav

The background audio set, "FAKEBOB/data/z-norm/", is a directory containing wav audio files, used as the environment where enrollment set is compared to, therefore enroll. The directory structure can be anything, as the script recursively collect all wav audio files from its subdirectories. 

Copy and replace the following files under "FAKEBOB" with custom scripts of the same name. The custom scripts can be found in the *SMACK_Supplementary_Files/FAKEBOB_File_Replace* (see README.md for google drive download link) folder provided in the root folder.
-->...FAKEBOB/
------> ivector_PLDA_SV.py (new)
------> gmm_ubm_SV.py (new)
------> ivector_PLDA_CSI.py (new)
------> gmm_ubm_CSI.py (new)
------> ivector_PLDA_OSI.py (new)
------> gmm_ubm_OSI.py (new)

Copy and place speaker_enroll.py, speaker_sv.py, speaker_close_set_identification.py, speaker_open_set_identification.py parallel to FAKEBOB:
-->...FAKEBOB
--> speaker_enroll.py
--> speaker_sv.py
--> speaker_csi.py
--> speaker_osi.py

# 9 Copy custom enroll script

To enroll speaker files in FAKEBOB/data/enrollment-set/, run the following. Note that the controller scripts are parallel to FAKEBOB
```
python speaker_enroll.py
```
The runtime of building speaker model heavily depend on the number of speakers and the size of the background audio set. Note that the script builds speaker models in parallesl with 8 cores, to change the number of cores (thereby the runtime), change 'n_jobs' in the script.

# 10 Copy custom speaker verification script

To execute speaker verification, run with the following:
```
python custom_sv.py <probe_wav> <probe_label> 
```
Similarly, changing 'n_jobs' in the script can modify the number of thread running as well as runtime. Modify "benign_wavs_rootdir" to reflect the benign audio of the probe_label, this directory should only contain wavs in the aforementioned wav format.

The script outputs the score of the probe wav file, the estimated threshold for it to pass, and the final decision (accept/reject). 

# 11 Copy custom speaker close set identification script

To execute speaker close set identification, run with the following:
```
python speaker_close_set_identification.py 
```
Similarly, changing 'n_jobs' in the script can modify the number of thread running as well as runtime. Modify "probe_wav" and "target_label" to reflect wav audio file to be tested and its target label.

# 12 Copy custom speaker open set identification script

To execute speaker open set identification, run with the following:
```
python speaker_open_set_identification.py 
```
Similarly, changing 'n_jobs' in the script can modify the number of thread running as well as runtime. Modify "probe_wav" and "target_label" to reflect wav audio file to be tested and its target label. Additionally, modify "benign_wavs_dir" to point to the benign dataset of wavs, used to estimate the threshold. 

# 13 Remember to edit ~/.bashc when removing FAKEBOB 

When removing FAKEBOB, it is suggested to remove the added sections in ~/.bashc to avoid bash profile error, as the path to FAKEBOB no longer exists. Failure to do so may cause bash crashing.

