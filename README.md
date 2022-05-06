# Quantized spectrogram/modulation spectrogram

The script covers the following functionalities:
1) compute spectrogram energies (original and log) using customized filterbanks
2) compute modulation spectrogram energies (original and log) using customized filterbanks
3) compute quantized modulation spectrogram features

It also provides short demos for each step of the computation, from the original spectrogram to filtered spectrogram, and filtered modulation spectrogram. <br />

# Demo1: High SNR setting<br />
- Spectrogram of a modulated signal (modulated by f_m = 1Hz):<br />
<img src="./docs/demo1_spec.png" width="400" height="300"><br />
- Quantized (20\*20) version of modulation spectrogram (using linear filterbank):<br />
*Notice the higher energy at f_m = 1Hz<br />
<img src="./docs/demo1_mod.png" width="400" height="300"><br />

# Demo2: Low SNR setting<br />
- Spectrogram of a modulated signal (modulated by f_m = 8Hz):<br />
<img src="./docs/demo2_spec.png" width="400" height="300"><br />
- Quantized version of modulation spectrogram (using linear filterbank):<br />
*Notice the higher energy at f_m = 8Hz<br />
<img src="./docs/demo2_mod.png" width="400" height="300"><br />

# Feature extraction pipeline
We firstly compute the quantized modulation spectrogram, then extract eight different descriptors from each frequency and modulation frequency bin. Steps are depiected in the following graph.<br />
<img src="./docs/msf.png" ><br />

# Related articles using the module
- Y. Zhu and T. H. Falk, "Fusion of Modulation Spectral and Spectral Features with Symptom Metadata for Improved Speech-Based Covid-19 Detection," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 8997-9001, doi: 10.1109/ICASSP43922.2022.9746471.

# Related resources
Part of the script is modified from 'python_speech_features' module and 'modulation spectrogram' module. Links are provided below:
- https://github.com/jameslyons/python_speech_features
- https://github.com/MuSAELab/amplitude-modulation-analysis-module
