# Quantized spectrogram/modulation spectrogram

The script covers the following functionalities:
1) compute spectrogram energies (original and log) using customized filterbanks
2) compute modulation spectrogram energies (original and log) using customized filterbanks
3) compute quantized modulation spectrogram features

It also provides short demos for each step of the computation, from the original spectrogram to filtered spectrogram, and filtered modulation spectrogram. <br />

# Demo1:<br />
- Spectrogram of a modulated signal (modulated by f_m = 0.25Hz):<br />
<img src="./docs/test_spec_og.png" width="400" height="300"><br />
- Modulation spectrogram of the signal (averaged over time):<br />
<img src="./docs/test_modspec.png" width="400" height="300"><br />
- Quantized (20\*20) version of modulation spectrogram (using linear filterbank):<br />
*Notice the higher energy at f_m < 1Hz
<img src="./docs/test_fbank.png" width="400" height="300"><br />

# Demo2: High SNR setting<br />
- Spectrogram of a modulated signal (modulated by f_m = 8Hz):<br />
<img src="./docs/demo2_spec.png" width="400" height="300"><br />
- Quantized version of modulation spectrogram (using linear filterbank):<br />
*Notice the higher energy at f_m = 8Hz
<img src="./docs/demo2_mod.png" width="400" height="300"><br />
