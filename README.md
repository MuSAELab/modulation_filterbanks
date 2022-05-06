# Quantized spectrogram/modulation spectrogram

The script covers the following functionalities:
1) compute spectrogram energies (original and log) using customized filterbanks
2) compute modulation spectrogram energies (original and log) using customized filterbanks
3) compute quantized modulation spectrogram features

It also provides short demos for each step of the computation, from the original spectrogram to filtered spectrogram, and filtered modulation spectrogram. <br />

# Demos:<br />
1.Original spectrogram of a modulated signal (modulated by f_m = 0.25Hz):<br />
<img src="./docs/test_spec_og.png" width="400" height="300"><br />
2.Original modulation spectrogram of the signal:<br />
<img src="./docs/test_modspec.png" width="400" height="300"><br />
3.Quantized (20\*20) version of modulation spectrogram (with linear filterbank):<br />
<img src="./docs/test_fbank.png" width="400" height="300">
