# -*- coding: utf-8 -*-
"""
Created on Sun May  1 21:08:04 2022

@author: richa
"""

import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import sigproc

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None,ftype='linear'):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    assert ftype in ['linear','mel'], "filterbank type incorrect (linear/mel)"
    
    if ftype == 'linear':

        lowmel = lowfreq
        highmel = highfreq
        melpoints = np.linspace(lowmel,highmel,nfilt+2)
        bin = np.floor((nfft+1)*melpoints/samplerate)
        
    elif ftype == 'mel':
        lowmel = hz2mel(lowfreq)
        highmel = hz2mel(highfreq)
        melpoints = np.linspace(lowmel,highmel,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank 

def center_filterbank(nfilt,lowfreq,highfreq,ftype='linear'):
    
    if ftype == 'linear':
        lowmel = lowfreq
        highmel = highfreq
        
    elif ftype == 'mel':
        lowmel = hz2mel(lowfreq)
        highmel = hz2mel(highfreq)
        
    edges = np.linspace(lowmel,highmel,nfilt+2)
    cts = [np.mean(edges[i-2:i]) for i in range(2,len(edges))]
    
    return cts

def fbank(signal,samplerate=16000,winlen=0.032,winstep=0.008,
          nfilt=20,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          winfunc=lambda x:np.ones((x,)),ftype='linear'):
    """Compute Mel-filterbank energy features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    if ftype == 'linear':
        preemph = 0
    
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    energy = np.sum(pspec,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq,ftype=ftype)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def msr_linear(x, fs,
               win_size1:float=.032, win_shift1:float=.008, 
               win_size2:float=.256, win_shift2:float=.064,
               n_fft_factor1:int=1, n_fft_factor2:int=1,
               n_freq_filters=20, n_mod_filters=20,
               low_freq=0, min_cf=0, max_cf=20):
    
    # convert second to number of FFT points (1st)
    nfft1 = np.ceil(n_fft_factor1*win_size1*fs).astype(int)
    # compute linear filterbank energies
    # output shape "(num_frame,num_freq_bin)"
    spec_en = fbank(signal=x,samplerate=fs,winlen=win_size1,
                    winstep=win_shift1,nfft=nfft1,winfunc=np.hamming,
                    lowfreq=low_freq,ftype='linear',nfilt=n_freq_filters)[0]

    # parameters for modulation energy computation
    mfs = int(1/win_shift1) # sampling frequency (modulation)
    mod_num_frame = np.ceil((spec_en.shape[0]/mfs-win_size2)/win_shift2+1).astype(int)
    mod_ens = np.empty((mod_num_frame,n_freq_filters,n_mod_filters))*np.nan
    nfft2 = np.ceil(n_fft_factor2*win_size2*fs).astype(int)
    
    # 2nd FFT along frequency axis
    # output shape "(num_timeframe,num_freq_bin,num_mod_bin)"
    for freq_bin_idx in range(n_freq_filters):
        sig_freq = np.abs(spec_en[:,freq_bin_idx])
        mod_ens[:,freq_bin_idx,: ] = fbank(signal=sig_freq,
                                        samplerate=mfs,
                                        winlen=win_size2,
                                        winstep=win_shift2,
                                        nfft=nfft2,winfunc=np.hamming,
                                        lowfreq=min_cf,highfreq=max_cf,
                                        ftype='linear',nfilt=n_mod_filters)[0]
    
    # if feat is zero, we get problems with log
    mod_ens = np.where(mod_ens == 0,np.finfo(float).eps,mod_ens) 
    return mod_ens


def msr_linear_log(x, fs,
               win_size1:float=.032, win_shift1:float=.008, 
               win_size2:float=.256, win_shift2:float=.064,
               n_fft_factor1:int=1, n_fft_factor2:int=1,
               n_freq_filters=20, n_mod_filters=20,
               low_freq=0, min_cf=0, max_cf=20):
    
    output = msr_linear(x, fs,win_size1, win_shift1, 
                        win_size2, win_shift2,
                        n_fft_factor1, n_fft_factor2,
                        n_freq_filters, n_mod_filters,
                        low_freq, min_cf, max_cf)
    
    return 10*np.log10(output)

# %%

if __name__ == "__main__":
    from scipy import signal
    from scipy.fft import fftshift
    import matplotlib.pyplot as plt
    
    # random seed
    rng = np.random.default_rng()
    
    # parameters to generate a modulated signal
    fs = 16e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 10*np.cos(2*np.pi*0.25*time) #modulation signal with fmod=4Hz
    carrier = amp * np.cos(2*np.pi*1e3*time) * (10+mod)
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + 5*noise
    
    # show log-spectrogram
    f, t, Sxx = signal.spectrogram(x, fs)
    plt.pcolormesh(t, f, 10*np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    # visualize mod filterbank
    filter_mod = get_filterbanks(nfilt=20,nfft=512,samplerate=125,lowfreq=0,highfreq=20)
    plt.pcolormesh(filter_mod)
    
    # center frequencies of mod filterbank
    cts = center_filterbank(nfilt=20,lowfreq=0,highfreq=20)
    print(cts)

    # show average log modulation spectrogram (averaged over time)
    toy_mod = msr_linear_log(x,fs,n_freq_filters=20,n_mod_filters=20)
    plt.figure(figsize=(20,10))
    plt.pcolormesh(np.mean(toy_mod,axis=0).squeeze())
    plt.colorbar()
    plt.yticks(np.arange(0,21,1),np.arange(0,8001,400))
    plt.xticks(np.arange(.5,20.5,1),np.round(cts,2),rotation=-20)
    plt.xlabel('Modulation frequency')
    plt.ylabel('Acoustic frequency')
    
    
    # # Compute modulation spectrogram with STFT
    # f = strfft_modulation_spectrogram(x, fs, .032*fs, 0.008*fs)
    # plot_modulation_spectrogram_data(f, c_map='jet')
    # plt.xlim([0,20])    
