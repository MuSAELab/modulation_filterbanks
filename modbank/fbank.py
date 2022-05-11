# -*- coding: utf-8 -*-
"""
Created on Sun May  1 21:08:04 2022

@author: richa
"""

import numpy as np
import matplotlib.pyplot as plt
import decimal
import math
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

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def calc_num_frame(sig_len:int,frame_len,frame_shift):

    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_shift))
    if sig_len <= frame_len:
        num_frame = 1
    else:
        num_frame = 1 + int(math.ceil((1.0*sig_len - frame_len) / frame_step))
    
    return num_frame

def msr_filtered(x, fs,
               win_size1:float=.032, win_shift1:float=.008, 
               win_size2:float=.256, win_shift2:float=.064,
               n_fft_factor1:int=1, n_fft_factor2:int=1,
               n_freq_filters=20, n_mod_filters=20,
               low_freq=0, high_freq=None, min_cf=0, max_cf=20,
               ftype1:str='linear',ftype2:str='linear'):
    
    assert ftype1 and ftype2 in ['linear','mel'], "unsupported filterbank type"
    
    # convert second to number of FFT points (1st)
    nfft1 = np.ceil(n_fft_factor1*win_size1*fs).astype(int)
    # compute linear filterbank energies
    # output shape "(num_frame,num_freq_bin)"
    spec_en = fbank(signal=x,samplerate=fs,winlen=win_size1,
                    winstep=win_shift1,nfft=nfft1,winfunc=np.hamming,
                    lowfreq=low_freq,highfreq=high_freq,ftype=ftype1,nfilt=n_freq_filters)[0]
    
    spec_en = np.where(spec_en == 0,np.finfo(float).eps,spec_en) 
    # parameters for modulation energy computation
    mfs = int(1/win_shift1) # sampling frequency (modulation)
    mod_num_frame = calc_num_frame(spec_en.shape[0],win_size2*mfs,win_shift2*mfs)
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
                                        ftype=ftype2,nfilt=n_mod_filters)[0]
    
    # if energy value is zero, we get problems with log
    mod_ens = np.where(mod_ens == 0,np.finfo(float).eps,mod_ens) 
    return mod_ens


def msr_filtered_log(x, fs,
               win_size1:float=.032, win_shift1:float=.008, 
               win_size2:float=.256, win_shift2:float=.064,
               n_fft_factor1:int=1, n_fft_factor2:int=1,
               n_freq_filters=20, n_mod_filters=20,
               low_freq=0, high_freq=None,min_cf=0, max_cf=20,
               ftype1:str='linear',ftype2:str='linear'):
    
    output = msr_filtered(x, fs,win_size1, win_shift1, 
                        win_size2, win_shift2,
                        n_fft_factor1, n_fft_factor2,
                        n_freq_filters, n_mod_filters,
                        low_freq, high_freq, min_cf, max_cf,
                        ftype1,ftype2)
    
    return 10*np.log10(output)


def get_subband_descriptors(psd, freq_range):
    #Initialize a matrix to store features
    ft=np.empty((8))*np.nan
    lo,hi = freq_range[0], freq_range[-1] #Smallest and largest value of freq_range
    
    #Centroid
    ft[0] = np.sum(psd*freq_range)/np.sum(psd)
    #Entropy
    ft[1]=-np.sum(psd*np.log(psd))/np.log(hi-lo)
    #Spread
    ft[2]=np.sqrt(np.sum(np.square(freq_range-ft[0])*psd)/np.sum(psd))
    #skewness
    ft[3]=np.sum(np.power(freq_range-ft[0],3)*psd)/(np.sum(psd)*ft[2]**3)
    #kurtosis
    ft[4]=np.sum(np.power(freq_range-ft[0],4)*psd)/(np.sum(psd)*ft[2]**4)
    #flatness
    arth_mn=np.mean(psd)/(hi-lo)
    geo_mn=np.power(np.exp(np.sum(np.log(psd))),(1/(hi-lo)))
    ft[5]=geo_mn/arth_mn
    #crest
    ft[6]=np.max(psd)/(np.sum(psd)/(hi-lo))
    #flux
    ft[7]=np.sum(np.abs(np.diff(psd)))
    
    return ft


def get_msf(x, fs,
            n_freq_filters=20, n_mod_filters=20,
            low_freq=0, high_freq=None,
            min_cf=0, max_cf=20,
            ftype1:str='linear',ftype2:str='linear'):
    
    assert x.ndim == 2, "required input size is 2D"
    assert x.shape[0] == n_freq_filters and x.shape[1] == n_mod_filters,\
        "incorrect input shape"

    if high_freq is None:
        high_freq = fs//2
        
    cfs1 = center_filterbank(nfilt=n_freq_filters,
                             lowfreq=low_freq,
                             highfreq=high_freq,
                             ftype=ftype1)
    
    cfs2 = center_filterbank(nfilt=n_mod_filters,
                             lowfreq=min_cf,
                             highfreq=max_cf,
                             ftype=ftype2)
    
    num_bins = n_freq_filters + n_mod_filters
    feat = np.empty((num_bins,8))*np.nan
    for i in range(num_bins):
        if i < n_freq_filters:
            feat[i,:] = get_subband_descriptors(x[i,:],cfs2)
        else:
            feat[i,:] = get_subband_descriptors(x[:,i-n_mod_filters],cfs1)
    
    assert np.nan not in feat, "NaN in returned features"
    return feat


def get_msf_3d(x, fs,
            n_freq_filters=20, n_mod_filters=20,
            low_freq=0, high_freq=None,
            min_cf=0, max_cf=20,
            ftype1:str='linear',ftype2:str='linear'):
    
    assert x.ndim == 3,"required input size is 3D"
    assert x.shape[1] == n_freq_filters and x.shape[2] == n_mod_filters,\
        "incorrect input shape"
        
    if high_freq is None:
        high_freq = fs//2
    
    num_timeframe = x.shape[0]
    num_bins = n_freq_filters + n_mod_filters
    feat_3d = np.empty((num_timeframe,num_bins,8))*np.nan
    for i in range(num_timeframe):
        feat_3d[i,::] = get_msf(x[i,::], fs,
                                n_freq_filters, n_mod_filters,
                                low_freq, high_freq,
                                min_cf, max_cf,
                                ftype1,ftype2)
        
    assert np.nan not in feat_3d, "NaN in returned features"
    
    return feat_3d


def msf_all(x, fs,
            win_size1:float=.032, win_shift1:float=.008, 
            win_size2:float=.256, win_shift2:float=.064,
            n_fft_factor1:int=1, n_fft_factor2:int=1,
            n_freq_filters=20, n_mod_filters=20,
            low_freq=0, high_freq=None,
            min_cf=0, max_cf=20,
            ftype1:str='linear',ftype2:str='linear'):
    
    feat_energy = msr_filtered(x, fs,
                               win_size1, win_shift1, 
                               win_size2, win_shift2,
                               n_fft_factor1, n_fft_factor2,
                               n_freq_filters, n_mod_filters,
                               low_freq, high_freq, 
                               min_cf, max_cf,
                               ftype1,ftype2)
    
    feat_msf = get_msf_3d(feat_energy, fs,
                          n_freq_filters, n_mod_filters,
                          low_freq, high_freq,
                          min_cf, max_cf,
                          ftype1,ftype2)
    
    assert feat_energy.shape[0] == feat_msf.shape[0], "inconsistent time dimension"
    
    # flatten both into 2d matrix "num_frame * num_features"
    feat_energy = feat_energy.reshape((feat_energy.shape[0],-1))
    feat_msf = feat_msf.reshape((feat_msf.shape[0],-1))
    
    feat_all = np.concatenate((10*np.log10(feat_energy),feat_msf),axis=1)
    
    return feat_all 

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
    mod = 10*np.cos(2*np.pi*1*time) #modulation signal with fmod=4Hz
    carrier = amp * np.cos(2*np.pi*1e3*time) * (10+mod)
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + 0.1*noise
    
    # show log-spectrogram
    f, t, Sxx = signal.spectrogram(x, fs)
    plt.pcolormesh(t, f, 10*np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    # visualize mod filterbank
    filter_mod = get_filterbanks(nfilt=20,nfft=512,samplerate=125,lowfreq=0,highfreq=20)
    plt.pcolormesh(filter_mod)
    
    # center frequencies of freq filterbank
    freq_cts = center_filterbank(nfilt=20,lowfreq=0,highfreq=8000)
    
    # center frequencies of mod filterbank
    cts = center_filterbank(nfilt=20,lowfreq=0,highfreq=20)

    # show average log modulation spectrogram (averaged over time)
    toy_mod = msr_filtered(x,fs,n_freq_filters=20,n_mod_filters=20,min_cf=0,max_cf=20,ftype1='linear',ftype2='linear')
    plt.figure(figsize=(20,10))
    plt.pcolormesh(np.mean(toy_mod,axis=0).squeeze())
    plt.colorbar()
    plt.yticks(np.arange(0.5,20.5,1),np.around(freq_cts))
    plt.xticks(np.arange(.5,20.5,1),np.round(cts,2),rotation=-20)
    plt.xlabel('Modulation frequency')
    plt.ylabel('Acoustic frequency')
    
    # calculate modulation spectrogram descriptors
    toy_feat = get_msf_3d(toy_mod,fs)
    
    # calculate all modulation spectrogram related features (log energies+descriptors)
    toy_feat_all = msf_all(x,fs)  
