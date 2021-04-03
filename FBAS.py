#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:48:51 2021

@author: Mrinmoy Bhattacharjee, Ph.D. Scholar, EEE Dept., IIT Guwahati, Assam
"""

import datetime
import os
import lib.preprocessing as preproc
from scipy.io import wavfile
import librosa
import numpy as np
from fbas_decompose import fbas
import sys
import matplotlib.pyplot as plt




def stft2wav(PARAMS, Stft, fs):
    frameShift = int((PARAMS['Ts']*fs)/1000) # Frame shift in number of samples    
    wav = librosa.core.istft(Stft, hop_length=frameShift, center=False)
    wav = wav + PARAMS['preemphasis_coef']*np.roll(wav,-1)
    return wav


def normalize(wav, nBits):
    wav = wav - np.mean(wav)
    wav = wav / np.max(np.abs(wav))
    if nBits==32:
        # For 32-bit PCM encoding
        wav *= 2**31
        wav = (wav).astype(np.int32)
    elif nBits==16:
        # For 16-bit PCM encoding
        wav *= 2**15
        wav = (wav).astype(np.int16)
    return wav


def save_wav(PARAMS, Spec, FgSpec, BgSpec, fs):
    Xin_fg = stft2wav(PARAMS, FgSpec, fs)
    Xin_fg = Xin_fg[frameShift:-frameShift]
    Xin_fg = normalize(Xin_fg, 16)
    print('Xin_fg: ', np.min(Xin_fg), np.max(Xin_fg))

    Xin_bg = stft2wav(PARAMS, BgSpec, fs)
    Xin_bg = Xin_bg[frameShift:-frameShift]
    Xin_bg = normalize(Xin_bg, 16)
    print('Xin_bg: ', np.min(Xin_bg), np.max(Xin_bg))

    fName_fg = PARAMS['opDir'] + '/' + PARAMS['audio_path'].split('/')[-1].split('.')[0]+'.wav'
    wavfile.write(fName_fg, fs, Xin_fg)
    print('Foreground file saved: ', fName_fg)        
    fName_bg = PARAMS['opDir'] + '/' + PARAMS['audio_path'].split('/')[-1].split('.')[0]+'_bg.wav'
    wavfile.write(fName_bg, fs, Xin_bg)
    print('Background file saved: ', fName_bg)
    


if __name__ == '__main__':
    PARAMS = {
        'today': datetime.datetime.now().strftime("%Y-%m-%d"),
        'audio_path': '',
        'save_path': './',
        'Tw': 25, # in miliseconds
        'Ts': 10, # in miliseconds
        'silThresh': 0,
        'sr': 16000,
        'preemphasis_coef': 0,
        'save_flag': True,
        'req_pks': 0.99,
        'epsilon': 0.05,
        'plot_flag': False,
        }
    
    if sys.argv[1].lower().endswith(('.wav', '.mp3')):
        PARAMS['audio_path'] = sys.argv[1]
    else:
        print('Not a file or not a supported file type')
        sys.exit(0)
    if sys.argv[2]:
        PARAMS['save_path'] = sys.argv[2]

    PARAMS['opDir'] = PARAMS['save_path'] + '/FBAS/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
    opFile = PARAMS['opDir'] + '/Performance.csv'
        
    print('fName: ', PARAMS['audio_path'])
    Xin, fs = preproc.load_and_preprocess_signal(PARAMS, PARAMS['audio_path'])
    
    frameSize = int((PARAMS['Tw']*fs)/1000) # Frame size in number of samples
    frameShift = int((PARAMS['Ts']*fs)/1000) # Frame shift in number of samples    
    Spec = librosa.core.spectrum.stft(y=Xin, n_fft=frameSize, hop_length=frameShift, window=np.hanning(frameSize), center=False)
    
    FBAS = fbas(S=Spec, req_pks=PARAMS['req_pks'], epsilon=PARAMS['epsilon'])
    # FBAS = fbas(S=Spec, req_pks=0.9, k=0.5, alpha=0.9, epsilon=0.00, beta=1e-5)
    FgSpec, BgSpec = FBAS.decompose()
    
    save_wav(PARAMS, Spec, FgSpec, BgSpec, fs)
    
    if PARAMS['plot_flag']:
        nFrames = -1
        plt.figure()
        subplot_num = 510
    
        subplot_num += 1
        plt.subplot(subplot_num)
        S_temp = None
        S_temp = np.abs(Spec)[::-1,:nFrames]
        S_temp = librosa.core.power_to_db(S_temp)
        plt.imshow(S_temp, aspect='auto')
    
        subplot_num += 1
        plt.subplot(subplot_num)
        plt.imshow(FBAS.FgMask, aspect='auto')
        plt.plot(FBAS.W_C_fg, 'k')
        plt.xlim([0, len(FBAS.W_C_fg)])
        plt.ylim([0, 1])
        plt.legend(['Foreground'])
    
        subplot_num += 1
        plt.subplot(subplot_num)
        S_temp = None
        S_temp = np.abs(FgSpec)[::-1,:nFrames]
        S_temp = librosa.core.power_to_db(S_temp)
        plt.imshow(S_temp, aspect='auto')
    
        subplot_num += 1
        plt.subplot(subplot_num)
        plt.imshow(FBAS.BgMask, aspect='auto')
        plt.plot(FBAS.W_C_bg, 'k')
        plt.xlim([0, len(FBAS.W_C_bg)])
        plt.ylim([0, 1])
        plt.legend(['Background'])
    
        subplot_num += 1
        plt.subplot(subplot_num)
        S_temp = None
        S_temp = np.abs(BgSpec)[::-1,:nFrames]
        S_temp = librosa.core.power_to_db(S_temp)
        plt.imshow(S_temp, aspect='auto')
        
        plt.show()
