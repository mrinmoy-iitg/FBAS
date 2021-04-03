#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:07:02 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

"""

import numpy as np
import scipy.signal
from joblib import Parallel, delayed
import multiprocessing
import scipy.stats
from kneed import KneeLocator
import librosa




class fbas():    

    S = np.empty([])
    req_pks = 1
    Thresh = 0
    nPks = 0
    nFrames = 0
    peak_repeat_count = 0
    SPT_stft = np.empty([])
    mask = False
    n_jobs = 1
    T=0
    k=0
    epsilon=0
    beta = 0
    peak_widths = []
    

    def __init__(self, S=None, req_pks=0.9, n_jobs=1, mask=False, k=0.5, alpha=0.5, epsilon=0.01, beta=1e-5, **kwargs):
        """Spectral Peak Tracking foreground background source separation (FBSS).

        This implementation is based upon the algorithm described by [#]_ .
    
        .. [#] M. Bhattacharjee, S. R. M. Prasanna and P. Guha, 
            "Speech/Music Classification Using Features From Spectral Peaks," 
            in IEEE/ACM Transactions on Audio, Speech, and Language Processing, 
            vol. 28, pp. 1549-1559, 2020, doi: 10.1109/TASLP.2020.2993152.
            
        Parameters
        ----------
        S : np.ndarray [shape=(d, n)]
            input complex spectrogram. Real spectrogram also works, but the 
            quality of the generated foreground and backround audio is 
            degraded and sounds artificial
    
        req_pks : int or float
            Number of peaks to be picked by the Spectral Peak Tracking 
            algorithm [1]
            If an integer greater than 1, then a fixed number of peaks 
            are picked from each frame spectra (given by req_pks). Frames
            with lesses peaks are appended with the highest-frequency 
            peak present in it to maintain cardinality.
            If a float, then a fraction of peaks present in a frame 
            spectra are picked. All frames may have separate number of 
            peaks in this case.
        
        n_jobs : int
            Number of parallel processes to spawn
    
        mask : bool
            Return the masking matrices instead of components.
    
            Masking matrices contain non-negative real values that
            can be used to measure the assignment of energy from ``S``
            into harmonic or percussive components.
    
            Components can be recovered by multiplying ``S * mask_F``
            or ``S * mask_B``.
            
        k : float
            standard deviation factor used for frame-weight generation
        
        epsilon: float
            threshold used for selecting foreground frames
        
        beta : float
            energy scaling factor used for background estimation
            
        **kwargs: 
            peak_widths: list
                peak widths to be used when selecting peaks using continuous
                wavelet transform
        
        Attributes
        ----------
        FgMask : ndarray [shape=(d, n)]
            Foreground mask
        
        BgMask : ndarray [shape=(d, n)]
            Background mask
            
        W_C_fg : ndarray [shape=(1, n)]
            Frame-weights for foreground decomposition
            
        W_C_bg : ndarray [shape=(1, n)]
            Frame-weights for background decomposition
            
        
        Returns
        -------
        FgSpec : ndarray [shape=(d, n)]
            foreground spectrogram component
    
        BgSpec : ndarray [shape=(d, n)]
            background spectrogram component
        
        """
        self.S = S
        self.req_pks = req_pks
        self.n_jobs = n_jobs
        self.mask = mask
        self.k=k
        self.alpha = alpha
        print(self.alpha)
        self.epsilon=epsilon
        self.beta = beta
        self.fg_thresh_type = 'hi' # 'lo', 'hi', 'mean'
        self.bg_thresh_type = 'mean' # 'lo', 'hi', 'mean'
        if 'peak_widths' in kwargs.keys():
            self.peak_widths = kwargs['peak_widths']
        else:
            self.peak_widths = [1]
        self.hist = []
        self.bin_centers = []
        self.hist_cumsum = []
        self.threshold_hi = 1e10
        self.threshold_lo = 1e-10
        self.fg_mask_hi = None
        self.fg_mask_lo = None
        self.fg_mask_mean = None
        self.bg_mask_hi = None
        self.bg_mask_lo = None
        self.bg_mask_mean = None
            
            
        
    def decompose(self):        
        PkLoc, PkVal = self.computeSPT(np.abs(self.S))
        self.FgMask, self.W_C_fg = self.get_foreground(np.abs(self.S), PkLoc)
        self.BgMask, self.W_C_bg = self.get_background(np.abs(self.S), PkLoc)

        self.compute_suppression_thresholds(np.abs(self.S))
        if self.fg_thresh_type=='lo':
            Fg_S = np.multiply(self.S,self.fg_mask_lo)
        elif self.fg_thresh_type=='hi':
            Fg_S = np.multiply(self.S,self.fg_mask_hi)
        elif self.fg_thresh_type=='mean':
            Fg_S = np.multiply(self.S,self.fg_mask_mean)

        if self.mask:
            return self.FgMask, self.BgMask
        else:
            ''' Computing Foreground spectrogram '''
            FgSpec = np.multiply(Fg_S, self.FgMask)
    
            ''' Computing Background spectrogram '''
            BgSpec = np.multiply(self.S, self.BgMask)

            return FgSpec, BgSpec



    def compute_suppression_thresholds(self, S):
        S = librosa.core.power_to_db(S)
        S_vals = S.flatten()
        self.hist, bin_edges = np.histogram(S_vals, bins=100, density=True)
        self.bin_centers = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
        self.hist_cumsum = np.cumsum(self.hist)
        
        idx_lo = np.squeeze(np.where(self.hist_cumsum<np.median(self.hist_cumsum))).tolist()
        kneedle_lo = KneeLocator(self.bin_centers[idx_lo], self.hist_cumsum[idx_lo], S=1.0, curve='convex', direction='increasing')
        knee_idx_lo = np.squeeze(np.where(self.hist_cumsum==kneedle_lo.knee_y))
        if np.size(knee_idx_lo)>1:
            knee_idx_lo = knee_idx_lo[0]
        self.threshold_lo = self.bin_centers[knee_idx_lo]
        self.fg_mask_lo = S>self.threshold_lo
        self.bg_mask_lo = np.invert(S>self.threshold_lo)
        
        idx_hi = np.squeeze(np.where(self.hist_cumsum>np.median(self.hist_cumsum))).tolist()
        kneedle_hi = KneeLocator(self.bin_centers[idx_hi], self.hist_cumsum[idx_hi], S=1.0, curve='concave', direction='increasing')
        knee_idx_hi = np.squeeze(np.where(self.hist_cumsum==kneedle_hi.knee_y))
        if np.size(knee_idx_hi)>1:
            knee_idx_hi = knee_idx_hi[0]
        self.threshold_hi = self.bin_centers[knee_idx_hi]
        self.fg_mask_hi = S>self.threshold_hi
        self.bg_mask_hi = np.invert(S>self.threshold_hi)

        self.threshold_mean = self.alpha*self.threshold_hi + (1-self.alpha)*self.threshold_lo
        self.fg_mask_mean = S>self.threshold_mean
        self.bg_mask_mean = np.invert(S>self.threshold_mean)
    

    
    def peak_finding(self, frmFt, nPks, Thresh):
        peak_repeat = False
        if self.req_pks>1:
            loc, prop = scipy.signal.find_peaks(frmFt)
            hgt = np.array(frmFt[loc])    
            if len(hgt.tolist())==0:
                return
            # Repeating the highest frequency peak to keep same cardinality in every frame
            if(len(hgt) < nPks):
                hgt = np.append(hgt, np.ones([nPks-len(hgt),1],int)*hgt[len(hgt)-1])
                loc = np.append(loc, np.ones([nPks-len(loc),1],int)*loc[len(loc)-1])
                peak_repeat = True
        
        elif self.req_pks<=1:
            loc, prop = scipy.signal.find_peaks(frmFt)
            all_peak_ener = np.sum(np.power(frmFt[loc], 2))
            hgt = np.array(frmFt[loc])
            val = np.sort(hgt)
            val = val[::-1] # arrange in descending order
            peak_ener = 0
            nPks = 0
            peak_ener = np.sum(np.power(val[:nPks],2))
            while peak_ener<Thresh*all_peak_ener:
                nPks += 1
                if nPks==len(val):
                    break
                peak_ener = np.sum(np.power(val[:nPks],2))
        hgtIdx = np.argsort(hgt)[::-1][:nPks]# Getting the top peaks in sorted order
        pkIdx = np.sort(hgtIdx)    
        amplitudes = np.array(hgt[pkIdx], ndmin=2)
        freq_bins = np.array(loc[pkIdx], ndmin=2)    

        return amplitudes, freq_bins, peak_repeat
    
    
    
    
    def computeSPT(self, S):    
        if self.req_pks>1:
            nPks = int(self.req_pks)
            PkLoc = np.array
            PkVal = np.array
        else:
            Thresh = self.req_pks
            nPks = 0
            PkLoc = np.zeros(S.shape)
            PkVal = np.zeros(S.shape)
        nFrames = S.shape[1]
        peak_repeat_count = 0
        SPT_stft = np.zeros(S.shape)
        if self.n_jobs==1:
            '''
            Sequential computation
            '''
            for k in range(nFrames):       
                frmFt = S[:,k]
                amplitudes, freq_bins, peak_repeat = self.peak_finding(frmFt, nPks, Thresh)
                peak_repeat_count += int(peak_repeat)
                SPT_stft[freq_bins, k] = amplitudes            
                if self.req_pks>1:
                    if(np.size(PkVal)==1):
                        PkVal = np.array(amplitudes, ndmin=2).T
                        PkLoc = np.array(freq_bins, ndmin=2).T
                    else:
                        PkVal = np.append(PkVal, np.array(amplitudes, ndmin=2).T, 1)
                        PkLoc = np.append(PkLoc, np.array(freq_bins, ndmin=2).T, 1)
                else:
                    PkVal[:np.shape(amplitudes)[1], k] = amplitudes[0,:]
                    PkLoc[:np.shape(freq_bins)[1], k] = freq_bins[0,:]

        elif self.n_jobs>1:
            '''
            Parallel computation
            '''
            num_cores = multiprocessing.cpu_count()-1
            num_cores = int(np.min([self.n_jobs, num_cores]))
            if self.req_pks<1:
                self.req_pks = int(self.req_pks*np.shape(S)[0])
                nPks = self.req_pks
            frmFt = S[:,k]
            peaks = Parallel(n_jobs=self.num_cores)(delayed(self.peak_finding)(frmFt, nPks, Thresh) for k in range(nFrames))
            PkVal, PkLoc, peak_repeat = zip(*peaks)
            peak_repeat_count = np.sum(peak_repeat)
            PkVal = np.array(np.squeeze(PkVal), ndmin=2).T
            PkLoc = np.array(np.squeeze(PkLoc), ndmin=2).T
            for i in range(np.shape(SPT_stft)[1]):
                SPT_stft[PkLoc[:,i],i] = PkVal[:,i]
        
            if self.req_pks==1:
                PkVal = np.transpose(PkVal)
                PkLoc = np.transpose(PkLoc)
            
        return PkLoc, PkVal
            
    
    
    def get_framewise_correlation(self, S):
        corr = np.empty([])
        corr_valid = np.zeros(np.shape(S)[1])
        for i in range(1,np.shape(S)[1]-1):
            corr_temp = np.correlate(S[:,i-1], S[:,i], 'same')
            loc, prop = scipy.signal.find_peaks(corr_temp)
            corr_val = np.sort(corr_temp[loc])[::-1]
            corr_valid[i] = corr_val[1]
            if np.size(corr)<=1:
                corr = np.array(corr_temp, ndmin=2).T
            else:
                corr = np.append(corr, np.array(corr_temp, ndmin=2).T, axis=1)            
        return corr, corr_valid
    


    def get_framewise_energy(self, S):
        energy = np.zeros(np.shape(S)[1])
        for i in range(np.shape(S)[1]):
            energy[i] = np.sum(np.power(S[:,i],2))
        return energy
    
    
    
    def get_foreground(self, S, PkLoc):
        self.corr, W_C = self.get_framewise_correlation(S)
        W_C = W_C/np.max(W_C)    

        FgMask = np.zeros(np.shape(S))
        self.fg_idx = W_C>self.epsilon
        for i in range(np.shape(PkLoc)[1]):
            nPks = np.sum(PkLoc[:,i]>0)

            if nPks==0:
                self.fg_idx[i] = False
                continue
            if W_C[i]>self.epsilon:
                n_fft = np.shape(S)[0]
                for j in range(nPks):
                    hi = None
                    lo = None
                    loc = int(PkLoc[j,i])
                    hi = loc+1
                    if hi<n_fft-1:
                        while S[hi,i]>S[hi+1,i]:
                            hi += 1
                            if hi==n_fft-1:
                                break
                    lo = np.min([loc-1,0])
                    if lo>0:
                        while S[lo,i]>S[lo-1,i]:
                            lo -= 1
                            if lo<0:
                                break
                    FgMask[lo:hi,i] = S[lo:hi,i]/np.max(S[:,i])

        return FgMask, W_C
    
    
    
    def get_background(self, S, PkLoc):
        Spec_corr, W_C = self.get_framewise_correlation(S)
        W_C = W_C/np.max(W_C)
        W_C = 1-W_C
        W_E = self.get_framewise_energy(S)
        W_E /= np.max(W_E)
        W_E = 1 - W_E
        W_C = np.multiply(W_C, W_E)
        W_C /= np.max(W_C)
        self.T_bg = np.mean(W_C) + self.k*np.std(W_C)
        BgMask = np.ones(np.shape(S))
        n_fft = np.shape(S)[0]
        self.bg_idx = W_C<=self.T_bg
        for i in range(np.shape(PkLoc)[1]):
            nPks = np.sum(PkLoc[:,i]>0)
            if W_C[i]<=self.T_bg:
                for j in range(nPks):
                    hi = None
                    lo = None
                    loc = int(PkLoc[j,i])
                    hi = loc+1
                    if hi<n_fft-1:
                        while S[hi,i]>S[hi+1,i]:
                            hi += 1
                            if hi==n_fft-1:
                                break
                    lo = np.min([loc-1,0])
                    if lo>0:
                        while S[lo,i]>S[lo-1,i]:
                            lo -= 1
                            if lo<0:
                                break
                    BgMask[lo:hi,i] = 0
            else:
                BgMask[:,i] = W_C[i]
                
            rescaling_values = (n_fft-np.sum(BgMask[:,i]))*(BgMask[:,i]/(np.sum(BgMask[:,i])+1e-10))
            BgMask[:,i] += rescaling_values
        return BgMask, W_C

