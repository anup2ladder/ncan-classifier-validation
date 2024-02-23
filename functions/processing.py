"""
    Set of functions to process the EEG data
"""

# Import libraries
import os
import mne
import pyxdf
import numpy as np
import pandas as pd
import scipy.signal as signal

def line_filter(eeg, srate, f_notch, f_order):

    [b,a] = signal.iirnotch(
        w0 = f_notch,
        Q = 30,
        fs = srate,
        )
    
    filtered = signal.filtfilt(
        b = b,
        a = a,
        x = eeg,
        )
    
    # sos = signal.butter(
    #     N = f_order,
    #     Wn = [f_notch-1, f_notch+1],
    #     fs = srate,
    #     btype = "bandstop",
    #     output = "sos"
    #     )
    
    # filtered = signal.sosfiltfilt(
    #     sos = sos,
    #     x = eeg
    # )

    return filtered

def common_spatial_pattern(eeg, labels, n_components, srate):

    # Create MNE Raw object
    info = mne.create_info(
        ch_names = labels,
        ch_types = "eeg",
        sfreq = srate
        )

    raw = mne.io.RawArray(
        data = eeg,
        info = info
        )

    # Apply common spatial pattern
    csp = mne.decoding.CSP(
        n_components = n_components,
        reg = "ledoit_wolf"
        )

    csp.fit(
        raw
        )

    # Apply the filter
    csp_eeg = csp.transform(
        raw
        )

    return csp_eeg