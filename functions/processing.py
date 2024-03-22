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

    # Create the notch filter
    [b,a] = signal.iirnotch(
        w0 = f_notch,
        Q = 30,
        fs = srate,
        )
    
    # Implement notch filter
    filtered = signal.filtfilt(
        b = b,
        a = a,
        x = eeg,
        )

    return filtered

def apply_csp(
    eeg_data:np.ndarray,
    labels:np.ndarray,
    n_components:int = 4
    ) -> np.ndarray:
    """
        Applies a CSP filter to EEG epoched data.

        Parameters:
            eeg_data: np.ndarray
                The EEG data. Shape should be [n_epochs, n_channels, n_samples].
            labels: np.ndarray
                The labels for each epoch.
            n_components: int
                The number of components to keep.

        Returns:
            transformed_data: np.ndarray
                The transformed EEG data.
    """

    # Initialize the CSP object
    csp = mne.decoding.CSP(n_components=n_components, reg=None, log=False, norm_trace=False)

    # Fit the CSP filters
    csp.fit(eeg_data, labels)

    # Manually apply the CSP filters to the data
    transformed_data = np.dot(csp.filters_[:n_components], eeg_data)
    transformed_data = np.transpose(transformed_data, (1, 0, 2))

    return transformed_data