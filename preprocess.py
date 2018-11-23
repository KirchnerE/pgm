#!/bin/bash
import os
import mne


DS_PATH = "MNE-eegbci-data/physiobank/database/eegmmidb/S001/"
motor3 = mne.io.read_raw_edf(DS_PATH + "S001R03.edf", preload=True).copy()
motor7 = mne.io.read_raw_edf(DS_PATH + "S001R07.edf", preload=True).copy()
motor11 = mne.io.read_raw_edf(DS_PATH + "S001R11.edf", preload=True).copy()

# strip channel name of .
motor3.rename_channels(lambda x: x.strip('.'))
motor7.rename_channels(lambda x: x.strip('.'))
motor11.rename_channels(lambda x: x.strip('.'))

# select for relevant channels
selection = ["C3", "Cz", "C4", "Fc1", "Fc2", "Cp1", "Cp2", "Pz"]
picks = mne.pick_channels(motor3.info["ch_names"], selection)

m3_events = mne.find_events(motor3)
m7_events = mne.find_events(motor7)
m11_events = mne.find_events(motor11)

m3_epochs = mne.Epochs(motor3, m3_events, picks=picks, tmin=-0.5, tmax=0)
m7_epochs = mne.Epochs(motor7, m7_events, picks=picks, tmin=-0.5, tmax=0)
m11_epochs = mne.Epochs(motor11, m11_events, picks=picks, tmin=-0.5, tmax=0)

m3_epochs.load_data()
m7_epochs.load_data()
m11_epochs.load_data()

# apply band-pass filter
m3_epochs.filter(l_freq=0.5, h_freq=5.0)
m7_epochs.filter(l_freq=0.5, h_freq=5.0)
m11_epochs.filter(l_freq=0.5, h_freq=5.0)

# downsample to 128Hz
m3_epochs.resample(128)
m7_epochs.resample(128)
m11_epochs.resample(128)

os.mkdir("preprocessed_files")
m3_epochs.save("preprocessed_files/pp_S001R03-epo.fif")
m7_epochs.save("preprocessed_files/pp_S001R07-epo.fif")
m11_epochs.save("preprocessed_files/pp_S001R11-epo.fif")
