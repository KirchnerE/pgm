import mne

#epochs = mne.read_epochs('preprocessed_files/pp_S001R03-epo.fif')
DS_PATH = "MNE-eegbci-data/physiobank/database/eegmmidb/S001/"
raw = mne.io.read_raw_edf(DS_PATH + "S001R03.edf", preload=True).copy()
raw.rename_channels(lambda x: x.strip('.'))
#left_events = epochs['1']
#right_events = epochs['2']

picks = mne.pick_channels(raw.info["ch_names"], ["C3", "C4"])

events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, picks=picks, tmin=-0.5, tmax=1)
epochs.load_data()
epochs.filter(l_freq=0.5, h_freq=5.0)

#left_events.average(picks=picks).plot()
#right_events.average(picks=picks).plot()
epochs.average().plot()