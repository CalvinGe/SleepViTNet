import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
import smote_variants as sv
import random
from random import shuffle
import os

if not os.path.exists('./data/sleep_edf_npy'):
    os.mkdir('./data/sleep_edf_npy')

def extract(data_file, anno_file):
    path1 = './data/'
    path2 = './data/sleep_edf_npy/'

    raw_data = mne.io.read_raw_edf(path1 + data_file)
    anno_data = mne.read_annotations(path1 + anno_file)
    # label the data:
    raw_data.set_annotations(anno_data, emit_warning=False)

    # Link digits with annotations:
    annotation_desc_2_event_id = {'Sleep stage W': 0,
                                  'Sleep stage 1': 1,
                                  'Sleep stage 2': 2,
                                  'Sleep stage 3': 3,
                                  'Sleep stage 4': 3,
                                  'Sleep stage R': 4}
    event_id = {'Sleep stage W': 0,
                'Sleep stage 1': 1,
                'Sleep stage 2': 2,
                'Sleep stage R': 4}

    events_data, _ = mne.events_from_annotations(
        raw_data, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    tmax = 30. - 1. / raw_data.info['sfreq']

    # Only some of the data have stage 3/4, we append it here:
    if np.any(np.unique(events_data[:, 2] == 3)):
        event_id['Sleep stage 3/4'] = 3

    epochs_data = mne.Epochs(raw=raw_data, events=events_data,
                             event_id=event_id, tmin=0., tmax=tmax, baseline=None, preload=True)

    x_data = epochs_data.get_data(picks=['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']) * 1e6
    y_data = epochs_data.events[:, 2]

    np.save(path2+data_file, x_data)
    np.save(path2+anno_file, y_data)

def data_preparation():
    data_file = pd.read_table('data_file.txt', header=None)
    anno_file = pd.read_table('anno_file.txt', header=None)
    for i in range(len(data_file)):
        extract(data_file.iloc[i, 0], anno_file.iloc[i, 0])


data_preparation()
