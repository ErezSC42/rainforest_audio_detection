import os
import tqdm
import time
import librosa
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

from signal_utils.noise_reduction import NoiseReducer

parser = argparse.ArgumentParser("rainforest audio detection")
parser.add_argument("--train-recording-count", type=int, default=0)


TRAIN_DATA_PATH = os.path.join("data", "train")
METADATA_FP = os.path.join("data", "train_fp.csv")
METADATA_TP = os.path.join("data", "train_tp.csv")


def load_recordings(data_path_folder, recording_num):
    train_records = []
    # load dataset
    filelist = os.listdir(data_path_folder)
    if recording_num > 0:
        filelist = filelist[:recording_num]
    for filename in tqdm.tqdm(filelist):
        relative_file_name = os.path.join(data_path_folder, filename)
        signal, sampling_rate = sf.read(relative_file_name)
        train_records.append(signal)
    return np.vstack(train_records)


if __name__ == '__main__':
    args = parser.parse_args()

    SAMPLING_RATE = 48000

    recording_signal_train = load_recordings(TRAIN_DATA_PATH, args.train_recording_count)

    metadata_train_fp_df = pd.read_csv(METADATA_FP)
    metadata_train_tp_df = pd.read_csv(METADATA_TP)

    # preprocess data
    noise_reducer = NoiseReducer()
    noise_reduced_data = NoiseReducer.batch_noise_reduce(recordings_array=recording_signal_train, batch_size=32)
    # train test split

    # build model

    # train model
