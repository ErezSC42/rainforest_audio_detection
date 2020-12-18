import os
import tqdm
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


def load_recordings(data_path_folder):
    train_records = []
    # load dataset
    filelist = os.listdir(data_path_folder)
    if args.train_recording_count > 0:
        filelist = filelist[:args.train_recording_count]
    for filename in tqdm.tqdm(filelist):
        relative_file_name = os.path.join(data_path_folder, filename)
        signal, sampling_rate = sf.read(relative_file_name)
        train_records.append(signal)
    return np.vstack(train_records)


if __name__ == '__main__':
    args = parser.parse_args()

    recording_signal_train = load_recordings(TRAIN_DATA_PATH)

    metadata_train_fp_df = pd.read_csv(METADATA_FP)
    metadata_train_tp_df = pd.read_csv(METADATA_TP)

    # preprocess data
    noise_reducer = NoiseReducer()
    reduced_noise_recordings = noise_reducer.batch_noise_reduce(recording_signal_train)

    # train test split

    # build model

    # train model
