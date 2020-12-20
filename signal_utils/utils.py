import os
import numpy as np
from main import load_recordings


def split_recordings_to_segments(
        recording_array : np.array,
        segment_sec_len : int = 5,
        sampling_rate : int = 48000):
    '''
    :param recording_array:
    :param segments_len:
    :param sampling_rate:
    :return:
    '''
    #TODO add dataframe as optional input (for training only) and reshape it to match the newly reshaped data
    original_sec_len = 60
    segment_len = segment_sec_len * sampling_rate
    segments_count = int(len(recording_array) * (original_sec_len / segment_sec_len))
    return recording_array.reshape([segments_count, segment_len])


if __name__ == '__main__':
    TRAIN_DATA_PATH = os.path.join("..","data", "train")
    recording_signal_train = load_recordings(TRAIN_DATA_PATH, 10)
    res = split_recordings_to_segments(recording_signal_train)
    print()
