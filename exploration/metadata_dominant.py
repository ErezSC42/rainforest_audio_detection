import pandas as pd
from dominant_freq_extraction import iterate_and_extract_dominant_freq
import os
from matplotlib import pyplot as plt


def merge_with_meta(meta_csv_path, flacs_path ):
    df_train = pd.read_csv(meta_csv_path)
    print(df_train.describe)
    path = os.path.realpath(flacs_path)
    df_dom_freq = iterate_and_extract_dominant_freq(path, 100)
    return df_train.merge(df_dom_freq, left_on=['recording_id'], right_on=['recording_id'])


if __name__ == '__main__':
    d = merge_with_meta("../Data/train_tp.csv", '../Data/init_test')
    print(d.describe)
    d['dom_freq'].plot()
    plt.show()

