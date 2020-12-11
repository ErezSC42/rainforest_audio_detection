import os
import librosa
import scipy as sp
import numpy as np
import soundfile as sf
import plotly.express as px
import matplotlib.pyplot as plt


# https://timsainburg.com/noise-reduction-python.html


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_path = os.path.join("data", "train", "0a4f02024.flac")
    signal, sampling_rate = sf.read(data_path)

    SEC = 10
    signal = signal[SEC*sampling_rate:(SEC+5)*sampling_rate]

    print(f"sampling rate: {sampling_rate}")
    print(f"signal type:{type(signal)}, shape:{signal.shape}")

    fig = px.line(signal)
    fig.show()

    stft_signal = np.abs(librosa.stft(y=signal, n_fft=64))  # shape: (1025, 5625), dtype=complex128
    plot_spectrogram(stft_signal, "Test")

    #print(stft_signal)


