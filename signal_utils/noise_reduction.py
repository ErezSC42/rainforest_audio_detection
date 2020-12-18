import os
import tqdm
import time
import scipy
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
import plotly.express as px
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import timedelta as td


SAMPLING_RATE = 48000
# https://timsainburg.com/noise-reduction-python.html


def extract_noise_from_signal(signal):
    '''
    this is a dummy function so we will untils Ella's function is ready
    :param signal:
    :return:
    '''
    return np.zeros(2 * SAMPLING_RATE)

def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


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

def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()

def _remove_noise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise
    1.35 seconds for 60 seconds
    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    #print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal


class NoiseReducer():
    def __init__(
            self,
            n_grad_freq=2,
            n_grad_time=4,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_std_thresh=1.5,
            prop_decrease=1.0,
            verbose=False,
            visual=False):
        self.n_grad_freq = n_grad_freq
        self.n_grad_time = n_grad_time
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_std_thresh = n_std_thresh
        self.prop_decrease = prop_decrease
        self.verbose = verbose
        self.visual = visual

    def reduce_noise(self, input_signal_clip, input_noise_clip):
        '''
        :param input_signal_clip: np.array of the original signal. contains both actual animal recording and noise
        :param input_noise_clip: np.array of noise. should not contain data from actual bird/frog in it
        :return:
        '''
        return _remove_noise(
            audio_clip=input_signal_clip,
            noise_clip=input_noise_clip,
            n_grad_freq=self.n_grad_freq,
            n_grad_time=self.n_grad_time,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_std_thresh=self.n_std_thresh,
            prop_decrease=self.prop_decrease,
            verbose=self.verbose,
            visual=self.visual)

    def batch_noise_reduce(self, recordings_array, batch_size=24, **params):
        '''
        :param recordings_array: np.array([recording_count, recording_len]
        :return: denoised audio
        '''
        # TODO get noise extractor function from Ella
        denoised_recordings = []
        batches_count = int(np.ceil(recordings_array.shape[0] / batch_size))
        recording_chunks = np.array_split(recordings_array, batches_count)

        for chunk in tqdm.tqdm(recording_chunks):
            current_recording = recordings_array[chunk, :]
            current_noise = extract_noise_from_signal(current_recording)
            current_denoised_recording = self.reduce_noise(current_recording, current_noise)
            denoised_recordings.append(current_denoised_recording)
            print(current_recording)
        denoised_recordings_array = np.vstack(denoised_recordings)
        assert denoised_recordings_array.shape == recordings_array.shape
        return denoised_recordings_array


if __name__ == '__main__':
    data_path = os.path.join("data", "train", "00b404881.flac")
    signal, sampling_rate = sf.read(data_path)

    SEC = 20
    NOISE = 4
    signal = signal[SEC*sampling_rate:(SEC+10)*sampling_rate]
    signal_noise = signal[int(NOISE*sampling_rate):int((NOISE+1)*sampling_rate)]

    print(f"sampling rate: {sampling_rate}")
    print(f"signal type:{type(signal)}, shape:{signal.shape}")

    t_axis = (1/sampling_rate) * np.arange(0,len(signal))
    fig = px.line(x=t_axis, y=signal)
    fig.show()


    # stft_signal = np.abs(librosa.stft(y=signal, n_fft=64))  # shape: (1025, 5625), dtype=complex128
    # plot_spectrogram(stft_signal, "Test")

    noise_reducer = NoiseReducer()

    output = noise_reducer.reduce_noise(
        input_signal_clip=signal,
        input_noise_clip=signal_noise)

    # original
    print(f"original signal shape: {signal.shape}")
    sd.play(signal, sampling_rate, blocking=True)

    # noise
    time.sleep(4)
    sd.play(signal_noise, sampling_rate, blocking=True)

    # denoised signal
    time.sleep(4)
    print(f"output signal shape: {output.shape}")
    sd.play(output, sampling_rate, blocking=True)

