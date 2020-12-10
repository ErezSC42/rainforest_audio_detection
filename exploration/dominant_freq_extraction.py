import librosa
import  os
import numpy as np
import pandas as pd

# C:\Workdir\rainforest_audio_detection\exploration\Data\init_test\
# C:\Workdir\rainforest_audio_detection\Data\init_test
def iterate_and_extract_dominant_freq(path,bins):
        for f in os.listdir(path):
            try:
                signal, sample_rate = librosa.load(path + "\\" + f)
                fft = np.fft.fft(signal)
                # calculate abs values on complex numbers to get magnitude
                spectrum = np.abs(fft)
                # create frequency variable
                f = np.linspace(0, sample_rate, len(spectrum))
                # take half of the spectrum and frequency
                left_spectrum = spectrum[:int(len(spectrum) / 2)]
                left_f = f[:int(len(spectrum) / 2)]
                print(left_f, left_spectrum)
                most_dom_freq_in_bin = [i + np.argmax(left_spectrum[i:i + bins]) for i in
                                        range(0, len(left_spectrum), bins)]  # the most dominant freq in bin
                print(most_dom_freq_in_bin)
                most_dom_freq = int(np.argmax([left_spectrum[i] for i in most_dom_freq_in_bin]))
                print(most_dom_freq)
                print(left_f[most_dom_freq * bins])
                input()
            except:
                print("fail")



if __name__ == '__main__':
    path = os.path.realpath('../data/init_test')
    iterate_and_extract_dominant_freq(path, 100)
    print("yes")
