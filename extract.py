import os
import numpy as np
import soundfile
import scipy
import h5py
from collections import defaultdict


# settings
supported_files = ['.wav', '.flac', '.ogg']
data_folder = "data/"
bins = 9 * 24  # I think there are about 9 octaves in the 50 - 22khz, range, we use 24 intervals per octave
samplerate = 11000
binsize = samplerate / bins  # 2**10
samplelength = 10  # seconds


def stft(samples, samplerate, framesz=0.050, hop=0.025):
    """
    spectragram

    args:
        framesz: frame  size
        hop: hop size
    """
    framesamp = int(framesz * samplerate)
    hopsamp = int(hop * samplerate)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w * samples[i:i + framesamp])
                     for i in range(0, len(samples) - framesamp, hopsamp)])

    transposed = np.transpose(X)  # time on xaxes
    return transposed


def istft(X, samplerate, T, hop=0.025):
    """
    inverse spectragram
    """
    transposed = np.transpose(X)
    X = transposed
    x = scipy.zeros(T*samplerate)
    framesamp = X.shape[1]
    hopsamp = int(hop*samplerate)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x


def get_train_files(data_folder):
    labels = [n for n in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, n))]
    labeled_files = []
    for label in labels:
        label_folder = os.path.join(data_folder, label)
        for dp, dn, filenames in os.walk(label_folder):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in supported_files:
                    labeled_files.append((os.path.join(dp, f), label))
    return labeled_files


def open_sound_and_normalise(path):
    """
    returns mono audio of given samplerate
    """
    orig_samples, orig_samplerate = soundfile.read(path)
    ratio = orig_samplerate / samplerate
    samples = orig_samples[::ratio, 0]
    return samples


def main():
    labeled_files = get_train_files(data_folder)

    # test run
    labeled_files = labeled_files[:2]

    dataset = []

    for path, label in labeled_files:
        print("** file: " + path)
        audio = open_sound_and_normalise(path)
        num_samples = len(audio) / samplerate / samplelength
        for sample_num in range(num_samples):
            start = sample_num * samplerate
            end = sample_num * samplerate + samplerate * samplelength
            sub_sample = audio[start:end]
            spectragram = stft(sub_sample, samplerate)
            width, height = spectragram.shape  # x is time, y is frequency
            dataset.append((sub_sample, label))

    f = h5py.File('audio.hdf5', 'w')
    f.create_dataset(name='data', shape=(width, height), dtype=)


if __name__ == '__main__':
    main()

