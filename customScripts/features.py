import librosa
import numpy as np

# apply short time fourier transform
# take absolute values and square them
# returns power spectrogram
def compute_power_spectrogram(y, frame_length=2048, hop_length=1024):
    spectrogram = librosa.core.stft(y=y, n_fft=frame_length, hop_length=hop_length)
    power_spectrogram = (np.abs(spectrogram))**2
    """"
    print(spectrogram[0])
    print()
    """
    # print(len(power_spectrogram), len(power_spectrogram[0]))
    # print('spectrogram first element:\n\n', spectrogram_real[0], '\n\n\n')
    return power_spectrogram


# apply mel scale to spectrogram (with 40 bins, also try 80)
def mel_transform(spectrogram, bin_number=40):
    mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram, n_mels=bin_number)
    # print(len(mel_spectrogram), len(mel_spectrogram[0]))
    # print('mel spectrogram first element:\n\n', mel_spectrogram[0], '\n\n\n')
    return mel_spectrogram


# apply a log10 scale to spectrogram (resulting magnitudes are in decibels)
def log_scale(spectrogram):
    log_mel_spectrogram = librosa.power_to_db(S=spectrogram)
    # print(len(log_mel_spectrogram), len(log_mel_spectrogram[0]))
    # print('log mel spectrogram first element:\n\n', log_mel_spectrogram[0], '\n\n\n')
    return log_mel_spectrogram


def compute_spectrogram(y, frame_length=2048, hop_length=1024, bin_number=40,
                        printDimensions=False, printFirstFrame=False, printFirstBin=False):

    ps = compute_power_spectrogram(y, frame_length=frame_length, hop_length=hop_length)
    mps = mel_transform(ps, bin_number=bin_number)
    lmps = log_scale(mps)

    if printDimensions:
        print('\ndimensions:\n\n', len(lmps), len(lmps[0]), '\n')
    if printFirstFrame:
        print('log mel spectrogram first frame:\n')
        for i, f in enumerate(lmps):
            print(i, f[0])
    if printFirstBin:
        print('\nlog mel spectrogram first freq. bin:\n\n', lmps[0], '\n\n\n')

    return lmps