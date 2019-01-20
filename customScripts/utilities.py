import librosa
import numpy as np

def main():
    print('main()')

def load_audio(file_path, sampling_rate, inf=False):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    if inf:
        print("\n\nSample values: ", y)
        print("sample #: ", len(y))
        print("sampling rate: ", sr, '\n\n\n')
    return y




if __name__ == "__main__":
    main()
