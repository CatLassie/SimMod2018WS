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

def load_ground_truth(file_path, delimiter):
    gt = np.genfromtxt(fname=file_path, delimiter=delimiter)
    return gt.astype(np.int64)

def compute_0_padded_gt(gt, audio_length):
    padded_gt = []
    for i in range(audio_length):
        padded_gt.append(0)
    for i in range(len(gt)):
        padded_gt[gt[i]] = 1
    return padded_gt

def adjust_gt_to_frames(gt, conversion_const):
    converted_gt = []
    for i, element in enumerate(gt):
        for j in range(conversion_const):
            converted_gt.append(element)
    return converted_gt



if __name__ == "__main__":
    main()
