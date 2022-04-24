import sys
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display, feature

from torchnmf.nmf import NMFD

def run_example(y=None, R=3, windowsize=2048, T=400, show=True):
    if y is None:
        if librosa.__version__ >= '0.9':
            y, sr = librosa.load(librosa.example('vibeace'))
        else:
            y, sr = librosa.load(librosa.util.example_audio_file())
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize), return_complex=True).abs().cuda()
    if S.ndim == 2:
        S = S.unsqueeze(0)
    elif S.ndim == 3 and S.size(0) > 1: # only one channel at a time in this example
        S = S[0:1]

    F = S.shape[0] - 1

    net = NMFD(S.shape, T=T, rank=R).cuda()
    net.fit(S.cuda(), verbose=True)
    V = net()
    W, H = net.W.detach().cpu().numpy(), net.H.squeeze().detach().cpu().numpy()
    V = V.squeeze().detach().cpu().numpy()

    if len(W.shape) < 3:
        W = W.reshape(*W.shape, 1)

    if show:
        plt.figure(figsize=(10, 15))
        for i in range(R):
            plt.subplot(R+3, 1, i + 1)
            display.specshow(librosa.amplitude_to_db(W[:, i], ref=np.max), y_axis='log')
            plt.title('Template ' + str(i + 1))

        plt.subplot(R+3, 1, i+2)
        display.specshow(librosa.amplitude_to_db(H, ref=np.max), x_axis='time')
        plt.title('Activations')

        plt.subplot(R+3, 1, i+3)
        display.specshow(librosa.amplitude_to_db(V, ref=np.max), y_axis='log', x_axis='time')
        plt.title('Reconstructed spectrogram')

        plt.subplot(R+3, 1, i+4)
        display.specshow(librosa.amplitude_to_db(S.clone().detach().cpu().numpy()[0], ref=np.max), y_axis='log', x_axis='time')
        plt.title('Source spectrogram')
        plt.tight_layout()
        plt.show()

    return V, S, W


if __name__ == '__main__':

    run_example(R=3 if len(sys.argv) == 1 else int(sys.argv[1]))
