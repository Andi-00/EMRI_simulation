import numpy as np

import audioflux as af

from audioflux.type import SpectralFilterBankScaleType, WaveletContinueType

sr = 1

t = np.arange(0, 1000, 5)

signal = np.sin(t / 30)


# Create CWT object and extract cwt

cwt_obj = af.CWT(num=100, radix2_exp=12, samplate=sr,

                 wavelet_type=WaveletContinueType.MORSE,

                 scale_type=SpectralFilterBankScaleType.OCTAVE)


# The cwt() method can only extract data of fft_length=2**radix2_exp=4096

cwt_arr = cwt_obj.cwt(signal)

cwt_arr = np.abs(cwt_arr)


# Display spectrogram

import matplotlib.pyplot as plt

from audioflux.display import fill_spec

audio_len = signal.shape[-1]

fig, ax = plt.subplots()

img = fill_spec(cwt_arr, axes=ax,

          x_coords=cwt_obj.x_coords(),

          y_coords=cwt_obj.y_coords(),

          x_axis='time', y_axis='log',

          title='CWT Spectrogram')

fig.colorbar(img, ax=ax)

plt.show()