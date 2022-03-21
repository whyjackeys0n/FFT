import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp

# Load the audio file
sample_rate, noised_signs = wf.read("./noised.wav")
print(sample_rate, noised_signs.shape)
noised_signs = noised_signs / (2 ** 15)
times = np.arange(noised_signs.size) / sample_rate

# Plot the audio frequency
mp.figure("Filter", facecolor="lightgray")
mp.subplot(221)
mp.title("Time Domain", fontsize=12)
mp.ylabel("Noised_signal", fontsize=12)
mp.grid(linestyle=":")
mp.plot(times[:200], noised_signs[:200], color="b", label="Noised")
mp.legend()
mp.tight_layout()

complex_ary = nf.fft(noised_signs)

fft_freqs = nf.fftfreq(noised_signs.size, times[1] - times[0])
fft_pows = np.abs(complex_ary)

mp.subplot(222)
mp.title("Frequency", fontsize=12)
mp.ylabel("pow", fontsize=12)
mp.grid(linestyle=":")
mp.semilogy(fft_freqs[fft_freqs > 0], fft_pows[fft_freqs > 0], color="orangered", label="Noised")
mp.legend()
mp.tight_layout()

fund_freq = fft_freqs[fft_pows.argmax()]
noised_indices = np.where(fft_freqs != fund_freq)
filter_fft = complex_ary.copy()
filter_fft[noised_indices] = 0
filter_pow = np.abs(filter_fft)

mp.subplot(224)
mp.title("Filter Frequency ", fontsize=12)
mp.ylabel("pow", fontsize=12)
mp.grid(linestyle=":")
mp.plot(fft_freqs[fft_freqs > 0], filter_pow[fft_freqs > 0], color="orangered", label="Filter")
mp.legend()
mp.tight_layout()

filter_sign = nf.ifft(filter_pow).real

mp.subplot(223)
mp.title("Filter Time Domain", fontsize=12)
mp.ylabel("filter_signal", fontsize=12)
mp.grid(linestyle=":")
mp.plot(times[:200], filter_sign[:200], color="b", label="Filter")
mp.legend()
mp.tight_layout()

wf.write('./filter.wav', sample_rate, (filter_sign * 2 ** 15).astype(np.int16))
mp.show()
