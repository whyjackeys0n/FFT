import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp

# 读取音频文件
sample_rate, noised_signs = wf.read("./noised.wav")
print(sample_rate, noised_signs.shape)  # 采样率 (每秒个数), 采样位移
noised_signs = noised_signs / (2 ** 15)
times = np.arange(noised_signs.size) / sample_rate  # x轴

# 绘制音频 时域图
mp.figure("Filter", facecolor="lightgray")
mp.subplot(221)
mp.title("Time Domain", fontsize=12)
mp.ylabel("Noised_signal", fontsize=12)
mp.grid(linestyle=":")
mp.plot(times[:200], noised_signs[:200], color="b", label="Noised")
mp.legend()
mp.tight_layout()

# 傅里叶变换 频域分析 音频数据
complex_ary = nf.fft(noised_signs)

fft_freqs = nf.fftfreq(noised_signs.size, times[1] - times[0])  # 频域序列
fft_pows = np.abs(complex_ary)     # 复数的摸-->能量  Y轴

# 绘制频域图
mp.subplot(222)
mp.title("Frequency", fontsize=12)
mp.ylabel("pow", fontsize=12)
mp.grid(linestyle=":")
mp.semilogy(fft_freqs[fft_freqs > 0], fft_pows[fft_freqs > 0], color="orangered", label="Noised")
mp.legend()
mp.tight_layout()

# 去除噪声
fund_freq = fft_freqs[fft_pows.argmax()]
noised_indices = np.where(fft_freqs != fund_freq)
filter_fft = complex_ary.copy()
filter_fft[noised_indices] = 0  # 噪声数据位置 =0
filter_pow = np.abs(filter_fft)

# 绘制去除噪声后的 频域图
mp.subplot(224)
mp.title("Filter Frequency ", fontsize=12)
mp.ylabel("pow", fontsize=12)
mp.grid(linestyle=":")
mp.plot(fft_freqs[fft_freqs > 0], filter_pow[fft_freqs > 0], color="orangered", label="Filter")
mp.legend()
mp.tight_layout()

# 对滤波后的数组，逆向傅里叶变换
filter_sign = nf.ifft(filter_pow).real

# 绘制去除噪声的 时域图像
mp.subplot(223)
mp.title("Filter Time Domain", fontsize=12)
mp.ylabel("filter_signal", fontsize=12)
mp.grid(linestyle=":")
mp.plot(times[:200], filter_sign[:200], color="b", label="Filter")
mp.legend()
mp.tight_layout()

# 重新写入新的音频文件
wf.write('./filter.wav', sample_rate, (filter_sign * 2 ** 15).astype(np.int16))
mp.show()
