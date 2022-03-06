import statsmodels.datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

data = statsmodels.datasets.co2.load_pandas().data
data.co2.interpolate(inplace=True)

# convert into x and y
x = list(range(len(data.index)))
y = data.co2
dic_data = {"week": x, "co2": y}
output = pd.DataFrame(dic_data)
output.to_csv("data.csv", index=False, encoding='utf8')

# plot the co2 data
fig = plt.figure()
plt.plot(x, y)
plt.ylabel('CO2')
plt.xlabel('Week')
plt.show()

# apply fast fourier transform and take absolute values
f = abs(np.fft.fft(y))

# get the list of frequencies
num = np.size(x)
freq = [i / num for i in list(range(num))]

# get the list of spectrums
spectrum = f.real * f.real + f.imag * f.imag
nspectrum = spectrum / spectrum[0]

# plot nspectrum per frequency, with a semilog scale on nspectrum
plt.semilogy(freq, nspectrum)
plt.show()

# improve the plot by adding periods in number of weeks rather than  frequency
results = pd.DataFrame({'freq': freq, 'nspectrum': nspectrum})
results['period'] = results['freq'] / (1/52)
plt.semilogy(results['period'], results['nspectrum'])
plt.show()

# improve the plot by convertint the data into grouped per week to avoid peaks
results['period_round'] = results['period'].round()
grouped_week = results.groupby('period_round')['nspectrum'].sum()
plt.semilogy(grouped_week.index, grouped_week)
plt.xticks([1, 13, 26, 39, 52])
plt.show()

# use the seasonal_decompose function to observe the same conclusion
res = sm.tsa.seasonal_decompose(data.co2)
resplot = res.plot()
resplot.show()
