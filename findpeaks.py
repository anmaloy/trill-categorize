from scripts import findpeaks
from src.readSGLX import readMeta, SampRate
from src.io import get_ni_analog


fileName = input('File Name: ')
gate = input('Gate: ')

channel = 0
t_delay = 0

peaks = findpeaks.NIDAQ(fileName, gate)
binMeta = readMeta(peaks.binPath)
sRate = SampRate(binMeta)
ni_time, ni_data = get_ni_analog(peaks.binPath, channel)
df, targetdf, p_cutoff = peaks.get_peaks(ni_time, ni_data, cutoff=(ni_data.std() * 2), samplerate=sRate, delay=t_delay,
                                         other=False)
peaks.spikes_chart(df, targetdf, ni_time, ni_data, p_cutoff)
