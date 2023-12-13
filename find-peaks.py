import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import find_peaks, peak_prominences, peak_widths

from src.readSGLX import readMeta, SampRate
from src.io import get_ni_analog


def populate_data(peaks_time, properties, widths, prominence, delay):
    targets = pd.read_csv(f'data\\targets\\{NIDAQ.fileName}_g{NIDAQ.gate}-targets.csv')
    targets.start = (targets.start + delay) / 1000
    targets.end = (targets.end + delay) / 1000
    data = pd.DataFrame()

    data['time(s)'] = peaks_time
    data['amplitude(v)'] = properties['peak_heights']
    data['delay(s)'] = data['time(s)'].diff()
    data['width(len)'] = widths[0]
    data['width(amp)'] = widths[1]
    data['width(start)'] = widths[2]
    data['width(finish)'] = widths[3]
    data['prominence'] = prominence
    data['thresholds'] = properties['left_thresholds'] + properties['right_thresholds']
    lst = targets.apply(lambda row: (row['start'], row['end']), axis=1)
    conditions = [(data['time(s)'] > x[0]) & (data['time(s)'] < x[1]) for x in lst]
    choices = targets.type.values
    data['type'] = np.select(conditions, choices, default=np.nan)
    return data, targets


def get_widths(nidaq_data, peaks, height):
    widths = []
    amps = []
    left = []
    right = []
    out = []
    for peak in peaks:
        left_cross = None
        right_cross = None
        left_amp = None
        right_amp = None
        for i in range(peak):
            left_idx = peak - i
            right_idx = peak + i
            if left_idx >= 0 and nidaq_data[left_idx] < height and not left_cross:
                left_cross = left_idx
                left_amp = nidaq_data[left_idx]
            if right_idx < len(nidaq_data) and nidaq_data[right_idx] < height and not right_cross:
                right_cross = right_idx
                right_amp = nidaq_data[right_idx]
            if left_cross and right_cross and left_amp and right_amp:
                width = right_cross - left_cross
                amp = np.mean([left_amp, right_amp])
                widths.append(width)
                amps.append(amp)
                left.append(left_cross)
                right.append(right_cross)
                break
    out.append(np.array(widths))
    out.append(np.array(amps))
    out.append(np.array(left))
    out.append(np.array(right))
    return out


def get_peaks(nidaq_time, nidaq_data, cutoff, distance=10, delay=0, other=False):
    peaks, properties = find_peaks(nidaq_data, height=cutoff, threshold=0, distance=distance * sRate / 1000)
    widths = get_widths(nidaq_data, peaks, cutoff)
    lst = list(widths)
    lst[0] = lst[0] / sRate
    lst[2:] = (x / sRate for x in lst[2:])
    widths = tuple(lst)
    peaks_time = nidaq_time[peaks]
    prominence = peak_prominences(nidaq_data, peaks)[0]
    data, targets = populate_data(peaks_time, properties, widths, prominence, delay)
    if other is True:
        data['type'] = data['type'].replace('nan', 'O')
    return data, targets, cutoff


class NIDAQ:
    channel = 0
    fileName = 'NPX-S2-39'
    gate = 3
    t_delay = 0
    binPath = Path(f'data\\{fileName}\\catgt_{fileName}_g{gate}\\{fileName}_g{gate}_tcat.nidq.bin')

    def spikes_chart(self, data, targets, nidaq_time, nidaq_data, cutoff):
        data = data.replace('nan', pd.NA).dropna(axis=0)
        plt.plot(nidaq_time, nidaq_data, linewidth=0.1)
        for index, row in data.iterrows():
            if row['type'] == 'S':
                plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='tomato', alpha=0.75)
            elif row['type'] == 'F':
                plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='forestgreen', alpha=0.75)
            elif row['type'] == 'I':
                plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='gold', alpha=0.75)
            elif row['type'] == 'VC':
                plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='slateblue', alpha=0.75)
            elif row['type'] == 'B':
                plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='silver', alpha=0.75)
            else:
                plt.plot(data['time(s)'][index], data['amplitude(v)'][index], '.', color='black', alpha=0.75)
            plt.hlines(data['width(amp)'][index], data['width(start)'][index], data['width(finish)'][index],
                       color="C2")
        plt.ylim(-0.4, 0.9)
        plt.axhline(y=cutoff, color='red', linewidth=0.3)
        legend = [mpatches.Patch(color='yellow', label='Intro Trill', alpha=0.5),
                  mpatches.Patch(color='green', label='Fast Trill', alpha=0.5),
                  mpatches.Patch(color='red', label='Slow Trill', alpha=0.5),
                  mpatches.Patch(color='blue', label='Vocal Comp AP', alpha=0.5),
                  mpatches.Patch(color='grey', label='Breathing', alpha=0.5)]
        plt.legend(handles=legend, loc='lower right')
        plt.xlabel('Times (s)')
        plt.ylabel('Nerve (v)')
        plt.title('NIDAQ Nerve Spikes')
        for index, row in targets.iterrows():
            if row[3] == 'S':
                plt.axvspan(row[1], row[2], color='red', alpha=0.25)
            if row[3] == 'F':
                plt.axvspan(row[1], row[2], color='green', alpha=0.25)
            if row[3] == 'I':
                plt.axvspan(row[1], row[2], color='yellow', alpha=0.25)
            if row[3] == 'VC':
                plt.axvspan(row[1], row[2], color='blue', alpha=0.25)
            if row[3] == 'B':
                plt.axvspan(row[1], row[2], color='grey', alpha=0.25)
        plt.show()

        data.to_csv(f'data\\data\\{self.fileName}_g{self.gate}-data.csv', index=False)


start = NIDAQ()
binMeta = readMeta(start.binPath)
sRate = SampRate(binMeta)
ni_time, ni_data = get_ni_analog(start.binPath, start.channel)
df, targetdf, p_cutoff = get_peaks(ni_time, ni_data, cutoff=(ni_data.std() * 2), delay=start.t_delay, other=False)
start.spikes_chart(df, targetdf, ni_time, ni_data, p_cutoff)
