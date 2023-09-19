import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import chirp, find_peaks, peak_widths
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def readMeta(binFullPath):
    metaName = binFullPath.stem + ".meta"
    metaPath = Path(binFullPath.parent / metaName)
    metaDict = {}
    if metaPath.exists():
        # print("meta file present")
        with metaPath.open() as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return metaDict


def Int2Volts(meta):
    if meta['typeThis'] == 'imec':
        fI2V = float(meta['imAiRangeMax'])/512
    else:
        fI2V = float(meta['niAiRangeMax'])/32768
    return fI2V


def SampRate(meta):
    if meta['typeThis'] == 'imec':
        srate = float(meta['imSampRate'])
    else:
        srate = float(meta['niSampRate'])
    return srate


def makeMemMapRaw(binFullPath, meta):
    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
    print("nChan: %d, nFileSamp: %d" % (nChan, nFileSamp))
    rawData = np.memmap(binFullPath, dtype='int16', mode='r',
                        shape=(nChan, nFileSamp), offset=0, order='F')
    return rawData


def get_ni_analog(ni_bin_fn, chan_id):
    '''
    Convenience function to load in a NI analog channel
    :param ni_bin_fn: filename to load from
    :param chan_id: channel index to load
    :return: analog_dat
    '''
    meta = readMeta(Path(ni_bin_fn))
    bitvolts = Int2Volts(meta)
    ni_dat = makeMemMapRaw(ni_bin_fn, meta)
    analog_dat = ni_dat[chan_id] * bitvolts
    sr = SampRate(meta)
    tvec = np.linspace(0, len(analog_dat)/sr, len(analog_dat))

    return tvec, analog_dat


chan_id = 0
fileName = 'NPX-S2-27'
gate = 3
binPath = Path(f'data\\{fileName}\\catgt_{fileName}_g{gate}\\{fileName}_g{gate}_tcat.nidq.bin')
meta = readMeta(binPath)
sRate = SampRate(meta)

ni_time, ni_data = get_ni_analog(binPath, chan_id)
# height: amplitude in v for filtering out spikes, distance: ms between spikes to avoid duplicates
cutoff = 0.03
distance = 10
peaks, properties = find_peaks(ni_data, height=cutoff, distance=distance*sRate/1000)
width_half = peak_widths(ni_data, peaks, rel_height=0.5)
lst = list(width_half)
lst[0] = lst[0]/sRate
lst[2:] = (x/sRate for x in lst[2:])
width_half = tuple(lst)
width_full = peak_widths(ni_data, peaks, rel_height=0.75)
lst = list(width_full)
lst[0] = lst[0]/sRate
lst[2:] = (x/sRate for x in lst[2:])
width_full = tuple(lst)
peaks_time = ni_time[peaks]

targetdf = pd.read_csv('data\\targets.csv')
# ~55ms was the time the original data was adjusted by
targetdf.start = (targetdf.start+55)/1000
targetdf.end = (targetdf.end+55)/1000

columns = ['time(s)', 'amplitude(v)', 'delay(s)', 'fwidth(len)', 'fwidth(amp)', 'hwidth(len)', 'hwidth(amp)', 'type']
df = pd.DataFrame(columns=columns)
df['time(s)'] = peaks_time
df['amplitude(v)'] = ni_data[peaks]
df['delay(s)'] = df['time(s)'].diff()
df['fwidth(len)'] = width_full[0]
df['fwidth(amp)'] = width_full[1]
df['hwidth(len)'] = width_half[0]
df['hwidth(amp)'] = width_half[1]
lst = targetdf.apply(lambda row: (row['start'], row['end']), axis=1)
conditions = [(df['time(s)'] > x[0]) & (df['time(s)'] < x[1]) for x in lst]
choices = targetdf.type.values
df['type'] = np.select(conditions, choices, default=np.nan)
df = df.dropna()

plt.plot(ni_time, ni_data, linewidth=0.05)
for index, row in df.iterrows():
    if row[7] == 'S':
        plt.plot(peaks_time[index], ni_data[peaks][index], '.', color='tomato', alpha=0.75)
    elif row[7] == 'F':
        plt.plot(peaks_time[index], ni_data[peaks][index], '.', color='forestgreen', alpha=0.75)
    elif row[7] == 'I':
        plt.plot(peaks_time[index], ni_data[peaks][index], '.', color='olive', alpha=0.75)
    elif row[7] == 'VC':
        plt.plot(peaks_time[index], ni_data[peaks][index], '.', color='slateblue', alpha=0.75)
    elif row[7] == 'B':
        plt.plot(peaks_time[index], ni_data[peaks][index], '.', color='silver', alpha=0.75)
    else:
        plt.plot(peaks_time[index], ni_data[peaks][index], '.', color='black', alpha=0.75)
plt.hlines(*width_half[1:], color="C2")
plt.hlines(*width_full[1:], color="C3")
plt.ylim(-0.4, 0.7)
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
lst = targetdf[['start', 'end']]
for index, row in targetdf.iterrows():
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

# plt.scatter(df['amplitude(v)'], df['hwidth(len)'])
# plt.show()

df.to_csv('data\\data.csv', index=False)
df.to_csv(f'{fileName}_g{gate}-data.csv')
