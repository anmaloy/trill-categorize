import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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


def get_imec_analog(imec_bin_fn, chan_id, t0, tf):
    '''
    Convenience function to load in an imce analog channel
    :param imec_bin_fn: filename to load from
    :param chan_id: channel index to load
    :return: analog_dat
    '''
    if type(chan_id) is int:
        chan_id = [chan_id]
    meta = readMeta(Path(imec_bin_fn))
    sr = SampRate(meta)
    s0, sf = (int(t0 * sr), int(tf * sr))
    tvec = np.arange(t0, tf, 1 / sr)
    bitvolts = Int2Volts(meta)
    imec_dat = makeMemMapRaw(imec_bin_fn, meta)
    analog_dat = imec_dat[chan_id, s0:sf] * bitvolts
    tvec = tvec[:analog_dat.shape[1]]
    return tvec, analog_dat


chan_id = [0]
binPath = Path('data\\NPX-S2-28_g0_tcat.nidq.bin')
meta = readMeta(binPath)
sRate = SampRate(meta)
ftime = float(meta['fileTimeSecs'])

tStart = 0
tEnd = ftime

ni_time, ni_data = get_imec_analog(binPath, chan_id, tStart, tEnd)
ni_data = ni_data.T
ni_data = np.squeeze(ni_data)
ni_time = ni_time*1000
# height: good mV height for filtering out spikes, distance: ms between spikes converted for the x-axis here
peaks, properties = find_peaks(ni_data, height=0.02, distance=10*sRate/1000)
peaks_time = [int(x//(sRate/1000)) for x in peaks]

# plt.ylim(-0.4, 0.7)
# plt.plot(ni_time, ni_data, linewidth=0.1)
# plt.plot(peaks_time, ni_data[peaks], 'x')
# plt.xlabel('Times (ms)')
# plt.ylabel('Nerve (V)')
# plt.title('Overlaid NIDAQ Example')
# plt.show()

df = pd.DataFrame(columns=['time(ms)', 'amplitude(v)'])
df['time(ms)'] = peaks_time
df['amplitude(v)'] = ni_data[peaks]
print(df)
df.to_csv('data\\NPX-S2-28_g0-data.csv', index=False)
