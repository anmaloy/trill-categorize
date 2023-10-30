import glob
import os
import numpy as np
from mtscomp import compress
from pathlib import Path

def readMeta(binFullPath):
    metaName = binFullPath.stem + ".meta"
    metaPath = Path(binFullPath.parent / metaName)
    print(metaPath)
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

def metaInfo(meta):
    if meta['typeThis'] == 'imec':
        srate = float(meta['imSampRate'])
    else:
        srate = float(meta['niSampRate'])
    nchan = int(meta['nSavedChans'])
    return srate, nchan

print(f'Compressing .ap files')
for file in glob.glob('**//*.ap.bin', recursive=True):
    stem = os.path.splitext(file)[0]
    base = os.path.basename(file)
    print(f'\tCompressing {base}')
    try:
        meta = readMeta(Path(file))
    except KeyError:
        print('KeyError: probably no meta file. Skipping.')
        continue
    sRate, nChan = metaInfo(meta)
    try:
        compress(file, f'{stem}.cbin', f'{stem}.ch', sample_rate=sRate, n_channels=nChan, dtype=np.int16)
        print(f'\t{base} compressed, removing original')
        os.remove(file)
    except AssertionError:
        print('Assertion Error, bin file is empty, skipping')
    
print(f'Compressing .lf files')
for file in glob.glob('**//*.lf.bin', recursive=True):
    stem = os.path.splitext(file)[0]
    base = os.path.basename(file)
    print(f'\tCompressing {base}')
    try:
        meta = readMeta(Path(file))
    except KeyError:
        print('KeyError: probably no meta file. Skipping.')
        continue
    sRate, nChan = metaInfo(meta)
    try:
        compress(file, f'{stem}.cbin', f'{stem}.ch', sample_rate=sRate, n_channels=nChan, dtype=np.int16)
        print(f'\t{base} compressed, removing original')
        os.remove(file)
    except AssertionError:
        print('Assertion Error, bin file is empty, skipping')

