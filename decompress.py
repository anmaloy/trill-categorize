import glob
import os
import numpy as np
from mtscomp import decompress
from pathlib import Path

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

def metaInfo(meta):
    if meta['typeThis'] == 'imec':
        srate = float(meta['imSampRate'])
    else:
        srate = float(meta['niSampRate'])
    nchan = int(meta['nSavedChans'])
    return srate, nchan

print(f'Decompressing .ap files')
for file in glob.glob('**//*.ap.cbin', recursive=True):
    meta = readMeta(Path(file))
    sRate, nChan = metaInfo(meta)
    stem = os.path.splitext(file)[0]
    base = os.path.basename(file)
    print(f'\tDecompressing {base}')
    try:
        decompress(f'{stem}.cbin', f'{stem}.ch', f'{stem}.bin', write_output=True, overwrite=False)
        print(f'\t{base} decompressed, removing original')
        os.remove(f'{stem}.cbin')
        os.remove(f'{stem}.ch')
    except AssertionError:
        print('\tAssertion Error: bin file is empty, skipping')
    except ValueError:
        print('\tValue Error: Output path already exists, skipping') 
    
print(f'Decompressing .lf files')
for file in glob.glob('**//*.lf.cbin', recursive=True):
    meta = readMeta(Path(file))
    sRate, nChan = metaInfo(meta)
    stem = os.path.splitext(file)[0]
    base = os.path.basename(file)
    print(f'\tDecompressing {base}')
    try:
        decompress(f'{stem}.cbin', f'{stem}.ch', f'{stem}.bin', write_output=True, overwrite=False)
        print(f'\t{base} compressed, removing original')
        os.remove(f'{stem}.cbin')
        os.remove(f'{stem}.ch')
    except AssertionError:
        print('\tAssertion Error, bin file is empty, skipping')
    except ValueError:
        print('\tValue Error: Output path already exists, skipping') 
