import pandas as pd
from pathlib import Path
import numpy as np
import os
import glob
import re
DATA_ROOT = Path(r'Y:\projects\frog\data')

def get_XII_feats(metadata_dir,gate_name):
    '''
    Load and arrange the annotated compound action potential annotations
    '''
    data_fn = metadata_dir.joinpath(f'{gate_name}-data.csv')
    if not data_fn.is_file():
        raise ValueError(f"No data.csv file found in {str(gate_name)}")
    df = pd.read_csv(data_fn)
    assert df.shape[0]>0, f"No compound APs found in {gate_name}-data.csv"

    return(df)


def get_ni_analog(ni_bin_fn, chan_id):
    '''
    Convenience function to load in a NI analog channel
    :param ni_bin_fn: filename to load from
    :param chan_id: channel index to load
    :return: tvec,analog_dat
    '''
    meta = readSGLX.readMeta(Path(ni_bin_fn))
    bitvolts = readSGLX.Int2Volts(meta)
    ni_dat = readSGLX.makeMemMapRaw(ni_bin_fn,meta)
    analog_dat = ni_dat[chan_id]*bitvolts
    sr = readSGLX.SampRate(meta)
    tvec = np.linspace(0,len(analog_dat)/sr,len(analog_dat))

    return(tvec,analog_dat)


def get_ni_fn(gate_dir):
    '''
    Convinence function to grab the nidaq bin file from the gate directory
    '''
    fn_list = list(gate_dir.glob('*.nidq.bin'))
    if len(fn_list)==0:
        raise ValueError('No analaog data file found')
    elif len(fn_list)>1:
        raise ValueError(f'Too many analog files found:{fn_list}')
    
    return(fn_list[0])


def get_ks_dirs(gate_dir,verbose=True):
    '''
    Given a gate directory, return the phy folders for each probe
    '''
    probe_list = list(gate_dir.rglob('imec*_ks2'))
    probe_list.sort()
    n_probes = len(probe_list)
    if verbose:
        print(f'Number of probes is {n_probes}')
    return(probe_list)


def get_gate_dirs(run_dir):
    '''
    Given a run directory, return all the gate directories 
    '''
    run_name = run_dir.name
    gate_list = list(run_dir.glob(f'*{run_name}*_g*'))
    gate_list.sort()
    return(gate_list)


def gen_ks_list(data_root=DATA_ROOT):
    '''
    Generate a list and metadata of all recordings 
    '''
    ks_df = pd.DataFrame(columns=['run_dir','gate_dir','ks2_dir','run','gate','probe','rec_id'])
    ii=0
    run_dirs = list(data_root.glob('NPX*'))
    run_dirs.sort()
    for run_dir in run_dirs:
        gate_dirs = get_gate_dirs(run_dir)
        for gate_dir in gate_dirs:
            ks2_dirs = get_ks_dirs(gate_dir,verbose=False)
            for probe_idx,ks2_dir in enumerate(ks2_dirs):
                idx = re.search('NPX*',gate_dir.name).start()
                rec_id = gate_dir.name[idx:]
                ks_df.loc[ii,'run_dir'] = run_dir
                ks_df.loc[ii,'gate_dir'] = gate_dir
                ks_df.loc[ii,'ks2_dir'] = ks2_dir
                ks_df.loc[ii,'run'] = run_dir.name
                ks_df.loc[ii,'gate'] = re.search('g\d+',gate_dir.name).group()
                ks_df.loc[ii,'probe'] = re.search('imec\d',ks2_dir.name).group()
                ks_df.loc[ii,'rec_id'] = rec_id + '_' + re.search('imec\d',ks2_dir.name).group()
                ii+=1
    return(ks_df)

def load_phy(ks_dir, use_label='intersect'):
    '''
    Given a phy directory, create the spikes and metrics dataframes. 
    The spikes dataframe is a pandas dataframe with columns "ts","cell_id","cluster_id". 
    "ts" is the time of a given spike
    "cluster_id" is the KS2 assigned cluster identity associated with that spikes
    "cell_id" is a remapped identifier for the unit a spike came from. cell_id maps one to one onto cluster_id, but takes values 0 to N, where N is the number of unique units

    "cell_id" is used after filtering out bad units, and resorting by depth
    
    The metrics data frame gives cluster level information (such as probe location, spike amplitude, firing rate, QC metrics etc...)
    '''

    # spike_df = create_spike_df(ks2_dir)
    if os.path.isfile(f'{ks_dir}/spike_times_sec.npy'):
        ts = np.load(f'{ks_dir}/spike_times_sec.npy').ravel()
    else:
        print('NO SPIKETIMES_SEC FOUND. Converting and saving')
        t_samps = np.load(f'{ks_dir}/spike_times.npy').ravel()
        try:
            ap_bin = glob.glob(f'{ks_dir}/../*ap.bin')[0]
            meta = readSGLX.readMeta(Path(ap_bin))
            spike_sr = readSGLX.SampRate(meta)
        except:
            param_fn = f'{ks_dir}/params.py'
            with open(param_fn,'r') as fid:
                ll = fid.readlines()
            spike_sr = float(ll[-2][14:])
            print(f'Using SR={spike_sr}')

        ts = t_samps/spike_sr
        with open(f'{ks_dir}/spike_times_sec.npy','wb') as fid:
            np.save(fid,ts)

    idx = np.load(f'{ks_dir}/spike_clusters.npy').ravel()
    try:
        metrics = pd.read_csv(f'{ks_dir}/metrics.csv')
    except:
        metrics = pd.read_csv(f'{ks_dir}/waveform_metrics.csv',)

    depths = np.load(f'{ks_dir}/channel_positions.npy')[:, 1]
    dd = pd.DataFrame()
    dd['peak_channel'] = np.arange(len(depths))
    dd['depth'] = depths
    metrics = metrics.merge(dd,how='left',on='peak_channel')

    spikes = pd.DataFrame()
    spikes['ts'] = ts
    spikes['cell_id'] = idx
    spikes = pd.merge(left=spikes, right=metrics[['cluster_id', 'depth']], how='left', left_on='cell_id',
                      right_on='cluster_id')

    if use_label == 'default':
        grp = pd.read_csv(f'{ks_dir}/cluster_group.tsv', delimiter='\t')
        clu_list = grp.query('group=="good"')['cluster_id']
        spikes = spikes[spikes['cluster_id'].isin(clu_list)]
        metrics = metrics.merge(grp,on='cluster_id')
        metrics = metrics[metrics['cluster_id'].isin(clu_list)]
    elif use_label == 'ks':
        grp = pd.read_csv(f'{ks_dir}/cluster_KSLabel.tsv', delimiter='\t')
        clu_list = grp.query('KSLabel=="good"')['cluster_id']
        spikes = spikes[spikes['cluster_id'].isin(clu_list)]
        metrics = metrics.merge(grp,on='cluster_id')
        metrics = metrics[metrics['cluster_id'].isin(clu_list)]
    elif use_label == 'intersect':
        grp = pd.read_csv(f'{ks_dir}/cluster_group.tsv', delimiter='\t')
        kslabel = pd.read_csv(f'{ks_dir}/cluster_KSLabel.tsv', delimiter='\t')
        temp = pd.merge(grp,kslabel,how='inner',on='cluster_id')
        metrics = metrics.merge(grp,on='cluster_id')
        metrics = metrics.merge(kslabel,on='cluster_id')
        temp.query('group=="good" & KSLabel=="good"',inplace=True)
        clu_list = temp['cluster_id']
        spikes = spikes[spikes['cluster_id'].isin(clu_list)]
        metrics = metrics[metrics['cluster_id'].isin(clu_list)]
    else:
        raise NotImplementedError('Use a valid label filter[default,ks,intersect]')


    return(spikes,metrics)


def filter_by_metric(spikes,metrics,expression):
    '''
    Finds all the clusters that pass a particular QC metrics filter expression and keeps only the spikes from
    those clusters
    :param metrics: the metrics csv
    :param spikes: the spikes dataframe with columns [ts,cluster_id,depth]
    :param expression: logical expression to filter the spikes by
    :return: filtered spikes dataframe

    spikes_filt = filter_by_metric(metrics,spikes,'amplitude_cutoff<0.1')
    '''
    clu_list = metrics.query(expression)['cluster_id']
    spikes = spikes[spikes['cluster_id'].isin(clu_list)]
    spikes.reset_index(inplace=True,drop=True)
    metrics = metrics[metrics['cluster_id'].isin(clu_list)]
    return(spikes,metrics)


def run_default_filters(spikes,metrics):
    '''
    Run a battery of filters

    amp default is 20
    fr default is 0.1
    '''
    spikes,metrics = filter_by_metric(spikes,metrics,'isi_viol<0.1')
    spikes,metrics = filter_by_metric(spikes,metrics,'amplitude_cutoff<0.1')
    spikes,metrics = filter_by_metric(spikes,metrics,'amplitude>20')
    spikes,metrics = filter_by_metric(spikes,metrics,'firing_rate>0.1')
    spikes = resort_by_depth(spikes)
    return(spikes,metrics)


def resort_by_depth(spikes):
    '''
    changes the cell_id column to be depth ordered, 0 indexed, and sequential
    That is, cell_id ranges from 0 to N_neurons and there are no skipped indexes
    :param spikes: spikes dataframe
    :return: spikes -- spikes dataframe with modified cell_id column
    '''
    dd = spikes[['cluster_id','depth']].groupby('cluster_id').mean()
    dd = dd.reset_index()
    dd = dd.sort_values('depth')
    dd = dd.reset_index().rename({'index':'cell_id'},axis=1)
    dd = dd.drop('depth',axis=1)
    spikes = spikes.drop('cell_id',axis=1)
    spikes = spikes.merge(dd,on='cluster_id')

    return(spikes)


def load_filtered_phy(ks_dir):
    '''
    Load spikes from a phy directory and apply a standard set of filters
    :param: ks_dir - path to the kilosort directory
    '''
    spikes,metrics = load_phy(ks_dir,use_label='ks')
    spikes,metrics = run_default_filters(spikes,metrics)
    n_units = len(spikes['cell_id'].unique())      

    return(spikes,metrics)

def subsample_spikes(spikes,starts,ends):
    '''
    Takes a spike dict and keeps only the times between starts and ends
    starts and ends must be the same length, and each entry of ends must be greater than starts
    '''
    
    assert(len(starts)==len(ends))
    assert(np.all(ends>starts))
    idx = pd.Index([],dtype='int64')
    
    for start,end in zip(starts,ends):
        dum = spikes.query('ts>@start & ts<@end')
        idx = idx.union(dum.index.values)
    return(spikes.loc[idx,:])

def load_epochs(gate_dir):
    '''
    Load in the regions to analyze
    '''
    epochs_fn_list = list(gate_dir.glob('epoc*s.csv'))
    if len(epochs_fn_list)==0:
        raise ValueError('No epochs file found')
    else:
        epochs_fn = epochs_fn_list[0]
    
    epochs = pd.read_csv(epochs_fn).dropna()

    assert epochs.shape[0]>0, 'No epochs found'
    assert np.all(epochs['start analyses (s)']<epochs['end analyses (s)']), 'Start analyses are not all before end analyses'

    return(epochs)

def load_trills(gate_dir):
    '''
    Load in the manually determined trill periods
    '''
    trills = pd.read_csv(gate_dir.joinpath('compound_aps.csv'))
    trills['start'] = trills['start']/1000
    trills['end'] = trills['end']/1000
    trills['duration'] = trills.eval('end-start')

    assert np.all(trills['duration']>0),'Not all trill durations are positive'
    return(trills)
