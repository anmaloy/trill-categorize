import pandas as pd
from pathlib import Path
import numpy as np

def get_XII_feats(gate_dir):
    '''
    Load and arrange the annotated compound action potential annotations
    '''
    data_fn = gate_dir.joinpath('data.csv')
    if not data_fn.isfile():
        raise ValueError(f"No data.csv file found in {str(gate_dir)}")
    df = pd.read_csv(data_fn)
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


def get_ks_dirs(gate_dir):
    '''
    Given a gate directory, return the phy folders for each probe
    '''
    probe_list = list(gate_dir.rglob('imec*_ks2'))
    n_probes = len(probe_list)
    print(f'Number of probes is {n_probes}')
    return(probe_list)


def get_gate_dirs(run_dir):
    '''
    Given a run directory, return all the gate directories 
    '''
    run_name = run_dir.name
    gate_list = list(run_dir.glob(f'*{run_name}*_g*'))
    return(gate_list)


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
        ap_bin = glob.glob(f'{ks_dir}/../*ap.bin')[0]
        meta = readSGLX.readMeta(Path(ap_bin))
        spike_sr = readSGLX.SampRate(meta)
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
    elif use_label == 'ks':
        grp = pd.read_csv(f'{ks_dir}/cluster_KSLabel.tsv', delimiter='\t')
        clu_list = grp.query('KSLabel=="good"')['cluster_id']
        spikes = spikes[spikes['cluster_id'].isin(clu_list)]
        metrics = metrics.merge(grp,on='cluster_id')
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
    metrics = metrics[metrics['cluster_id'].isin(clu_list)]
    return(spikes,metrics)


def run_default_filters(spikes,metrics):
    '''
    Run a battery of filters
    '''
    spikes,metrics = filter_by_metric(spikes,metrics,'isi_viol<0.1')
    spikes,metrics = filter_by_metric(spikes,metrics,'amplitude_cutoff<0.1')
    spikes,metrics = filter_by_metric(spikes,metrics,'amplitude>50')
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
    return(spikes,metrics)