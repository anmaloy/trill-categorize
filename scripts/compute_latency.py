'''
Compute the latency of a unit with respect to APs. Identify if it is a PBN or NA unit
This actually computes the peak of the PSTH, not the latency to first, latency from last, spike. They are probably very similar...
'''
from pathlib import Path
import sys
sys.path.append('..')
from src import io
import brainbox.singlecell
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import json
import seaborn as sns
import click

BINSIZE = 0.0005
PRE_WIN = 0.01
POST_WIN = 0.01
PCT_TRILL = 25
rec_id = 5
DATA_ROOT = Path('../../data/')
METADATA_DIR =DATA_ROOT.joinpath('meta_data')
KS_LIST = io.gen_ks_list(DATA_ROOT)
today = datetime.now().strftime('%Y-%m-%d')
SAVE_PATH = Path(f'../results/latencies_{today}')

def run_one(rec_id):
    """compute the latencies of every cell recorded. Also determine the average spiking within a window 
    of AP onset

    Args:
        rec_id (int): integer that references into the dataframe that has data references
    Returns: pandas dataframe with the fields: 
                cluster_id: cluster id - referenced within a recording (i.e., not unique across all recordings)
                fast_latency: spiking latency with reference to the compound AP of fast trills
                slow_latency: spiking latency with reference to the compound AP of slow trills
                is_trill_active: boolean if the unit has a high enough spike rate in the AP window
                avg_trill_spike_rate: average spikerate of the unit in the AP window
                run:
                gate:
                probe: 
    """    
    # Map into this dataset
    gate_dir = KS_LIST.loc[rec_id]['gate_dir']
    ks_dir = KS_LIST.loc[rec_id]['ks2_dir']
    run,gate,probe = KS_LIST.loc[rec_id][['run','gate','probe']]
    logging.debug(f'{rec_id=}\t{run=}\t{gate=}\t{probe=}')

    # Load this data from disk
    spikes,metrics = io.load_filtered_phy(ks_dir)
    cluster_ids = metrics['cluster_id'].unique()
    APs = io.get_XII_feats(METADATA_DIR,gate_dir.name[6:])
    AP_starts = APs['time(s)'].values
    fast_idx = APs.query('type=="F"').index
    slow_idx = APs.query('type=="S"').index


    # Compute the PETHs, rasters
    fast_AP_peths,fast_AP_raster = brainbox.singlecell.calculate_peths(spikes['ts'].values,spikes['cluster_id'].values,cluster_ids,
                                        align_times=AP_starts[fast_idx],
                                        pre_time=PRE_WIN,
                                        post_time=POST_WIN,
                                        bin_size=BINSIZE,
                                        smoothing=0
                                        )

    slow_AP_peths,slow_AP_raster = brainbox.singlecell.calculate_peths(spikes['ts'].values,spikes['cluster_id'].values,cluster_ids,
                                        align_times=AP_starts[slow_idx],
                                        pre_time=PRE_WIN,
                                        post_time=POST_WIN,
                                        bin_size=BINSIZE,
                                        smoothing=0
                                        )

    # Determine if trill active by percentage of trills (fast/slow) with a spike
    thresh = PCT_TRILL/100.
    pct_spikes_fast = fast_AP_raster.sum(2).mean(0)
    pct_spikes_slow = slow_AP_raster.sum(2).mean(0)
    is_trill = np.logical_or(pct_spikes_fast>thresh,pct_spikes_slow>thresh)

    # Extract latency of PETH peak for fast and slow trills
    fast_latency_samp = fast_AP_peths.means.argmax(1)
    slow_latency_samp = slow_AP_peths.means.argmax(1)

    # Compute the spike rate in the pre-post trill window
    fast_avg_rate = pct_spikes_fast/(PRE_WIN+POST_WIN)
    slow_avg_rate = pct_spikes_fast/(PRE_WIN+POST_WIN)
    avg_rate = np.maximum(fast_avg_rate,slow_avg_rate)

    fast_latency_time = fast_AP_peths.tscale[fast_latency_samp]
    slow_latency_time = slow_AP_peths.tscale[slow_latency_samp]


    #Outputs latencies
    latency_data = pd.DataFrame()
    latency_data['cluster_id'] =cluster_ids
    latency_data['fast_latency'] = fast_latency_time
    latency_data['slow_latency'] = slow_latency_time
    latency_data['is_trill_active'] = is_trill
    latency_data['avg_trill_spike_rate'] = avg_rate
    latency_data[['run','gate','probe']] = run,gate,probe

    return(latency_data)

@click.command()
def compute():
    # Set up error log
    df_errors = pd.DataFrame()
    df_errors['err'] = []

    # Set up logging    
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug(f'Data root set to: {DATA_ROOT.absolute()}')
    logging.info(f'Will save results to {SAVE_PATH.absolute()}')

    # Check for output file 
    SAVE_FN = SAVE_PATH.joinpath('latencies.csv')
    if SAVE_FN.exists():
        logging.warning(f'Save file exists: {SAVE_FN}. Continuing and will overwrite')
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir()
    

    # Get parameters for the computation 
    parameters_fn = SAVE_PATH.joinpath('parameters.json')
    now = datetime.now().isoformat()
    parameters = dict(binsize=BINSIZE,pre_win=PRE_WIN,post_win=POST_WIN,threshold_pct_trill=PCT_TRILL,run_date=now)
    with open(parameters_fn,'w') as fid:
        logging.info('Saving computation parameters to {}')
        json.dump(parameters,fid)
    
    # MAIN LOOP OVER THE DATA
    all_latency = pd.DataFrame()
    for ii in KS_LIST.index:
        try:
            rec_latency = run_one(ii)
        except Exception as e:
            # Log if there was an error - this happens sometimes if there are not enough units
            df_errors.loc[rec_id] =[e]
            df_errors.to_csv(SAVE_PATH.joinpath('err_log.csv'))
            logging.error(e)

        # Concatenate the data
        all_latency = pd.concat([all_latency,rec_latency])
    
@click.command()
@click.argument('latency_fn')
def plot(latency_fn):
    latency_fn = Path(latency_fn)
    save_path = latency_fn.parent

    latencies = pd.read_csv(latency_fn)
    latencies['fast_latency']*=1000
    latencies['slow_latency']*=1000
    data = latencies.query('is_trill_active')

    f = plt.figure()
    ax = f.add_subplot(111)
    ax  = sns.scatterplot(data=data,x='fast_latency',y='slow_latency',hue='avg_trill_spike_rate',size='avg_trill_spike_rate',
                          palette='Purples',
                          ax=ax,
                          alpha=0.5)
    ax.set_xlabel('Fast trill latency (ms)')
    ax.set_ylabel('Slow trill latency (ms)')
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.axvline(0,color='k',ls=':')
    ax.axhline(0,color='k',ls=':')
    ax.plot([-10,10],[-10,10],color='tab:red',ls='--')
    ax.axis('equal')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(save_path.joinpath('latency_fast_v_slow_scatter.png'),dpi=300)
    plt.close('all')



    f = plt.figure()
    sns.boxenplot(data=data,x='probe',y='fast_latency',palette= ['silver','tab:red'],showfliers=False)
    sns.stripplot(data=data,x='probe',y='fast_latency',marker='.',color='black',size=5,alpha=0.5)
    plt.axhline(0,color='k',ls=':')
    plt.ylabel('Latency to CAP in fast trill (ms)')
    sns.despine(bottom=True,trim=True)
    plt.tight_layout()
    plt.savefig(save_path.joinpath('latency_boxen.png'),dpi=300)
    plt.close('all')


    f = plt.figure()
    sns.histplot(data=latencies,x='avg_trill_spike_rate',hue='probe',palette=['silver','tab:red'],bins=np.arange(0,100,5),
                 element='step',common_norm=False,stat='count',log_scale=[False,True],fill=None)
    plt.xlabel('Mean Firing Rate During Trill (sp/s)')
    plt.ylabel('# Neurons')
    plt.tight_layout()
    plt.savefig(save_path.joinpath('average_spike_rates.png'),dpi=300)
    plt.close('all')

    f,ax = plt.subplots(figsize=(8,4),ncols=2,sharey=True,sharex=True)
    sns.histplot(data=data,x='fast_latency',hue='probe',palette=['silver','tab:red'],element='step',fill=None,ax=ax[0])
    sns.histplot(data=data,x='slow_latency',hue='probe',palette=['silver','tab:red'],element='step',fill=None,ax=ax[1])
    ax[0].axvline(0,color='k',ls=':')
    ax[1].axvline(0,color='k',ls=':')
    ax[0].set_xlabel('Latency to CAP fast trill (ms)')
    ax[1].set_xlabel('Latency to CAP slow trill (ms)')
    ax[0].set_ylabel('# Neurons')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(save_path.joinpath('latency_histogram.png'),dpi=300)
    plt.close('all')

@click.group()
def main():
    pass

main.add_command(compute)
main.add_command(plot)

if __name__=='__main__':
    main()
