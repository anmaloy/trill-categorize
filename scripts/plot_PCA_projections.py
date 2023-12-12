'''
Go through each unit and compute the PCA plots
'''

import sys
sys.path.append('..')
import click
import pandas as pd
from src import proc,io
from pathlib import Path
import re
from brainbox import singlecell,plot
import matplotlib.pyplot as plt
from brainbox.processing import bincount2D
import numpy as np
import seaborn as sns

# PLOT PARAMETERS
f_trill_color = 'k'
s_trill_color = 'tab:red'

# Get data 
METADATA_DIR =Path(r'Y:\projects\frog\data\meta_data')
DATA_DIR = Path(r"Y:\projects\frog\data")
PCA_BINSIZE=0.010
SIGMA = 1
SAVE_ROOT = Path(r'Y:\projects\frog\xeno_npx\results\2023-12-11_pca')
SAVE_ROOT.mkdir(exist_ok=True)

def prepare_for_pca(spikes,epochs,binsize):
    '''
    Since we only want to fit during the calls,
    mash up a frankenstein matrix of all the concatenated calls
    '''
    n_units = len(spikes['cell_id'].unique())
    assert n_units>10,'Number of units is less than 10 - not suitable for PCA'

    starts = epochs['start analyses (s)']
    ends = epochs['end analyses (s)']
    raster = []
    raster_bins = []
    for start,end in zip(starts,ends):
        temp_raster,cell_id,bins = proc.bin_trains(spikes['ts'],spikes['cell_id'],start_time=start,max_time=end,binsize=binsize)
        raster.append(temp_raster)
        raster_bins.append(bins)
    raster = np.concatenate(raster,1)
    raster_bins = np.concatenate(raster_bins)

    max_spikes_per_bin = int(1/binsize)
    raster[raster>max_spikes_per_bin]= max_spikes_per_bin
    return(raster,cell_id,raster_bins)


def ap_aligned_raster(spikes,compound_aps,n,f_trill_color='k',s_trill_color='tab:red',sort_by_delay=False,f=None,ax=None,xlim=(-100,100)):
    '''
    Plot the spike times of a given unit relative to the compound action potentials
    '''
    this_unit_spike_times = spikes.query('cell_id==@n')['ts'].values
    ap_times = compound_aps['time(s)'].values
    # Create a [n_cap x n_spikes] matrixs where each row is a compound action potential.
    d_time = np.subtract.outer(this_unit_spike_times,ap_times)
    d_time = d_time.T
    color_vector = compound_aps['type'].map({'F':f_trill_color,'S':s_trill_color}).values
    if f is None:
        f = plt.figure(figsize=(2,5))
    if ax is None:
        ax = f.add_subplot(111)

    if sort_by_delay:
        ## This is to order by latency of first spike
        # temp = d_time.copy()
        # temp[temp<0] = np.inf
        # latency = np.min(temp,1)
        # latency_order = np.argsort(latency)
        
        ## This is to order by latency to next CAP
        latency_order = np.argsort(compound_aps['t_to_next'].dropna()).values
        for ii,jj in enumerate(latency_order):
            rr = d_time[jj]
            yy = np.ones(rr.shape) * ii
            ax.plot(rr*1000,yy,'|',color=color_vector[jj])
        ax.set_ylabel('CAP (Sorted)')
    else:
        for ii,rr in enumerate(d_time):
            yy = np.ones(rr.shape) * ii
            ax.plot(rr*1000,yy,'|',color=color_vector[ii])
        ax.set_ylabel('CAP #')

    ax.set_xlim(xlim)    
    ax.axvline(0,color='c')
    
    ax.set_xlabel('Time (ms)')
    
    return(f,ax)

def plot_peth(peth,ii,n_obs,color,f=None,ax=None):
    '''
    Convinience function to plot PETHS'''
    if f is None and ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    if ax is None:
        f.add_subplot(111)
    ax.plot(peth['tscale']*1000,peth['means'][ii],color=color)
    lb = peth['means'][ii] - peth['stds'][ii]/np.sqrt(n_obs)
    ub = peth['means'][ii] + peth['stds'][ii]/np.sqrt(n_obs)
    ax.fill_between(peth['tscale']*1000,lb,ub,color=color,alpha=0.5)
    return(f,ax)


def main(rec_id,ks2_dir,probe_idx):
    probe_id = rec_id+f'_imec{probe_idx}'
    # Make the save location
    save_fn = SAVE_ROOT.joinpath(probe_id+f'_{PCA_BINSIZE*1000:04.0f}ms_pca.png')
    # ========================== #
    # Load data
    # ========================== #
    spikes,metrics = io.load_filtered_phy(ks2_dir)
    epochs = io.load_epochs(gate_dir)
    trills = io.load_trills(gate_dir)

    # ========================== #
    # Compute PCA decomp
    # ========================== #
    raster,cell_id,raster_bins = prepare_for_pca(spikes,epochs,PCA_BINSIZE)
    X,pca = proc.compute_pca_raster(raster,sigma=SIGMA)
    trills_bins = proc.label_time_vector(raster_bins,trills['start'],trills['end'],trills['type'])

    f = plt.figure(figsize=(4,4))
    ax12 = f.add_subplot(223)
    ax13 = f.add_subplot(221,sharex=ax12)
    ax23 = f.add_subplot(224,sharey=ax12)
    cmap = {'none':'silver',
            'F':'k',
            'S':'tab:red'
            }
    ax_leg = f.add_subplot(222)
    ax_leg.axis('off')
    ax_leg.text(0,0.4,'None',color='silver')
    ax_leg.text(0,0.5,'Fast trill',color='k')
    ax_leg.text(0,0.6,'Slow trill',color='tab:red')

    for label in trills_bins['label'].unique():
        idx = trills_bins.query('label==@label').index
        ax12.plot(X[idx,0],X[idx,1],'.',c = cmap[label],alpha=0.5)
        ax13.plot(X[idx,0],X[idx,2],'.',c = cmap[label],alpha=0.5)
        ax23.plot(X[idx,2],X[idx,1],'.',c = cmap[label],alpha=0.5)

    ax12.set_xlabel('PC1')
    ax12.set_ylabel('PC2')
    ax13.set_xlabel('PC1')
    ax13.set_ylabel('PC3')
    ax23.set_xlabel('PC3')
    ax23.set_ylabel('PC2')
    sns.despine()
    plt.tight_layout()
    plt.suptitle(f'{rec_id}')
    plt.savefig(save_fn,dpi=300,transparent=True,bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    df_errors = pd.DataFrame()
    df_errors['err'] = []
    run_dirs = list(DATA_DIR.glob('NPX*'))
    for run_dir in run_dirs:
    
        gate_dirs = io.get_gate_dirs(run_dir)
        for gate_dir in gate_dirs:
            ks2_dirs = io.get_ks_dirs(gate_dir)
            for probe_idx,ks2_dir in enumerate(ks2_dirs):
                
                idx = re.search('NPX*',gate_dir.name).start()
                rec_id = gate_dir.name[idx:]
                print(f'REC_ID:{rec_id}')
                print(f'Probe:{probe_idx}')

                try:
                    main(rec_id,ks2_dir,probe_idx)
                except Exception as e:
                    uid = rec_id+f'_imec{probe_idx}'
                    df_errors.loc[uid] =[e]
                    df_errors.to_csv(SAVE_ROOT.joinpath('err_log_pca.csv'))
                    print(f"Error on {uid}. Logging and continuing")

                    
