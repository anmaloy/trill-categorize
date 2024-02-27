'''
Go through each unit and compute the relationship between fast and slow trills
'''
# TODO: Fix memory error/leak - likely in the ap_aligned_raster code.
# TODO: Clean up code
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
# TODO: add root logic
if sys.platform =='win32':
    FROG_ROOT = Path(r'Y:\projects\frog')
else:
    FROG_ROOT = Path('/active/ramirez_j/ramirezlab/nbush/projects/frog')
    import matplotlib
    matplotlib.use('Agg')

METADATA_DIR =FROG_ROOT.joinpath('data/meta_data')
DATA_ROOT = FROG_ROOT.joinpath('data')
PCA_BINSIZE=0.1
SAVE_ROOT = FROG_ROOT.joinpath('xeno_npx/results/2023-12-11_sc_summary')
SAVE_ROOT.mkdir(exist_ok=True)
def ap_aligned_raster(spikes,compound_aps,n,f_trill_color='k',s_trill_color='tab:red',sort_by_delay=False,f=None,ax=None,xlim=(-50,50)):
    '''
    Plot the spike times of a given unit relative to the compound action potentials
    '''
    n_max = 15 # max number of spikes to keep before and after a CAP
    this_unit_spike_times = spikes.query('cell_id==@n')['ts'].values
    ap_times = compound_aps['time(s)'].values
    # Create a [n_cap x n_spikes] matrixs where each row is a compound action potential.
    d_time = np.subtract.outer(this_unit_spike_times,ap_times)
    d_time = d_time.T
    # Subset to 100 spikes before and after 0
    d_time_matrix = np.empty([ap_times.shape[0],n_max*2])* np.nan
    for jj,rr in enumerate(d_time):
        idx = np.searchsorted(rr,0)
        if idx<n_max:
            continue
        if (idx+n_max)>d_time.shape[1]:
            continue
        d_time_matrix[jj,:] = d_time[jj,idx-n_max:idx+n_max]

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
        color_vector = color_vector[latency_order]
        d_time_matrix = d_time_matrix[latency_order]
        
        yy = np.where(color_vector=='k')[0]
        rr = d_time_matrix[yy,:]
        ax.plot(rr*1000,yy,'|',color='k')

        yy = np.where(color_vector=='tab:red')[0]
        rr = d_time_matrix[yy,:]
        ax.plot(rr*1000,yy,'|',color='tab:red')
        ax.set_ylabel('CAP (Sorted)')
    else:
        yy = np.where(color_vector=='k')[0]
        rr = d_time_matrix[yy,:]
        ax.plot(rr*1000,yy,'|',color='k')

        yy = np.where(color_vector=='tab:red')[0]
        rr = d_time_matrix[yy,:]
        ax.plot(rr*1000,yy,'|',color='tab:red')
        
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

def make_composite_cap_fig(ii,sort_by_delay=False):
    '''
    Make a figure with multiple subpanels that plots the PETH and the CAP aligned spikes'''
    clu_id = int(cell2cluster.loc[ii]['cluster_id'])

    f = plt.figure(figsize=(3,6))
    gs = f.add_gridspec(nrows=10,ncols=1)
    ax_rast = f.add_subplot(gs[2:])
    ax_hist = f.add_subplot(gs[:2],sharex=ax_rast)

    ap_aligned_raster(spikes,compound_aps,ii,sort_by_delay=sort_by_delay,f=f,ax=ax_rast)

    n_slow_cap = slow_ap.shape[0]
    n_fast_cap = fast_ap.shape[0]

    plot_peth(peth_slow,ii,n_slow_cap,color=s_trill_color,ax=ax_hist)
    plot_peth(peth_fast,ii,n_fast_cap,color=f_trill_color,ax=ax_hist)
    ax_hist.set_ylabel('sp/s')

    sns.despine(trim=True)
    f.suptitle(f'Cluster ID:{clu_id}\nCell ID{ii}')
    plt.tight_layout()

def make_composite_trill_fig(ii):
    clu_id = int(cell2cluster.loc[ii]['cluster_id'])

    # Set up figures
    f = plt.figure(figsize=(6,6))
    gs = f.add_gridspec(nrows=6,ncols=2)
    ax_rast_slow = f.add_subplot(gs[2:,0])
    ax_hist_slow = f.add_subplot(gs[:2,0],sharex=ax_rast_slow)
    ax_rast_fast = f.add_subplot(gs[2:,1],sharex=ax_rast_slow)
    ax_hist_fast = f.add_subplot(gs[:2,1],sharex=ax_rast_slow,sharey=ax_hist_slow)

    # Extract this unit
    this_unit_spike_times = spikes.query('cell_id==@ii')['ts'].values
    fr_slow = frs_slow[:,ii,:]
    fr_fast = frs_fast[:,ii,:]

    # Plot trill averages
    plot_peth(peth_slow,ii,n_slow_trills,color=s_trill_color,ax=ax_hist_slow)
    plot_peth(peth_fast,ii,n_fast_trills,color=f_trill_color,ax=ax_hist_fast)

    # Plot per-trill firing rates ordered by trill duration
    ax_rast_slow.pcolormesh(peth_slow['tscale']*1000,np.arange(fr_slow.shape[0]),frs_slow[slow_order,ii,:],cmap='Reds')
    ax_rast_fast.pcolormesh(peth_fast['tscale']*1000,np.arange(fr_fast.shape[0]),frs_fast[fast_order,ii,:],cmap='Greys')

    # Plot duration of the trill
    ax_rast_slow.plot(slow_trills['duration'][slow_order]*1000,np.arange(n_slow_trills),'bo',ms=5,alpha=0.5)
    ax_rast_fast.plot(fast_trills['duration'][fast_order]*1000,np.arange(n_fast_trills),'bo',ms=5,alpha=0.5)
    
    # If the trill is longer than the 2 second window, plot it differently (NEB: this is not super clean, but it works.)
    aa = slow_trills.iloc[slow_order]['duration']>2
    aa = np.where(aa.values)[0]
    ax_rast_slow.plot(np.ones(len(aa))*1950,aa,'wo',mec='b')
    aa = fast_trills.iloc[fast_order]['duration']>2
    aa = np.where(aa.values)[0]
    ax_rast_fast.plot(np.ones(len(aa))*1950,aa,'wo',mec='b')

    # Prettify plots
    ax_hist_slow.set_ylabel('sp/s')
    ax_rast_slow.set_ylabel('Trill (slow duration ordered)')
    ax_rast_fast.set_ylabel('Trill (fast duration ordered)')

    ax_rast_slow.set_xlabel('Time(ms)')
    ax_rast_fast.set_xlabel('Time(ms)')

    ax_rast_slow.axvline(0,color='c',lw=1)
    ax_rast_fast.axvline(0,color='c',lw=1)
    ax_hist_slow.axvline(0,color='c',lw=1)
    ax_hist_fast.axvline(0,color='c',lw=1)

    ax_hist_slow.set_title('Slow Trill')
    ax_hist_fast.set_title('Fast Trill')

    ax_rast_slow.set_xlim(-2000,2000)
    f.suptitle(f'Cluster ID:{clu_id}\nCell ID{ii}')
    
    #TODO: Colorbar
    sns.despine()
    plt.tight_layout()

def plot_all_heatmap():
    raster_temp,_,_ = prepare_for_pca(spikes,epochs,slow_binsize)
    mean_fr = np.mean(raster_temp,1)
    std_fr = np.nanstd(raster_temp,1)

    fr_slow_mean = (np.mean(frs_slow,0) - mean_fr[:,np.newaxis])/std_fr[:,np.newaxis]
    fr_fast_mean = (np.mean(frs_fast,0) - mean_fr[:,np.newaxis])/std_fr[:,np.newaxis]

    order = np.argsort(np.argmax(fr_fast_mean,1))
    # - if you want to remove not-tuned units
    # above_t = np.where(np.max(np.abs(fr_fast_mean),1)>1)[0]
    # order =order[above_t]
    fr_slow_mean_sub = fr_slow_mean[order,:]
    fr_fast_mean_sub = fr_fast_mean[order,:]

    f,ax = plt.subplots(figsize=(6,6),nrows=1,ncols=2,sharex=True,sharey=True)
    ax_slow,ax_fast = ax

    ax_slow.pcolormesh(peth_slow['tscale'],np.arange(fr_slow_mean_sub.shape[0]),fr_slow_mean_sub,cmap='PiYG',vmin=-3,vmax=3)
    cc = ax_fast.pcolormesh(peth_fast['tscale'],np.arange(fr_fast_mean_sub.shape[0]),fr_fast_mean_sub,cmap='PiYG',vmin=-3,vmax=3)

    ax_slow.axvline(0,lw=1,color='c')
    ax_fast.axvline(0,lw=1,color='c')

    ax_slow.axvspan(0,slow_trills['duration'].mean(),color='grey',alpha=0.4)
    ax_fast.axvspan(0,fast_trills['duration'].mean(),color='grey',alpha=0.4)

    ax_fast.set_yticks([0,fr_fast_mean_sub.shape[0]])


    ax_slow.set_ylabel("Unit #")
    ax_fast.set_xlabel('Time (s)')
    ax_slow.set_xlabel('Time (s)')

    ax_fast.set_title('Fast Trill')
    ax_slow.set_title('Slow Trill')
    sns.despine(trim=True)

    f.subplots_adjust(right=0.9)
    cbar_ax = f.add_axes([0.9, 0.25, 0.05, 0.5])
    cbar_ax.set_ylabel('F.R. (Z-score)')
    f.colorbar(cc, cax=cbar_ax)

def plot_CAP_within_trill():
    f = plt.figure()
    t0s,tfs = (fast_trills['start'].values,fast_trills['end'].values)
    for t0,tf in zip(t0s,tfs):
        idx = np.logical_and(compound_aps['time(s)']>t0,compound_aps['time(s)']<tf)
        this_trill = compound_aps.loc[idx]
        xx = np.arange(-this_trill.shape[0],0,1)+1
        plt.plot(xx,this_trill['t_to_next'].values*1000,'.-',color=f_trill_color,alpha=0.7)

    t0s,tfs = (slow_trills['start'].values,slow_trills['end'].values)
    for t0,tf in zip(t0s,tfs):
        idx = np.logical_and(compound_aps['time(s)']>t0,compound_aps['time(s)']<tf)
        this_trill = compound_aps.loc[idx]
        xx = np.arange(-this_trill.shape[0],0,1)+1
        plt.plot(xx,this_trill['t_to_next'].values*1000,'.-',color=s_trill_color,alpha=0.7)
    plt.ylim(0,150)
    plt.xlabel('CAP number(aligned to last CAP')
    plt.ylabel('Time to next CAP (ms)')

def plot_spikes_within_trill(n):
    #TODO: Clean this up - might be the memory error
    t0s,tfs = (fast_trills['start'].values,fast_trills['end'].values)
    compound_aps['trill_num'] = None
    compound_aps['CAP_num_by_trill'] = None
    clu_id = int(cell2cluster.loc[n]['cluster_id'])

    for ii,(t0,tf) in enumerate(zip(t0s,tfs)):
        idx = np.logical_and(compound_aps['time(s)']>t0,compound_aps['time(s)']<tf)
        idx = compound_aps.loc[idx].index
        compound_aps.loc[idx,'trill_num'] =ii
        compound_aps.loc[idx,'CAP_num_by_trill'] =np.arange(-len(idx),0,1)
        
    this_unit_spike_times = spikes.query('cell_id==@n')['ts'].values
    ap_times = compound_aps['time(s)'].values
    # Create a [n_cap x n_spikes] matrixs where each row is a compound action potential.
    d_time = np.subtract.outer(this_unit_spike_times,ap_times)
    d_time = d_time.T
    d_time_pos = np.ma.masked_less(d_time,0)
    d_time_neg = np.ma.masked_greater(d_time,0)

    df_temp = pd.DataFrame()
    df_temp['t_to_first_spike'] =  np.min(d_time_pos,1).data
    df_temp['t_from_last_spike']  = np.max(d_time_neg,1).data
    df_temp['trill_num'] = compound_aps['trill_num']
    df_temp['CAP_num_by_trill'] = compound_aps['CAP_num_by_trill']

    # -------- Line plot ------- #
    f = plt.figure(figsize=(4,4))
    sns.lineplot(data=df_temp,x='CAP_num_by_trill',y='t_to_first_spike',color=f_trill_color)
    sns.lineplot(data=df_temp,x='CAP_num_by_trill',y='t_from_last_spike',color=f_trill_color,ls=':')
    plt.axhline(0,color='c')
    plt.title(f'Cluster ID:{clu_id}\nCell ID{ii}')
    plt.xlabel('CAP #')
    plt.ylabel('Spike latency (s)')
    plt.ylim(-0.5,0.5)
    plt.savefig(save_fn.joinpath(f'spike_latency_{n:03.0f}_line.png'),dpi=300,transparent=True,bbox_inches='tight')
    plt.close('all')
    # -------- Scatters ------- #
    f,ax = plt.subplots(figsize=(3,6),nrows=2,sharex=True)
    sns.scatterplot(data=df_temp,y='CAP_num_by_trill',x='t_to_first_spike',color=f_trill_color,alpha=0.2,ax=ax[0])
    sns.scatterplot(data=df_temp,y='CAP_num_by_trill',x='t_from_last_spike',color=f_trill_color,alpha=0.2,ax=ax[0])
    plt.xlim(-0.1,0.1)
    plt.axvline(0,color='c')

    # ----- Slow trills ----- #
    t0s,tfs = (slow_trills['start'].values,slow_trills['end'].values)
    compound_aps['trill_num'] = None
    compound_aps['CAP_num_by_trill'] = None

    for ii,(t0,tf) in enumerate(zip(t0s,tfs)):
        idx = np.logical_and(compound_aps['time(s)']>t0,compound_aps['time(s)']<tf)
        idx = compound_aps.loc[idx].index
        compound_aps.loc[idx,'trill_num'] =ii
        compound_aps.loc[idx,'CAP_num_by_trill'] =np.arange(-len(idx),0,1)
        
    this_unit_spike_times = spikes.query('cell_id==@n')['ts'].values
    ap_times = compound_aps['time(s)'].values
    # Create a [n_cap x n_spikes] matrixs where each row is a compound action potential.
    d_time = np.subtract.outer(this_unit_spike_times,ap_times)
    d_time = d_time.T
    d_time_pos = np.ma.masked_less(d_time,0)
    d_time_neg = np.ma.masked_greater(d_time,0)

    df_temp = pd.DataFrame()
    df_temp['t_to_first_spike'] =  np.min(d_time_pos,1).data
    df_temp['t_from_last_spike']  = np.max(d_time_neg,1).data
    df_temp['trill_num'] = compound_aps['trill_num']
    df_temp['CAP_num_by_trill'] = compound_aps['CAP_num_by_trill']

    sns.scatterplot(data=df_temp,y='CAP_num_by_trill',x='t_to_first_spike',color=s_trill_color,alpha=0.2,ax=ax[1])
    sns.scatterplot(data=df_temp,y='CAP_num_by_trill',x='t_from_last_spike',color=s_trill_color,alpha=0.2,ax=ax[1])
    plt.xlim(-0.1,0.1)
    ax[0].axvline(0,color='c')
    ax[1].axvline(0,color='c')
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Time (s)')

    sns.despine()
    plt.tight_layout()


if __name__ == '__main__':
    df_errors = pd.DataFrame()
    df_errors['err'] = []
    ks_df = io.gen_ks_list(DATA_ROOT)
    for idx,rec in ks_df.iterrows():
        try:
            rec_id = rec['rec_id']
            ks2_dir = rec['ks2_dir']
            gate_dir = rec['gate_dir']
            print(f'REC_ID: {rec_id}')
            # Make the save location
            save_fn = SAVE_ROOT.joinpath(rec_id)
            save_fn.mkdir(exist_ok=True)
            # ========================== #
            # Load data
            # ========================== #
            spikes,metrics = io.load_filtered_phy(ks2_dir)
            max_t = spikes['ts'].max()
            n_units = len(spikes['cell_id'].unique())
            cell2cluster = spikes.groupby('cell_id').mean()
            compound_aps = io.get_XII_feats(METADATA_DIR,gate_dir.name[6:])
            epochs = io.load_epochs(gate_dir)
            trills = io.load_trills(gate_dir)

            # Seperate to make later code cleaner
            slow_ap = compound_aps.query('type=="S"')
            fast_ap = compound_aps.query('type=="F"')

            slow_trills = trills.query('type=="S"').reset_index(drop=True)
            fast_trills = trills.query('type=="F"').reset_index(drop=True)
            # Get number of trills
            n_slow_trills = slow_trills.shape[0]
            n_fast_trills = fast_trills.shape[0]
            compound_aps['t_to_next'] = np.concatenate([np.diff(compound_aps['time(s)']),[np.nan]])

            # ========================== #
            # Single cell analyses zoom
            # ========================== #
            # COMPUTE
            peth_slow,frs_slow =singlecell.calculate_peths(spikes['ts'],spikes['cell_id'],spikes['cell_id'].unique(),
                slow_ap['time(s)'],pre_time=0.1,post_time=0.1,bin_size=0.001,smoothing=0)
            peth_fast,frs_fast =singlecell.calculate_peths(spikes['ts'],spikes['cell_id'],spikes['cell_id'].unique(),
                fast_ap['time(s)'],pre_time=0.1,post_time=0.1,bin_size=0.001,smoothing=0)

            # PLOT
            for ii in range(n_units):
                make_composite_cap_fig(ii,sort_by_delay=True)
                plt.savefig(save_fn.joinpath(f'sc_CAP_sorted{ii:03.0f}.png'),dpi=300,transparent=True,bbox_inches='tight')
                plt.close('all')

                make_composite_cap_fig(ii,sort_by_delay=False)
                plt.savefig(save_fn.joinpath(f'sc_CAP_{ii:03.0f}.png'),dpi=300,transparent=True,bbox_inches='tight')
                plt.close('all')

            # ========================== #
            # Single cell analyses slower
            # ========================== #
            # COMPUTE
            slow_binsize = 0.05
            peth_slow,frs_slow =singlecell.calculate_peths(spikes['ts'],spikes['cell_id'],spikes['cell_id'].unique(),
                slow_trills['start'],pre_time=2,post_time=2,bin_size=slow_binsize,smoothing=0.025)
            peth_fast,frs_fast =singlecell.calculate_peths(spikes['ts'],spikes['cell_id'],spikes['cell_id'].unique(),
                fast_trills['start'],pre_time=2,post_time=2,bin_size=slow_binsize,smoothing=0.025)

            # Order by trill duration
            slow_order = slow_trills['duration'].sort_values().index
            fast_order = fast_trills['duration'].sort_values().index

            # PLOT
            for ii in range(n_units):
                make_composite_trill_fig(ii)
                plt.savefig(save_fn.joinpath(f'sc_trill_{ii:03.0f}.png'),dpi=300,transparent=True,bbox_inches='tight')
                plt.close('all')

            # ========================== #
            # Plot all units heatmap
            # ========================== #
            #TODO: Fix here
            plot_all_heatmap()
            plt.savefig(save_fn.joinpath(f'trill_averaged_all.png'),dpi=300,transparent=True,bbox_inches='tight')
            plt.close('all')

            # ========================== #
            # Plot time to next CAP as a funciton of AP number in a given trill
            # ========================== #
            plot_CAP_within_trill()
            plt.savefig(save_fn.joinpath(f'CAPnumber_latency_by_trill.png'),dpi=300,transparent=True,bbox_inches='tight')
            plt.close('all')

            # ========================== #
            # Single cell analyses by trill
            # ========================== #
            # TODO: could be made substantially cleaner
            for n in range(n_units):
                plot_spikes_within_trill(n)
                plt.savefig(save_fn.joinpath(f'spike_latency_{n:03.0f}_scatter.png'),dpi=300,transparent=True,bbox_inches='tight')
                plt.close('all')
        except Exception as e:
            df_errors.loc[rec_id] =[e]
            df_errors.to_csv(SAVE_ROOT.joinpath('err_log_sc.csv'))
            print(f"Error on {rec_id}. Logging and continuing")