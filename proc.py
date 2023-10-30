
'''
This module contains functions to process and manipulate NPX and auxiliary data
'''
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from tqdm import tqdm
from pathlib import Path

def phy2spikes(ks2_dir):
    '''
    Given a phy directory, create the spikes and metrics dataframes. 
    The spikes dataframe is a pandas dataframe with columns "ts","cell_id","cluster_id". 
    "ts" is the time of a given spike
    "cluster_id" is the KS2 assigned cluster identity associated with that spikes
    "cell_id" is a remapped identifier for the unit a spike came from. cell_id maps one to one onto cluster_id, but takes values 0 to N, where N is the number of unique units

    "cell_id" is used after filtering out bad units
    
    The metrics data frame gives cluster level information (such as probe location, spike amplitude, firing rate, QC metrics etc...)
    '''
    #TODO: implement
    pass
    return(spikes,metrics)


def bin_trains(ts,idx,max_time=None,binsize=0.05,start_time=5):
    '''
    Create a 2D histogram of the number of spikes per bin across all neurons
    bin_trains(ts,idx,n_neurons,binsize=0.05,start_time=5):
    :param ts: Array of all spike times across all neurons
    :param idx: cell index
    :param binsize:
    :param start_time:
    :return: raster,cell_id,bins
    '''
    if max_time is None:
        max_time = np.max(ts)

    # Keep neuron index correct
    n_neurons = np.max(idx)+1
    cell_id = np.arange(n_neurons)
    bins = np.arange(start_time, max_time, binsize)
    raster = np.empty([n_neurons, len(bins)])
    # Remove spikes that happened before the start time
    idx = idx[ts>start_time]
    ts = ts[ts>start_time]
    # Remove spikes that happened after the max time
    idx = idx[ts<max_time]
    ts = ts[ts<max_time]
    # Loop through cells
    for cell in cell_id:
        cell_ts = ts[idx==cell]
        raster[cell, :-1]= np.histogram(cell_ts, bins)[0]
    return(raster,cell_id,bins)


def compute_PCA_decomp(spikes,t0,tf,binsize=0.005,sigma=2):
    '''
    Compute the PCA decomposition on the observed spiking
    :param spikes: A spikes dataframe
    :param t0: first time to fit to
    :param tf: last time to fit to
    :param binsize: in seconds. default = 0.005
    :param sigma: integer.. default=2
    :return:
    '''
    raster, cell_id, bins = bin_trains(spikes['ts'], spikes['cell_id'], binsize=binsize)
    aa = gaussian_filter1d(raster, sigma=sigma, axis=1)
    aa[np.isnan(aa)] = 0
    bb = np.sqrt(aa).T
    bb[np.isnan(bb)] = 0
    bb[np.isinf(bb)] = 0
    s0 = np.searchsorted(bins, t0)
    sf = np.searchsorted(bins, tf)
    pca = PCA(10)
    pca.fit(bb[s0:sf,:])
    X = pca.transform(bb)
    X_bins = bins
    return(X,X_bins,pca)


def remap_time_basis(x,x_t,y_t):
    '''
    Convinience function to map an analog signal x into the time
    basis for another signal y.
    ex: x is phase, y is the PCA decomposition. This allows you to get the phase value for
    each sample in the PCA time
    :param x: Analog signal to change time basis (1D numpy array)
    :param x_t: Time basis of original analog signal (1D numpy array)
    :param y_t: Time basis of target signal (1D numpy array)
    :return: x_mapped - x in the time basis of y (1D numpy array)
    '''
    assert(len(x)==len(x_t))
    idx = np.searchsorted(x_t,y_t)-1
    return(x[idx])


def compute_PCA_speed(X,n = 3):
    '''
    Compute the euclidean speed through PCA space
    :param X: PCA decompositions (2D numpy array: N_timepoints x N_dims)
    :param n: number of dimensions to use (int)
    :return: D - 1D numpy array of PCA speed
    '''
    if X.shape[0]<=X.shape[1]:
        raise Warning(f'Number of timepoints:{X.shape[0]} is fewer than number of dimensions:{X.shape[1]}. Confirm you do not need to transpose the matrix')
    X_s = X[:,:n]
    X_sd = np.diff(X_s,axis=0)
    D = np.concatenate([[0], np.sqrt(np.sum(X_sd ** 2, axis=1))])
    return(D)


def get_eta(x,tvec,ts,pre_win=0.5,post_win=None):
    '''
    Compute the event triggered average, std, sem of a covariate x
    :param x: The analog signal to check against
    :param tvec: the time vector for x
    :param ts: the timestamps (in seconds) of the event
    :param pre_win: the window before to average
    :param post_win: the window after the event to average

    :return:
    '''
    assert(len(tvec)==len(x))
    if post_win is None:
        post_win=pre_win

    dt = tvec[1]-tvec[0]
    samps = np.searchsorted(tvec,ts)
    win_samps_pre = int(pre_win/dt)
    win_samps_post = int(post_win/dt)
    spike_triggered = np.zeros([win_samps_pre+win_samps_post,len(samps)])
    for ii,samp in enumerate(samps):
        if (samp-win_samps_pre)<0:
            continue
        if (samp+win_samps_post)>len(x):
            continue
        spike_triggered[:,ii] = x[samp-win_samps_pre:samp+win_samps_post]

    st_average = np.nanmean(spike_triggered,1)
    st_sem = np.nanstd(spike_triggered,1)/np.sqrt(len(samps))
    st_std = np.nanstd(spike_triggered,1)
    win_t = np.linspace(-pre_win,post_win,(win_samps_pre+win_samps_post))
    lb = st_average-st_sem
    ub = st_average+st_sem
    eta = {'mean':st_average,
           'sem':st_sem,
           'std':st_std,
           't':win_t,
           'lb':lb,
           'ub':ub}

    return(eta)


def parse_annotations(fn):
    '''
    Load and arrange the annotated compound action potential annotations
    '''
    #TODO: implement
    pass