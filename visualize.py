import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def map_event_to_time(events_df,t_exog,event_types=None):    
    """Maps a dataframe of events to individual samples in time.
    Given an events dataframe and a time vector, return a vector
    of length len(t_exog) that labels each sample with the given event.

    Args:
        events_df (pandas dataframe): Data frame must have columns "start","end", and "type". Units must be seconds. 
        t_exog (1D numpy vector): Must be an ordered time vector in seconds. 
        event_types (list, optional): _description_. A list of event types to look for in the pandas dataframe. If None, it will map all unique events
    
    Returns:
        t_label (1D numpy vector same length as t_exog, dtype is int)
    """    
    if event_types is None:
        event_types = event_df['type'].unique()
    t_label = np.zeros_like(t_exog,dtype='int')
    for ii,evt in enumerate(event_types):
        sub_events = events_df.query('type==@evt')
        for start,stop in zip(sub_events['start'],sub_events['end']):
            s0,sf = np.searchsorted(t_exog,[start,stop])
            t_label[s0:sf]=ii+1
    return(t_label)

def shade_events(gate_events,ax,alpha=0.3):
    event_types = ['I','F','S','V']
    colors = ['r','g','b','y']
    for ii,evt in enumerate(event_types):
        sub_events = gate_events.query('type==@evt')
        for start,stop in zip(sub_events['start'],sub_events['end']):
            ax.axvspan(start,stop,color=colors[ii],alpha=alpha)