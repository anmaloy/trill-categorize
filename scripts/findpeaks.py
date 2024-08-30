import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, peak_prominences

BASE_DIR = Path(__file__).resolve().parent.parent
matplotlib.use('TkAgg')  # Ensure proper rendering of charts


def get_widths(nidaq_data, peaks, height):
    """
    Calculate the widths, amplitudes, and left/right boundaries of the spikes.

    Parameters:
    - nidaq_data (np.array): The electrophysiology data from the recording.
    - peaks (np.array): Indices of the detected peaks in the data.
    - height (float): The height threshold for detecting the peaks.

    Returns:
    - tuple: Contains arrays for widths, amplitudes, left boundary, and right boundary of the peaks.
    """
    widths, amps, left, right = [], [], [], []

    for peak in peaks:
        lc, rc = None, None
        for i in range(peak):
            left_idx = peak - i
            right_idx = peak + i
            if left_idx >= 0 and nidaq_data[left_idx] < height and lc is None:
                lc = left_idx + ((left_idx + 1) - left_idx) * (height - nidaq_data[left_idx]) / (nidaq_data[left_idx + 1] - nidaq_data[left_idx])
            if right_idx < len(nidaq_data) and nidaq_data[right_idx] < height and rc is None:
                rc = right_idx + (right_idx - (right_idx - 1)) * (height - nidaq_data[right_idx - 1]) / (nidaq_data[right_idx] - nidaq_data[right_idx - 1])
            if lc and rc:
                widths.append(rc - lc)
                amps.append(height)
                left.append(lc)
                right.append(rc)
                break

    return np.array(widths), np.array(amps), np.array(left), np.array(right)


class NIDAQ:
    """
    Class for handling NIDAQ data, detecting spikes, and generating plots.

    Attributes:
    - fileName (str): The name of the file to process.
    - gate (int): The gate number for the data.
    - binPath (Path): The path to the binary file containing NIDAQ data.
    """
    def __init__(self, fileName, gate):
        """
        Initialize the NIDAQ object with file and gate information.

        Parameters:
        - fileName (str): The name of the file to process.
        - gate (int): The gate number for the data.
        """
        self.fileName = fileName
        self.gate = gate
        self.binPath = Path(f'{BASE_DIR}/data/{fileName}/catgt_{fileName}_g{gate}/{fileName}_g{gate}_tcat.nidq.bin')

    def populate_data(self, peaks_time, properties, widths, prominence):
        """
        Create a DataFrame with spike information.

        Parameters:
        - peaks_time (np.array): The time points of the detected peaks.
        - properties (dict): Properties of the detected peaks from scipy's find_peaks.
        - widths (tuple): Width-related data calculated by get_widths function.
        - prominence (np.array): Prominence of each detected peak.

        Returns:
        - pd.DataFrame: DataFrame containing spike information.
        """
        data = {
            'time(s)': peaks_time,
            'amplitude(v)': properties['peak_heights'],
            'width(len)': widths[0],
            'width(amp)': widths[1],
            'width(start)': widths[2],
            'width(finish)': widths[3],
            'prominence': prominence,
            'thresholds': properties['left_thresholds'] + properties['right_thresholds']
        }
        return pd.DataFrame(data)

    def get_peaks(self, nidaq_time, nidaq_data, cutoff, samplerate, distance=10, delay=0, other=False):
        """
        Detect peaks in the NIDAQ data and populate spike information.

        Parameters:
        - nidaq_time (np.array): The time array corresponding to the NIDAQ data.
        - nidaq_data (np.array): The NIDAQ data from which to detect peaks.
        - cutoff (list): A list containing lower and upper thresholds for peak detection.
        - samplerate (float): The sampling rate of the NIDAQ data.
        - distance (int): Minimum distance between consecutive peaks (in ms). Default is 10.
        - delay (float): Delay in seconds to account for. Default is 0.
        - other (bool): Flag to replace NaNs with a specific type if True.

        Returns:
        - pd.DataFrame: DataFrame containing detected peaks and their properties.
        """
        peaks, properties = find_peaks(nidaq_data, height=cutoff, threshold=0, distance=distance * samplerate / 1000)
        widths = get_widths(nidaq_data, peaks, cutoff[0])
        widths = (
            widths[0] / samplerate,
            widths[1],
            widths[2] / samplerate,
            widths[3] / samplerate
        )
        peaks_time = nidaq_time[peaks]
        prominence = peak_prominences(nidaq_data, peaks)[0]
        df = self.populate_data(peaks_time, properties, widths, prominence)
        if other:
            df['type'] = df['type'].replace('nan', 'O')
        return df

    def spikes_chart(self, data, nidaq_time, nidaq_data, upper, lower):
        """
        Generate a plot of NIDAQ data with spikes color-coded by cluster.

        Parameters:
        - data (pd.DataFrame): DataFrame containing spike information and cluster IDs.
        - nidaq_time (np.array): The time array corresponding to the NIDAQ data.
        - nidaq_data (np.array): The NIDAQ data to plot.
        - upper (float): The upper threshold for spike detection.
        - lower (float): The lower threshold for spike detection.
        """
        data = data.replace('nan', pd.NA).dropna(axis=0)
        plt.plot(nidaq_time, nidaq_data, linewidth=0.1)
        unique_clusters = data['cluster_id'].unique()
        num_clusters = len(unique_clusters)
        cmap = plt.get_cmap('viridis', num_clusters)
        legend_patches = []

        for cluster_id in unique_clusters:
            color = cmap(cluster_id / num_clusters)
            cluster_data = data[data['cluster_id'] == cluster_id]
            plt.plot(cluster_data['time(s)'], cluster_data['amplitude(v)'], '.', color=color, alpha=0.75)
            legend_patches.append(mpatches.Patch(color=color, label=f'Cluster {cluster_id}', alpha=0.75))

        plt.ylim(-0.4, 0.9)
        plt.axhline(y=upper, color='red', linewidth=0.3)
        plt.axhline(y=lower, color='red', linewidth=0.3)
        plt.legend(handles=legend_patches, loc='lower right')
        plt.xlabel('Times (s)')
        plt.ylabel('Nerve (v)')
        plt.title('NIDAQ Nerve Spikes')
        plt.show()

        data.to_csv(f'{BASE_DIR}/data/data/{self.fileName}_g{self.gate}-data.csv', index=False)

    def filter_by_spike_density(self, data, time_window, min_spikes):
        """
        Filter spikes based on the density of nearby spikes of the same type.
        If a spike does not meet the density requirement in its cluster, attempt to switch it to another cluster.

        Parameters:
        - data (pd.DataFrame): DataFrame containing spike information and cluster IDs.
        - time_window (float): Time window in seconds to check for nearby spikes.
        - min_spikes (int): Minimum number of spikes required within the time window to retain a spike.

        Returns:
        - pd.DataFrame: Filtered DataFrame with spikes that meet the density requirement.
        """
        keep_indices = []
        switch_indices = []

        for index, row in data.iterrows():
            original_cluster_id = row['cluster_id']
            other_cluster_id = 1 - original_cluster_id  # Assuming binary clusters
            time_s = row['time(s)']

            # Count spikes of the same type within the time window
            nearby_same_type = data[
                (data['cluster_id'] == original_cluster_id) &
                (data['time(s)'] >= (time_s - time_window)) &
                (data['time(s)'] <= (time_s + time_window))
            ]

            if len(nearby_same_type) >= min_spikes:
                keep_indices.append(index)
            else:
                # Check if the spike meets the density criteria for the other cluster
                nearby_other_type = data[
                    (data['cluster_id'] == other_cluster_id) &
                    (data['time(s)'] >= (time_s - time_window)) &
                    (data['time(s)'] <= (time_s + time_window))
                ]

                if len(nearby_other_type) >= min_spikes:
                    switch_indices.append(index)  # Switch to the other cluster

        # Keep spikes that meet the density requirement
        df_filtered = data.loc[keep_indices]

        # Switch spikes that meet the criteria for the other cluster
        df_filtered = pd.concat([df_filtered, data.loc[switch_indices]])
        df_filtered.loc[switch_indices, 'cluster_id'] = 1 - data.loc[switch_indices, 'cluster_id']

        return df_filtered
