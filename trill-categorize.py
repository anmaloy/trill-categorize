import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scripts import findpeaks as findpeaks
from src import readSGLX
from src.io import get_ni_analog
import matplotlib.cm as cm

# Parameters and setup
fileName = 'NPX-S2-39'
gate = 3
channel = 0
t_delay = 0
n_clusters = 2  # Number of clusters to classify spikes
time_window = 0.33  # Time window in seconds for density qualification
min_spikes = 5  # Minimum number of spikes required within the time window

# Load data
peaks = findpeaks.NIDAQ(fileName, gate)
binMeta = readSGLX.readMeta(peaks.binPath)
sRate = readSGLX.SampRate(binMeta)
ni_time, ni_data = get_ni_analog(peaks.binPath, channel)

# Define spike detection thresholds
upper_thresh = 1
lower_thresh = ni_data.std() * 2

# Extract peak data
df = peaks.get_peaks(ni_time, ni_data, [lower_thresh, upper_thresh], samplerate=sRate, delay=t_delay, other=False)

# Calculate spike frequency based on time differences
df['time_diff'] = df['time(s)'].diff()
df['frequency'] = 1 / df['time_diff'].fillna(df['time_diff'].mean())

# Extract spike waveforms
window_size = 160
offset = 40
window_array = []

for peak_time in df['time(s)']:
    peak_idx = np.searchsorted(ni_time, peak_time)
    start_idx = max(0, peak_idx - window_size // 2 + offset)
    end_idx = min(len(ni_data), peak_idx + window_size // 2 + offset)
    spike_waveform = ni_data[start_idx:end_idx]

    if len(spike_waveform) < window_size:
        spike_waveform = np.pad(spike_waveform, (0, window_size - len(spike_waveform)), 'constant')

    window_array.append(spike_waveform)

# Normalize and combine features for clustering
window_array = np.array(window_array)
scaler_waveform = StandardScaler()
window_array_scaled = scaler_waveform.fit_transform(window_array)
features = np.column_stack(
    (window_array_scaled, np.square(df['width(len)']), np.square(df['thresholds']), df['frequency']))

# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters)
df['cluster_id'] = kmeans.fit_predict(features)

# Define a consistent color map for clusters
cluster_colors = cm.viridis(np.linspace(0, 1, n_clusters))

# Reduce dimensionality with PCA and plot clusters
pca = PCA(n_components=2)
spikes_pca = pca.fit_transform(features)
plt.scatter(spikes_pca[:, 0], spikes_pca[:, 1], c=[cluster_colors[label] for label in df['cluster_id']])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Spike Clustering')
plt.show()

# Plot a sample of aligned spike traces color-coded by cluster
num_traces = min(len(df), 100)
selected_peaks = np.random.choice(df.index, size=num_traces, replace=False)
time_vector = np.linspace(-window_size, window_size, 2 * window_size + 1)

plt.figure(figsize=(10, 6))
for idx in selected_peaks:
    # 1: Detected peak time 2: Middle of half width
    # peak_time = df.loc[idx, 'time(s)']
    peak_time = df.loc[idx, 'width(mid)']
    cluster_id = df.loc[idx, 'cluster_id']
    peak_idx = np.searchsorted(ni_time, peak_time)
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(len(ni_data), peak_idx + window_size)
    trace = ni_data[start_idx:end_idx]

    if len(trace) < len(time_vector):
        trace = np.pad(trace, (0, len(time_vector) - len(trace)), 'constant')

    plt.plot(time_vector, trace, alpha=0.3, color=cluster_colors[cluster_id])

plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title(f'Overlapping Spike Traces ({num_traces} traces)')
plt.show()

# Apply density-based filtering
filtered_df = peaks.filter_by_spike_density(df.copy(), time_window, min_spikes)

# Example usage
time_window = 0.5  # 500 ms, the maximum time gap between spikes in the same zone
merge_threshold = 0.05  # 50 ms
buffer_size = 0.1  # 100 ms
min_zone_length = 0.3  # 300ms


# Define zones based on spike proximity and cluster ID
zone_df = peaks.define_zones(filtered_df, time_window)

# Adjust the zone boundaries to eliminate gaps and filter out very short zones
adjusted_zone_df = peaks.adjust_zone_boundaries(zone_df, merge_threshold)
adjusted_zone_df = adjusted_zone_df[(adjusted_zone_df['zone_end'] - adjusted_zone_df['zone_start']) >= min_zone_length]
filtered_df = filtered_df[filtered_df.apply(lambda row: any(
    # (row['time(s)'] >= start) and (row['time(s)'] <= end)
    (row['width(mid)'] >= start) and (row['width(mid)'] <= end)
    for start, end in zip(adjusted_zone_df['zone_start'], adjusted_zone_df['zone_end'])), axis=1)]

# Plot the NIDAQ data with the adjusted zones
peaks.spikes_chart(ni_time, ni_data, adjusted_zone_df, filtered_df, n_clusters, upper_thresh, lower_thresh)

# Request user input to name each cluster group
cluster_names = {}
for cluster_id in range(n_clusters):
    cluster_name = input(f"What is cluster {cluster_id}? ")
    cluster_names[cluster_id] = cluster_name

# Apply the names to the DataFrame
adjusted_zone_df['cluster_id'] = adjusted_zone_df['cluster_id'].map(cluster_names)

# Save as CSV
peaks.export_csv(adjusted_zone_df)
