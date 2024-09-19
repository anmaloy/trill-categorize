import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os
import scipy.signal
import seaborn as sns
import sys

# Data is often used code snippets that allows me to import data from my files.
# Proc is often used processing code. We will use some of it here.
from src import proc, data

from matplotlib import cm
from matplotlib.animation import FuncAnimation


def spikes_view(spikes, metrics):
    pd.DataFrame.to_csv(metrics, 'csv\\metrics.csv', index=False)


def clean_axis(ax):
    ax.set_zlabel('PC3')


def func_names():
          'pca_anim, pca_raster_trills'


class NEB:
    # ---------- PARAMETERES ---------- #
    # Detects the bin file and ks directory in the input folder
    for file in os.listdir('data/input'):
        if file.endswith('.bin'):
            raw_bin_fn = file
            break
    try:
        raw_bin_fn
    except NameError:
        print('No .bin file detected')
        quit()
    if len(os.listdir('data/input/imec0_ks2')) == 0:
        print('ks directory is empty')
        quit()
    # Sets the directory of the ks folder
    ks_dir = 'data/input/imec0_ks2'
    # Sets the folder in which charts will save.
    chart_path = 'data/charts/'
    # Set the path to FFMPEG
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
    # Select whether to display and save example charts used in processing, this is a bulk enable/disable,
    # you may enable individual ones in the code if you wish.
    example_charts = False
    # This is the start and end time for the recording you'll be looking at, select this sample based on the activity
    # you're looking for. Set to 0, 0 to set dynamically throughout the program.
    t0, tf = 144, 212
    # This is the start and end time of an area which you want to further zoom in on, used in some charts to depict
    # cleared spikes. Set to 0, 0 to set dynamically throughout the program.
    zt0, ztf = 182, 188
    # This defines the channel range used in the heatmap
    c0, cf = 0, 350
    # This defines the dots per inch which the charts will be saved in, 1200 is fairly standard in most journals
    dpi = 1200
    # Defines whether a chart will be popped up when generated, or else just saved. it is set to true by default
    # but the bulk generation functions will disable it for afk efficiency.
    show_charts = True
    dirname = os.path.dirname(__file__)
    # --------- END ----------#

    def lfp_ephys(self, channels):
        lfp_channels = channels
        t_start = self.t0
        t_stop = self.tf
        bin = os.path.join(self.dirname, 'data/input/' + self.raw_bin_fn)
        lfp_time, lfp_excerpt = data.get_imec_analog(bin, lfp_channels, t_start, t_stop)
        return lfp_time, lfp_excerpt, lfp_channels

    def assign_channel(self):
        return self.c0, self.cf

    def change_time(self):
        """
        Changes the time which you are looking at
        :return: t0, tf, start and finish time of recording
        """
        while True:
            try:
                self.t0 = float(input('Start Time: '))
            except ValueError:
                print('Time must be an number')
                continue
            else:
                break
        while True:
            try:
                self.tf = float(input('End Time: '))
            except ValueError:
                print('Time must be an number')
                continue
            else:
                break
        return self.t0, self.tf

    def change_zoom(self):
        """
        Changes the zoomed in time which you are looking at
        :return: zt0, ztf, start and finish time of zoomed recording
        """
        while True:
            try:
                self.zt0 = float(input('Start Time: '))
            except ValueError:
                print('Time must be an number')
                continue
            else:
                break
        while True:
            try:
                self.ztf = float(input('End Time: '))
            except ValueError:
                print('Time must be an number')
                continue
            else:
                break

        return self.zt0, self.ztf

    def lfp_sample(self):
        """
        Produces a line chart depicting the LFP output for a small sample subset of channels.
        :returns: Two charts, saved as lfp_overlay_sample.png and lfp_clarified_sample.png in the charts directory.
        """
        print('Generating lfp_sample')
        if self.tf == 0:
            t0, tf = self.change_time()
        if self.ztf == 0:
            zt0, ztf = self.change_zoom()
        else:
            t0, tf, zt0, ztf = self.t0, self.tf, self.zt0, self.ztf
        lfp_time, lfp_excerpt, lfp_channels = self.lfp_ephys([201, 206, 211, 216])
        plt.figure()
        plt.plot(lfp_time, lfp_excerpt.T)
        plt.xlabel('Times (s)')
        plt.ylabel('LFP (V)')
        plt.title('Overlaid LFP Example')
        plt.savefig('charts\\lfp_overlay_sample.png', dpi=self.dpi)

        plt.figure()
        for ii in range(lfp_excerpt.shape[0]):
            plt.plot(lfp_time, lfp_excerpt[ii] + ii / 10, 'k', lw=1)

        plt.axis('off')
        xmin = plt.gca().get_xlim()[0]
        ymin = plt.gca().get_ylim()[0]
        plt.hlines(ymin, xmin, xmin + 5, color='k')
        plt.text(xmin, ymin - 0.01, '5s', horizontalalignment='left', va='top')
        plt.title('Clarified LFP Example')
        plt.savefig('{}lfp_clarified_sample.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def heatmap(self):
        """
        Produces a heatmap depicting the LFP output over several channels.
        :return: lfp_heatmap.png saved to the charts directory.
        """
        print('Generating lfp_heatmap')
        c0, cf = self.assign_channel()
        lfp_time, lfp_excerpt, lfp_channels = self.lfp_ephys(np.arange(c0, cf, 1))
        plt.pcolormesh(lfp_time, lfp_channels, lfp_excerpt, vmin=-0.1, vmax=0.1, cmap="RdBu_r", shading='auto')
        plt.ylabel('Channel ID')
        plt.xlabel("Time (s)")
        cbar = plt.colorbar()
        cbar.set_label('Volts')
        plt.title('Many Channels Heatmap')
        plt.savefig('{}lfp_heatmap.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def spikes_clean(self):
        """
        Gives you all the spikes that Kilosort found, linked to the cluster ID and channel. There is also a "metrics"
        data frame which gives you some summary data about each cluster.
        This keeps only spikes that the pipeline has labelled as "Good". This is usually a pretty OK estimate, but
        you will maybe want to be more picky than that.
        :return: spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf
        """
        spikes, metrics = data.get_concatenated_spikes(self.ks_dir)
        spikes_view(spikes, metrics)
        # # Removes all units that spike less than 100 times in the entire recording.
        spikes = data.filter_by_spikerate(spikes, metrics)
        # # You can pass any expression here that reads into metrics e.g.: "firing_rate>1" or "amplitude_cutoff<0.01"
        # # repeated calls will further filter the dataset.
        spikes, metrics = data.filter_by_metric(spikes, metrics, 'amplitude_cutoff<0.1')
        spikes, metrics = data.filter_by_metric(spikes, metrics, 'isi_viol<2')
        # Re-assigns "cell_id" so that the neurons that we filtered out are gone.
        spikes = data.resort_by_depth(spikes)

        # Map number of spikes per bin to firing rate (in spikes per second)
        binsize = 0.005
        raster, cell_id, bins = proc.bin_trains(spikes['ts'].values, spikes['cell_id'].values, binsize=binsize,
                                                start_time=0)
        raster = raster / binsize
        mean_fr = np.mean(raster, 0)
        neuron_list, pop = data.create_spykes_pop(spikes)
        t0, tf = self.t0, self.tf
        s0, sf = np.searchsorted(bins, [t0, tf])

        return spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf

    def psth_clean(self):
        """
        Auxilliary function cleans data from spikes_clean and further for use in psth chart production.
        :return: smoothed_mean, pks, pk_dat, peak_times, trill_times, units_by_spikerate
        """
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()

        smoothed_mean = scipy.signal.savgol_filter(mean_fr, 3, 1)
        pks, pk_dat = scipy.signal.find_peaks(smoothed_mean, height=5, prominence=3)
        peak_times = bins[pks]

        trill_times = pd.DataFrame(peak_times, columns=['peak_time'])
        units_by_spikerate = spikes.groupby('cell_id').count().sort_values('ts', ascending=False).index

        return smoothed_mean, pks, pk_dat, peak_times, trill_times, units_by_spikerate

    def fr_map(self):
        """
        Produces a map of the fire-rate of each channel over time.
        :return: fr_map.png saved to the charts directory
        """
        print('Generating fr_map')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()

        f = plt.figure(figsize=(5, 5))
        plt.pcolormesh(bins[s0:sf], cell_id, raster[:, s0:sf], vmin=0, vmax=50, cmap='plasma', shading='auto')
        sns.despine()
        plt.xlim(t0, tf)
        plt.xlabel("Time (s)")
        plt.ylabel('Neuron ID')
        plt.yticks([0, np.max(cell_id)])
        cax = plt.colorbar()
        cax.set_label('sp/s')
        plt.title('FR Activity Map')
        plt.savefig('{}fr_map.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def fr_meaned(self):
        """
        Produces a line chart depicting the average firing rate across all channels over time.
        :return: fr_mean.png saved to the charts directory.
        """
        print('Generating fr_meaned')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        f = plt.figure(figsize=(10, 3))
        plt.plot(bins, np.mean(raster, 0), color='k', lw=0.5)
        plt.xlim(self.t0, self.tf)
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Firing Rate (sp/s)')
        sns.despine()
        plt.title('Meaned Firing Rate')
        plt.savefig('{}fr_mean.png'.format(self.chart_path), dpi=self.dpi, bbox_inches='tight')
        if self.show_charts:
            plt.show()
        plt.close()

    def fr_raster(self):
        """
        Produces a combination of fr_map and fr_mean.
        :return: fr_raster.png saved to the charts directory.
        """
        print('Generating fr_raster')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()

        f = plt.figure(figsize=(10, 7))
        gs = f.add_gridspec(nrows=9, ncols=1)

        ax_pop = f.add_subplot(gs[:2, :])
        ax_raster = f.add_subplot(gs[2:, :], sharex=ax_pop)

        ax_pop.plot(bins[s0:sf], mean_fr[s0:sf], color='k', lw=0.5)
        ax_raster.pcolormesh(bins[s0:sf], cell_id, raster[:, s0:sf], vmin=0, vmax=50, cmap='plasma', shading='auto')

        ax_pop.set_ylim(0, np.max(mean_fr))
        ax_pop.set_title('Firing Map & Rate')
        sns.despine()

        ax_pop.set_ylabel('Mean FR (sp/s)')
        ax_raster.set_ylabel('Neuron ID')
        ax_raster.set_yticks([0, np.max(cell_id)])
        plt.tight_layout()
        ax_raster.set_xlabel('Time (s)')
        f.savefig('{}fr_raster.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def fr_marked(self):
        """
        Produces a chart depicting the average fire rate across channels with marked events.
        :return: fr_marked.png saved to the charts directory.
        """
        print('Generating fr_marked')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()

        smoothed_mean = scipy.signal.savgol_filter(mean_fr, 3, 1)
        pks, pk_dat = scipy.signal.find_peaks(smoothed_mean, height=5, prominence=3)
        peak_times = bins[pks]

        f, ax = plt.subplots(nrows=1, ncols=2, sharey='all')
        f.suptitle('Marked Mean Spikes')
        ax[0].plot(bins, smoothed_mean, color='k', lw=0.5)
        ax[0].plot(peak_times, smoothed_mean[pks], 'ro')
        ax[0].set_xlim(self.t0, self.tf)
        ax[1].plot(bins, smoothed_mean, color='k', lw=0.5)
        ax[1].plot(bins[pks], smoothed_mean[pks], 'ro')
        ax[1].set_xlim(self.zt0, self.ztf)
        ax[0].set_ylabel('Mean F.R. (sp/s)')
        f.savefig('{}fr_marked.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def psth_populate(self):
        """
        Populates a series of charts for each channel for the period of time selected, shows activity.
        :return: xx_peak_aligned_psth.png saved in the charts/single_cell_psth directory
        """
        print('Generating psth charts')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        smoothed_mean, pks, pk_dat, peak_times, trill_times, units_by_spikerate = self.psth_clean()

        p_save = 'charts\\single_cell_psth'
        if not os.path.exists(p_save):
            os.makedirs(p_save)

        for ii, unit in enumerate(units_by_spikerate):
            plt.figure()
            neuron_list[unit].get_psth(event='peak_time', df=trill_times, binsize=25, window=[-1000, 1000])
            plt.ylim(0, plt.gca().get_ylim()[1])
            plt.savefig(os.path.join(p_save, f'{ii:03d}_peak_aligned_psth.png'), dpi=self.dpi)
            plt.close('all')

    def psth_chart(self):
        """
        Depicts a heatmap for psth.
        :return: psth_heatmap.png saved in the charts directory
        """
        print('Generating psth_chart')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        smoothed_mean, pks, pk_dat, peak_times, trill_times, units_by_spikerate = self.psth_clean()
        psth = pop.get_all_psth(event='peak_time', df=trill_times, window=[-1000, 1000], binsize=1, plot=False)

        plt.figure(figsize=(10, 10))
        pop.plot_heat_map(psth)
        plt.title('PSTH Heatmap')
        plt.savefig('{}psth_heatmap.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def psth_colorbar(self):
        """
        Depicts a colorbar of psth spikes.
        :return: psth_colorbar.png saved in the charts directory.
        """
        print('Generating psth_colorbar')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        smoothed_mean, pks, pk_dat, peak_times, trill_times, units_by_spikerate = self.psth_clean()
        psth = pop.get_all_psth(event='peak_time', df=trill_times, window=[-1000, 1000], binsize=1, plot=False)

        dat_mat = psth['data'][0]
        dat_mat = np.sqrt(dat_mat)
        order = np.argsort(np.argmax(dat_mat, 1))[::-1]

        plt.pcolormesh(dat_mat[order, :])
        plt.colorbar()
        plt.title('Spikes Colorbar')
        plt.savefig('{}psth_colorbar.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_spikes(self):
        """
        Depicts the frequency of spikes (?) meaned across all channels.
        :return: meaned_spikes_aoi.png saved to the charts directory
        """
        print('Generating pca_spikes')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        raster, cell_id, bins = proc.bin_trains(spikes['ts'].values, spikes['cell_id'].values, binsize=0.025,
                                                start_time=0)
        mean_fr = np.mean(raster, 0)
        plt.plot(bins, mean_fr)
        plt.xlim(self.t0, self.tf)
        plt.title('Meaned Spikes - AOI')
        plt.savefig('{}meaned_spikes_aoi.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_var(self):
        """
        Depicts the cumulative variance and dimensions of freedom of the data.
        :return: pca_cumvar.png saved to the charts directory
        """
        print('Generating pca_var')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.0025
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)

        plt.plot(np.arange(X.shape[1]) + 1, np.cumsum(pca.explained_variance_ratio_), 'k.-')
        plt.ylim(0, 1)
        plt.ylabel('Cumulative Variance Explained')
        plt.xlabel('# Dimensions included')
        plt.title('Cumulative Variance by Dimensionality')
        plt.savefig('{}cumvar_by_dimension.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_fr(self):
        """
        Depicts the spike chart (top) and the fourier decomposition pca analysis (bottom) of the waveform.
        :return: pca_mean_fr.png saved to the charts directory
        """
        print('Generating pca_fr')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.0025
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)

        f, ax = plt.subplots(nrows=2, sharex='all')
        ax[0].plot(bins, mean_fr, color='k', lw=1)
        ax[1].plot(X_t, X[:, :4], lw=0.5)
        plt.xlim(self.t0, self.tf)
        ax[0].set_ylabel('Mean FR')
        ax[1].set_ylabel('PCA')
        ax[1].set_xlabel('Time (s)')
        plt.title('Mean Fire Rate PCA')
        plt.savefig('{}pca_mean_fr.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_fr_zoom(self):
        """
        Depicts the spike chart (top) and the fourier decomposition pca analysis (bottom) of the waveform for the
        defined zoomed time.
        :return: pca_mean_fr_zoom.png saved to the charts directory
        """
        print('Generating pca_fr_zoom')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.0025
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)

        f, ax = plt.subplots(nrows=2, sharex='all', figsize=(10, 10))
        f.suptitle('Mean Fire Rate PCA')
        ax[0].plot(bins, mean_fr, color='k', lw=1)
        ax[1].plot(X_t, X[:, :4], lw=0.5)
        plt.xlim(self.zt0, self.ztf)
        ax[1].legend(['PC1', 'PC2', 'PC3'])
        ax[0].set_ylabel('Mean FR')
        ax[1].set_ylabel('PCA')
        ax[1].set_xlabel('Time (s)')
        plt.savefig('{}pca_mean_fr_zoom.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_spikes_3d(self):
        """
        Depicts 3d chart of the spikes intensity across channels.
        The image is saved in the default position, if you want additional angles you may rotate it in the pop-up and
        save manually.
        :return: pca_spikes_3d.png saved to the charts directory
        """
        print('Generating pca_spikes_3d')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.0025
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)

        fr2 = proc.remap_time_basis(mean_fr, bins, X_t)
        s0, sf = np.searchsorted(X_t, [self.t0, self.tf])
        f = plt.figure()
        ax = f.add_subplot(projection='3d')
        ax.scatter(X[s0:sf, 0], X[s0:sf, 1], X[s0:sf, 2], c=fr2[s0:sf], cmap='plasma', alpha=0.5)
        plt.title('Spikes 3D')
        plt.savefig('{}pca_spikes_3d.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_trills_3d(self):
        """
        Depicts 3d chart of the mean fire rate across channels.
        The images are saved in the default position and in a 90 degree offset, if you want additional angles you may
        rotate it in the pop-up and save manually.
        :return: pca_trills_3d.png saved to the charts directory
        """
        print('Generating pca_trills_3d')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.0025
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)
        fr2 = proc.remap_time_basis(mean_fr, bins, X_t)

        dt = 0.01
        f = plt.figure(figsize=(4, 3), dpi=300)
        ax = f.add_subplot(111, projection='3d')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(fr2), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        while t0 < self.tf:
            s0, sf = np.searchsorted(X_t, [t0, t0 + dt])
            sf += 1
            ax.plot(X[s0:sf, 0], X[s0:sf, 1], X[s0:sf, 2], color=mapper.to_rgba(fr2[sf]), alpha=0.3, lw=0.5)
            t0 += dt
        clean_axis(ax)
        cax = plt.colorbar(mapper, ax=ax, shrink=0.75)
        cax.set_label('Mean F.R. (sp/s/neuron)')
        plt.tight_layout()
        ax.view_init(0, 0)
        plt.savefig('{}pca_trills_3d_v1.png'.format(self.chart_path), dpi=self.dpi)
        ax.view_init(90, 0)
        plt.savefig('{}pca_trills_3d_v2.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_trills_simp(self):
        """
        Depicts a simplified visualization of the mean fire rate across channels.
        :return: pca_trills_simp.png saved to the charts directory
        """
        print('Generating pca_trills_simp')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.0025
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)

        s0, sf = np.searchsorted(X_t, [self.t0, self.tf])
        plt.plot(X[s0:sf, 0], X[s0:sf, 1], alpha=1, color='k')
        plt.title('Simplified PCA Line')
        plt.savefig('{}pca_trills_simp.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_trills(self):
        """
        Depicts a line chart of determined trills.
        :return: pca_trills.png saved to the charts directory
        """
        print('Generating pca_trills')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.0025
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)
        fr2 = proc.remap_time_basis(mean_fr, bins, X_t)

        s0, sf = np.searchsorted(X_t, [self.t0, self.tf])
        plt.plot(fr2[s0:sf])
        plt.title("Trills")
        plt.savefig('{}pca_trills.png'.format(self.chart_path), dpi=self.dpi)
        if self.show_charts:
            plt.show()
        plt.close()

    def pca_anim(self):
        """
        Depicts a rotating construction of the pca_trills_3d chart.
        Depending on the size of the dataset fed into it, could take upwards of an hour to generate.
        :return: pca_trills_animation.mp4 saved to the charts directory
        """
        print('Generating pca_anim')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.0025
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)
        fr2 = proc.remap_time_basis(mean_fr, bins, X_t)

        dt = 0.01
        t0 = self.t0
        f = plt.figure(figsize=(4, 3), dpi=300)
        ax = f.add_subplot(111, projection='3d')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(fr2), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        while t0 < self.tf:
            s0, sf = np.searchsorted(X_t, [t0, t0 + dt])
            sf += 1
            ax.plot(X[s0:sf, 0], X[s0:sf, 1], X[s0:sf, 2], color=mapper.to_rgba(fr2[sf]), alpha=0.3, lw=0.5)
            t0 += dt
        clean_axis(ax)

        def init():
            ax.view_init(0, 0)
            return ax,

        def update(frame):
            ax.view_init(frame / 2, frame)
            return ax,

        ani = FuncAnimation(f, update, frames=np.linspace(0, 720, 720), init_func=init, repeat=False)
        Writer = animation.FFMpegWriter(fps=30)
        ani.save('{}pca_trills_animation.mp4'.format(self.chart_path), writer=Writer)
        plt.close()

    def pca_raster_trills(self):
        """
        Puts together the mean fire rate, spikes map, and 3d depiction into an animation.
        :return: pca_raster_trills.mp4 saved in the charts directory
        """
        print('Generating pca_raster_trills')
        spikes, metrics, raster, cell_id, bins, mean_fr, neuron_list, pop, t0, tf, s0, sf = self.spikes_clean()
        binsize = 0.005
        mean_fr = np.mean(raster, 0)
        X, X_t, pca = proc.compute_PCA_decomp(spikes, self.t0, self.tf, binsize=binsize)
        Xbins = X_t
        fr2 = proc.remap_time_basis(mean_fr, bins, X_t)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(fr2), clip=False)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        mapper.to_rgba(0)

        dum_raster, cell_id, bins = proc.bin_trains(spikes['ts'].values, spikes['cell_id'].values, binsize=0.005,
                                                    start_time=0)
        mean_fr = scipy.signal.savgol_filter(np.mean(dum_raster, 0), 3, 1)

        # Trills
        t0 = self.zt0
        tf = self.ztf

        f = plt.figure(figsize=(3, 8), dpi=600)
        gs = f.add_gridspec(nrows=15, ncols=1)

        ax = f.add_subplot(gs[8:, :], projection='3d')
        ax_raster = f.add_subplot(gs[1:8, :])
        ax_dia = f.add_subplot(gs[0, :], sharex=ax_raster)

        # Plot trail================================
        s0, sf = np.searchsorted(X_t, [t0 - .01, t0])
        pp = fr2[s0]
        cc = mapper.to_rgba(pp)

        trail1, = ax.plot(X[s0:sf, 0], X[s0:sf, 1], X[s0:sf, 2], color=cc, lw=2)
        s0, sf = np.searchsorted(X_t, [t0 - .2, t0 - 0.1])
        trail2, = ax.plot(X[s0:sf, 0], X[s0:sf, 1], X[s0:sf, 2], color='k', lw=0.4, alpha=0.5)

        clean_axis(ax)
        ax.view_init(80, 220)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        # Plot Raster ============================
        s0, sf = np.searchsorted(bins, [t0 - 1, t0 + 1])
        quad = ax_raster.pcolormesh(bins[s0:sf] - t0, cell_id, dum_raster[:, s0:sf], cmap='Greys', vmax=1,
                                    shading='auto')
        ax_raster.axvline(0, color='w', ls=':', lw=2)
        sns.despine(left=True, trim=True)
        ax_raster.set_yticks([])
        ax_raster.set_ylabel('')
        ax_raster.set_xlabel('Time (s)')
        ax_raster.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_raster.set_xlim([-1.1, 1.1])
        ax_raster.vlines(-1.05, 0, 25, color='k', lw=3)
        ax_raster.text(-1.08, 0, '25 neurons', fontsize=8, color='k', rotation=90, ha='right', va='bottom')

        # Plot FR =================================
        s0, sf = np.searchsorted(bins, [t0 - 1, t0 + 1])
        dd, = ax_dia.plot(bins[s0:sf] - t0, mean_fr[s0:sf], lw=0.5, color='k')
        ax_dia.axis('off')
        ax_dia.axvline(0, color='k', ls=':', lw=2)
        s0, sf = np.searchsorted(bins, [t0, tf])
        ax_dia.set_ylim(0, np.max(mean_fr[s0:sf]) * 1.5)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)

        def init():
            return trail1,

        def update(frames):
            s0, sf = np.searchsorted(Xbins, [t0 + frames - .05, t0 + frames])
            trail1.set_data(X[s0:sf, 0], X[s0:sf, 1])
            trail1.set_3d_properties(X[s0:sf, 2])

            pp = fr2[sf]
            cc = mapper.to_rgba(pp)
            trail1.set_color(cc)

            s0, sf = np.searchsorted(Xbins, [t0 - .1, t0 + frames - .05])
            trail2.set_data(X[s0:sf, 0], X[s0:sf, 1])
            trail2.set_3d_properties(X[s0:sf, 2])
            trail2.set_alpha(0.2)

            s0, sf = np.searchsorted(bins, [t0 + frames - 1, t0 + frames + 1])
            C = dum_raster[:, s0:sf]
            quad.set_array(C.ravel())

            s0, sf = np.searchsorted(bins, [t0 + frames - 1, t0 + frames + 1])
            dd.set_data(bins[s0:sf] - t0 - frames, mean_fr[s0:sf])

            if frames > .05:
                ax.view_init(ax.elev - .1, ax.azim - .1)

            elif frames > 3:
                pass

            return trail1, trail2, quad, dd

        ani = FuncAnimation(f, update, frames=np.arange(0, tf - t0, 1 / 240), init_func=init, blit=True)
        Writer = animation.FFMpegWriter(fps=30)
        ani.save('{}pca_raster_trills.mp4'.format(self.chart_path), writer=Writer)
        plt.close()

    def fr_all(self):
        """
        Generates all charts concerning the initial dataset. fr_map, fr_meaned, fr_raster, fr_marked, psth_chart,
        psth_colorbar. Use the help function for info on each of them.
        """
        self.show_charts = False
        # self.lfp_sample()
        # self.heatmap()
        self.fr_map()
        self.fr_meaned()
        self.fr_raster()
        self.fr_marked()
        self.psth_chart()
        self.psth_colorbar()

    def pca_all(self):
        """
        Generates all charts concerning the PCA analysis. pca_spikes, pca_var, pca_fr, pca_fr_zoom, pca_spikes_3d,
        pca_trills_3d, pca_trills_simp, pca_trills, pca_anim, pca_raster_trills. Use the help function for info on them.
        """
        self.show_charts = False
        self.pca_spikes()
        self.pca_var()
        self.pca_fr()
        self.pca_fr_zoom()
        self.pca_spikes_3d()
        self.pca_trills_3d()
        self.pca_trills_simp()
        self.pca_trills()
        self.pca_anim()
        self.pca_raster_trills()

    def charts_all(self):
        """
        Generates all charts except for those in psth_populate
        """
        self.fr_all()
        self.pca_all()


test = NEB()
func_dict = {'change_time': test.change_time, 'change_zoom': test.change_zoom, 'assign_channel': test.assign_channel,
             'charts_all': test.charts_all, 'fr_all': test.fr_all, 'fr_map': test.fr_map, 'fr_meaned': test.fr_meaned,
             'fr_raster': test.fr_raster, 'fr_marked': test.fr_marked, 'psth_chart': test.psth_chart, 'psth_colorbar':
                 test.psth_colorbar, 'psth_populate': test.psth_populate, 'pca_all': test.pca_all, 'pca_spikes':
                 test.pca_spikes, 'pca_var': test.pca_var, 'pca_fr': test.pca_fr, 'pca_fr_zoom': test.pca_fr_zoom,
             'pca_spikes_3d': test.pca_spikes_3d, 'pca_trills': test.pca_trills, 'pca_trills_3d': test.pca_trills_3d,
             'pca_trills_simp': test.pca_trills_simp, 'pca_anim': test.pca_anim, 'pca_raster_trills':
                 test.pca_raster_trills, 'heatmap': test.heatmap, 'lfp_sample': test.lfp_sample, 'func_names':
                 func_names}
func_names()
while True:
    func = input('Function Name: ')
    if 'help' in func:
        try:
            help_name = str.split(func, ' ')
            help(func_dict[help_name[1]])
            continue
        except KeyError:
            print('Not a function')
            continue
    elif 'exit' in func:
        break
    else:
        try:
            func_dict[func]()
            continue
        except KeyError:
            print('Not a function')
            continue
