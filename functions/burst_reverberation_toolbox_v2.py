import numpy as np
from scipy.stats import norm
from scipy.signal import find_peaks
from scipy.signal import convolve
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from scipy.stats import skew
from scipy import signal
from unidip import UniDip
import pandas as pd
from scipy.stats import mode
from scipy.stats import skew
from itertools import chain
import neo
import quantities as pq
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.spike_train_synchrony import spike_contrast
from elephant.spike_train_correlation import spike_time_tiling_coefficient
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
from scipy.stats import mode


def generate_spikematrix(spiketrain, fs=12500, duration=300):
    '''
    :param spiketrain: Takes spikes times from a single channel
    :param fs: Sampling frequency in Hz
    :param duration: Duration of recording in seconds
    :return: spikematrix: Binary matrix containing spikes
    '''
    spiketimes = np.array(spiketrain)
    spiketimes = spiketimes[spiketimes <= duration]  # Ensure recording is desired length
    spikematrix = [0] * (duration * fs)  # Generate empty spike matrix with appropriate number of bins
    for spike in spiketimes:
        spikematrix[int(spike * fs)] = 1
    return spikematrix

def generate_gaussian_kernel(sigma=0.1, fs=12500):
    '''
    :param sigma: Width of kernel
    :param fs: Sampling frequency in Hz
    :return: Gaussian kernel
    '''
    edges = np.arange(-3*sigma, 3*sigma, 1/fs)
    kernel = norm.pdf(edges, 0, sigma)
    kernel = kernel*1/fs
    return kernel

def generate_sdf(spikematrix, gaussian_kernel):
    '''
    :param spikematrix: Binary matrix containing spikes
    :param gaussian_kernel: Gaussian kernel to be convolved with spikematrix
    :return: sdf: Continuous timeseries representing probability distribution of activity
    '''
    sdf_tmp = convolve(spikematrix, gaussian_kernel)
    sdf = sdf_tmp[int((len(sdf_tmp)-len(spikematrix))/2):int(len(sdf_tmp)-((len(sdf_tmp)-len(spikematrix))/2))]
    sdf = sdf/max(gaussian_kernel)
    return sdf

def detect_burst_peaks(sdf, height = 0.5, prom=0.5, fs=12500):
    burst_peaks, _ = find_peaks(sdf, height=height, prominence=prom, distance=5)
    return burst_peaks/fs

def find_rmax_unidip(burst_peaks, steps=5):
    kernel_width = 0.30 # E.g. length of kernel/ sampling frequency
    x = np.diff(burst_peaks)
    dat = np.msort(x)
    rmax = 0
    for alpha in (np.linspace(0.15,0.95, steps)):
        intervals = UniDip(dat, alpha=alpha, mrg_dst=0).run()
        if len(intervals) == 1: # Unimodal
            rmax = 0.0001
        if len(intervals) > 1: # Bimodal or more
            for peak in intervals:
                if (dat[peak[1]] <= 2) & (dat[peak[1]] > rmax):
                    rmax = dat[peak[1]]
                    print(f"New rmax: {rmax}")
    rmax = rmax + kernel_width
    return dat, rmax

def find_rmax_hist_fixed_step(burst_peaks):
    kernel_width = 0.30
    x = np.diff(burst_peaks)
    rmax = 0
    if (np.median(x) < 2.5):
        for size in (np.arange(75, 25, -10)):
            counts, binEdges = np.histogram(x, bins=size)
            first_min = np.where(counts==0)[0][0]
            rmax_tmp = binEdges[first_min]
            if (rmax_tmp > rmax) & (rmax_tmp < 5):
                print(f"New RMAX: {rmax_tmp} using size: {size}")
                rmax = rmax_tmp
    return rmax

def find_rmax_hist(burst_peaks, prime_burst_peaks, steps = 1):
    # Finds minima of IBI histogram based on different histogram bin size
    x = np.diff(burst_peaks)
    prime_x = np.diff(prime_burst_peaks)
    max_num_bins = int(len(x)/1.5)
    rmax = 0
    r_tmp = []
    s_tmp = []
    rmax_tmp = 0
    if (np.median(x) < np.median(prime_x)):
        for size in (np.arange(max_num_bins, 5, -steps)):
            counts, binEdges = np.histogram(x, bins=size)
            if len(np.where(counts==0)[0]) > 0:
                first_min = np.where(counts==0)[0][0]
                rmax_tmp = binEdges[first_min]
                r_tmp.append(rmax_tmp)
                s_tmp.append(size)
                if (rmax_tmp > rmax) & (rmax_tmp < (mode(list(map(int,prime_x)))[0]*0.9)):
                    print(f"New RMAX: {rmax_tmp} using size: {size}")
                    rmax = rmax_tmp
    #return rmax
    return np.median(r_tmp)

def find_rmax_hist_subtractive(burst_peaks, prime_burst_peaks, steps = 1):
    # Modification of the find_rmax function that looks at the left skew minima of the subtractive histogram between
    # prime and non-prime burst peaks
    x = np.diff(burst_peaks)
    prime_x = np.diff(prime_burst_peaks)
    nonprime = [tmp for tmp in burst_peaks if tmp not in prime_burst_peaks]
    nonprime_x = np.diff(nonprime)

    rmax = 0
    r_tmp = []
    s_tmp = []
    rmax_tmp = 0
    if (np.median(x) < np.median(prime_x)):
        for size in (np.arange(15, 5, -steps)):
            counts, binEdges = np.histogram(nonprime_x, bins=size)
            startingPoint = np.abs(binEdges - np.median(nonprime_x)).argmin()
            if len(np.where(counts==0)[0]) > 0:
                first_min = np.where(counts==0)[0][-1]
                rmax_tmp = binEdges[first_min]
                r_tmp.append(rmax_tmp)
                s_tmp.append(size)
                if (rmax_tmp > rmax):
                    print(f"New RMAX: {rmax_tmp} using size: {size}")
                    rmax = rmax_tmp
    #return rmax
    return np.median(r_tmp)

def burst_skewness_reverberating(burst_peaks):
    if skew(np.diff(burst_peaks)) > 0:
        return True
    else:
        return False

def find_burst_border(value, sdf, threshold, fs=12500):
    value = (value*fs).astype(int)
    ind_below_threshold = np.where(sdf < threshold)[0]
    try:
        sb_start = ind_below_threshold[ind_below_threshold < value].max()
    except:
        if ((value/fs) < 2):
            sb_start = 0
        else:
            pass
    try:
        sb_end = ind_below_threshold[ind_below_threshold > value].min()
    except:
         sb_end = len(sdf)-1
    return (sb_start / fs, sb_end / fs)

def detect_burst_borders(burst_peaks, sdf, threshold, fs=12500):
    tmp = []
    for peak in burst_peaks:
        #try:
            border = find_burst_border(peak, sdf, threshold, fs)
            tmp.append(border)
        #except:
         #   print(f"One of the bursts had an issue.")
          #  pass
    return tmp

def calculate_isi(raster):
    isi = []
    for channel in raster:
        isi.append(np.diff(np.array(channel)))
    isi_flat = np.array(list(chain.from_iterable(isi)))
    return isi_flat

def above_noise(burst_peaks, sdf, fs=12500):
    min_burst_peak = np.nanmedian(sdf[(burst_peaks*fs).astype(int)])
    signal_threshold = min_burst_peak*0.5
    return signal_threshold

def above_noise2(burst_peaks, sdf, fs=12500):
    min_burst_peak = min(sdf[(burst_peaks*fs).astype(int)])
    signal_threshold = min_burst_peak*0.8
    return signal_threshold

def detect_reverberation(burst_peaks, sdf, rmax):
    i = 1
    kernel_width = 0.5
    in_super_burst = False
    sb_end = []
    sb_start = [(burst_peaks[0])-kernel_width/2]
    num_reverbs = []
    r = 0
    while i < len( burst_peaks):
        if ((burst_peaks[i] - burst_peaks[i - 1]) < rmax):
            if in_super_burst:
                r += 1
            in_super_burst = True
        elif (burst_peaks[i] - burst_peaks[i - 1]) > rmax:
            if in_super_burst:
                r += 1
            in_super_burst = False
            sb_end.append((burst_peaks[i - 1]) + kernel_width/2)
            sb_start.append((burst_peaks[i]) - kernel_width/2)
            num_reverbs.append(r)
            r = 0
        i += 1
    i -= 1
    if in_super_burst:
        r += 1
    sb_end.append((burst_peaks[i]) + kernel_width/2)
    num_reverbs.append(r)
    return np.array(sb_start), np.array(sb_end), np.array(num_reverbs)

def detect_reverberations_merge(burst_borders, burst_peaks, rmax):
    sb_start = [burst_borders[0][0]]
    sb_end = []
    num_reverbs = []
    r = 0
    in_super_burst = False
    for i in range(1,len(burst_borders)):
        if (burst_borders[i][0]-burst_borders[i-1][1]) <= rmax:
            if in_super_burst:
                r += 1
            in_super_burst = True
        elif (burst_borders[i][0]-burst_borders[i-1][1]) > rmax:
            if in_super_burst:
                r += 1
            in_super_burst = False
            sb_end.append(burst_borders[i-1][1])
            sb_start.append(burst_borders[i][0])
            num_reverbs.append(r)
            r = 0
        i += 1
    i -= 1
    if in_super_burst:
        r +=1
    sb_end.append(burst_borders[i][1])
    num_reverbs.append(r)
    return sb_start, sb_end, num_reverbs

def detect_reverberations_merge2(burst_borders, burst_peaks, prime_burst_peaks, rmax):
    sb_start = []
    sb_end = []
    num_reverbs = []
    r = 0
    in_super_burst = False
    initial_burst = False
    for i in range(0,len(burst_borders)-1):
        if (burst_peaks[i] in prime_burst_peaks) & (initial_burst == False):
            #print(f"Initial burst peak at {burst_peaks[i]}")
            initial_burst = True
            sb_start.append(burst_borders[i][0])
        if ((burst_borders[i+1][0]-burst_borders[i][1]) <= rmax) & (initial_burst==True):
                if in_super_burst:
                    r+=1
                in_super_burst=True
        elif ((burst_borders[i+1][0]-burst_borders[i][1]) > rmax) & (initial_burst==True):
            if in_super_burst:
                r += 1
            in_super_burst = False
            initial_burst = False
            sb_end.append(burst_borders[i][1])
            num_reverbs.append(r)
            r = 0
        i += 1
    if (burst_peaks[len(burst_borders)-1] in prime_burst_peaks):
        sb_start.append(burst_borders[i][0])
        sb_end.append(burst_borders[i][1])
    else:
        sb_end.append(burst_borders[i][1])
    num_reverbs.append(r)
    if len(sb_start) == len(sb_end):
        return sb_start, sb_end, num_reverbs
    else:
        if len(sb_start) < len(sb_end):
            return (sb_start, sb_end[:len(sb_start)], num_reverbs[:len(sb_start)])
        elif len(sb_start) > len(sb_end):
            return (sb_start[1:], sb_end, num_reverbs[1:])

def detect_reverberations_merge3(burst_borders, burst_peaks, prime_burst_peaks, rmax):
    # I think this fixes cases where IBI is within rmax but an "initialization" burst - interrupts the progressing SB
    sb_start = []
    sb_end = []
    num_reverbs = []
    r = 0
    in_super_burst = False
    initial_burst = False
    for i in range(0, len(burst_borders) - 1):
        if (burst_peaks[i] in prime_burst_peaks) & (initial_burst == True):
            initial_burst = False
            sb_end.append(burst_borders[i - 1][1])
            num_reverbs.append(r)
            r = 0
        if (burst_peaks[i] in prime_burst_peaks) & (initial_burst == False):
            # print(f"Initial burst peak at {burst_peaks[i]}")
            initial_burst = True
            sb_start.append(burst_borders[i][0])
        if ((burst_borders[i + 1][0] - burst_borders[i][1]) <= rmax) & (initial_burst == True):
            if in_super_burst:
                r += 1
            in_super_burst = True
        elif ((burst_borders[i + 1][0] - burst_borders[i][1]) > rmax) & (initial_burst == True):
            if in_super_burst:
                r += 1
            in_super_burst = False
            initial_burst = False
            sb_end.append(burst_borders[i][1])
            num_reverbs.append(r)
            r = 0
        i += 1
    if (burst_peaks[len(burst_borders) - 1] in prime_burst_peaks):
        sb_start.append(burst_borders[i][0])
        sb_end.append(burst_borders[i][1])
    else:
        sb_end.append(burst_borders[i][1])
    num_reverbs.append(r)

    if len(sb_start) == len(sb_end):
        return sb_start, sb_end, num_reverbs
    else:
        if len(sb_start) < len(sb_end):
            return (sb_start, sb_end[:len(sb_start)], num_reverbs[:len(sb_start)])
        elif len(sb_start) > len(sb_end):
            return (sb_start[1:], sb_end, num_reverbs[1:])

def find_partial_reverb(sdf, height=0.5, prom=0.5, partial_height=0.50, fs=12500):
    '''
    DOES NOT RELIABLY WORK
    '''
    re_sdf = signal.resample(sdf,37500)
    partial_burst_peaks = []
    burst_peaks = detect_burst_peaks(re_sdf, height=height, prom=prom)
    localMins = argrelextrema(re_sdf, np.less, order=25)[0]
    for peak in burst_peaks:
        try:
            closest_value = localMins[localMins < peak*fs].max()
            if re_sdf[closest_value] >= (re_sdf[peak]*partial_height):
                partial_burst_peaks.append((peak)*100)
        except:
            pass
            #print("Find Partial Reverb - No minimum found before peak.")
    return np.array(partial_burst_peaks)/fs

def super_burst_duration(sb_start, sb_end):
    return np.array(sb_end)-np.array(sb_start)

def inter_super_burst_interval(sb_start, sb_end):
    i = 1
    isbi_tmp = []
    while i < len(sb_start):
        interval = sb_start[i]-sb_end[i-1]
        if interval > 0:
            isbi_tmp.append(interval)
        i+=1
    return np.array(isbi_tmp)

def inter_reverberation_interval(burst_peaks, sb_start, sb_end):
    iri_tmp = []
    for i in range(len(sb_start)):
        start = sb_start[i]
        end = sb_end[i]
        reverb_burst_ind = np.where((burst_peaks >= start) & (burst_peaks <= end))
        iri_tmp.append(np.diff(burst_peaks[reverb_burst_ind]))
        i+=1
    return np.concatenate(iri_tmp).ravel()

def reverb_strength(raster, burst_peaks, kernel, fs=12500):
    num_active_electrodes = len(raster)
    channels_in_bursts = []
    for channel in raster:
        active_in_burst = []
        spiketimes = np.array(channel)
        burst_window = []
        for b in range(len(burst_peaks)):
            start = burst_peaks[b] - ((len(kernel)/3)/fs)
            end = burst_peaks[b] + ((len(kernel)/2) / fs)
            spikes_in_burst = sum((spiketimes >= start) & (spiketimes <= end))
            if spikes_in_burst > 5:
                active_in_burst.append(True)
            else:
                active_in_burst.append(False)
                # print("not participating")
            burst_window.append((start,end))
        channels_in_bursts.append(active_in_burst)
    num_electrodes_in_burst = np.sum(np.array(channels_in_bursts).T, axis=1)
    fraction_of_participating_electrodes = np.array(num_electrodes_in_burst/num_active_electrodes)

    strength = []
    for each_burst in fraction_of_participating_electrodes:
        if each_burst >= 0.8:
            strength.append("Strong")
        elif each_burst >=0.5:
            strength.append("Moderate")
        else:
            strength.append("Weak")
    return fraction_of_participating_electrodes, np.array(strength), np.array(burst_window)

def compute_propagation(raster, sb_start, sb_end):
    t_act = []
    first_spike = []
    channel_spike = np.empty((len(raster), len(sb_start))) * np.nan
    i = 0
    for c in raster:
        channel_act = np.empty(len(sb_start)) * np.nan
        for s in range(len(sb_start)):
            first_spike_time = np.array(c[c > sb_start[s]])
            if len(first_spike_time) > 0:
                first_spike_time = first_spike_time[0]
                if first_spike_time < sb_end[s]:
                    channel_act[s] = (first_spike_time - sb_start[s]) * 1000
                    channel_spike[i, s] = first_spike_time
        i += 1
        t_act.append(channel_act)
    return np.array(t_act), channel_spike

def adaptation(times):
    interval = np.diff(times)
    sum = 0
    if len(interval) > 2:
        for i in range(len(interval)-1):
            sum += ((interval[i+1] - interval[i])/(interval[i+1] + interval[i]))
        adaptation_index = sum/(len(interval)-1)
    else:
        adaptation_index = -1
    return adaptation_index

def burst_adaptation(burst_peaks, sb_start, sb_end):
    a = []
    adaptation_index = 0
    for i in range(len(sb_start)):
        start = sb_start[i]
        end = sb_end[i]
        peaks = burst_peaks[(burst_peaks>start) & (burst_peaks<end)]
        adaptation_index = adaptation(peaks)
        a.append(adaptation_index)
    return np.array(a)

def sttc_all(raster, channel_id):
    x = 0
    STTC_matrix = np.zeros((64,64)) * np.nan
    for ch1_x in range(1,9,1):
        for ch1_y in range(1,9,1):
            channel_search1 = str(ch1_x)+str(ch1_y)
            x +=1
            y = 0
            try:
                channel_index1 = [i for i, s in enumerate(channel_id) if channel_search1 in s][0]
                for ch2_x in range(1, 9, 1):
                    for ch2_y in range(1, 9, 1):
                        channel_search2 = str(ch2_x) + str(ch2_y)
                        y += 1
                        if channel_search1 != channel_search2:
                            try:
                                channel_index2 = [i for i, s in enumerate(channel_id) if channel_search2 in s][0]
                                a = neo.SpikeTrain(raster[channel_index1], units="s", t_stop=300)
                                b = neo.SpikeTrain(raster[channel_index2], units="s", t_stop=300)
                                STTC = spike_time_tiling_coefficient(a,b)
                                STTC_matrix[x,y] = STTC
                                #print(f"Row {x}, Column {y}: {STTC}")
                            except:
                                pass
            except:
                pass
    return STTC_matrix

def sttc(raster):
    x_ind = 0
    STTC_matrix = np.zeros((len(raster),len(raster))) * np.nan
    for x in raster:
        x_train = neo.SpikeTrain(x, units="s", t_stop=300)
        y_ind = 0
        for y in raster:
            y_train = neo.SpikeTrain(y, units="s", t_stop=300)
            if x_ind != y_ind:
                STTC = spike_time_tiling_coefficient(x_train, y_train)
                STTC_matrix[x_ind, y_ind] = STTC
            y_ind += 1
        x_ind += 1

    return STTC_matrix

def sttc_distance(raster, channel_id, spacing=200):
    x = 0
    STTC_matrix = np.zeros((64,64)) * np.nan
    distance_matrix = np.zeros((64,64)) * np.nan
    for ch1_x in range(1,9,1):
        for ch1_y in range(1,9,1):
            channel_search1 = str(ch1_x)+str(ch1_y)
            x +=1
            y = 0
            try:
                channel_index1 = [i for i, s in enumerate(channel_id) if channel_search1 in s][0]
                for ch2_x in range(1, 9, 1):
                    for ch2_y in range(1, 9, 1):
                        channel_search2 = str(ch2_x) + str(ch2_y)
                        y += 1

                        distance = (np.sqrt((ch2_y-ch1_y)**2 + (ch2_x-ch1_x)**2)) * spacing
                        if channel_search1 != channel_search2:
                            try:
                                channel_index2 = [i for i, s in enumerate(channel_id) if channel_search2 in s][0]
                                a = neo.SpikeTrain(raster[channel_index1], units="s", t_stop=300)
                                b = neo.SpikeTrain(raster[channel_index2], units="s", t_stop=300)
                                STTC = spike_time_tiling_coefficient(a,b,dt=0.005*pq.s)
                                STTC_matrix[x,y] = STTC
                                distance_matrix[x, y] = distance
                                #print(f"Row {x}, Column {y}: {STTC}")
                            except:
                                pass
            except:
                pass
    return STTC_matrix, distance_matrix

def compute_synchrony_all_bursts(raster, burst_borders):
    synchrony = []
    for burst in burst_borders:
        start = burst[0]
        end = burst[1]
        burst_spikes = []
        for channel in raster:
            channel = np.array(channel)
            tmp = neo.SpikeTrain(channel[np.where(np.logical_and(channel >= start, channel <= end))], units="s",
                                 t_start=start, t_stop=end)
            burst_spikes.append(tmp)
        synchrony.append(spike_contrast(burst_spikes))

    return synchrony

def compute_synchrony_reverb(raster, sb_start, sb_end):
    synchrony = []
    for b in range(len(sb_start)):
        if sb_end[b] > sb_start[b]:
            start = sb_start[b]
            end = sb_end[b]
            burst_spikes = []
            for channel in raster:
                channel = np.array(channel)
                tmp = neo.SpikeTrain(channel[np.where(np.logical_and(channel >= start, channel <= end))], units="s",
                                     t_start=start, t_stop=end)
                burst_spikes.append(tmp)
            synchrony.append(spike_contrast(burst_spikes))
            b+=1
    return synchrony

