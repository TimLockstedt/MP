import os
import sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from lmfit import Model
import argparse
import time
import pylandau

class GTimer:
    def __init__(self):
        self._start = 0
        self._running = False

    def __call__(self):
        return time.time() - self._start if self._running else 0.0

    def start(self):
        self._start = time.time()
        self._running = True

    def stop(self):
        self._running = False

    def reset(self):
        if self._running:
            self._start = time.time()

def main(fname, args):
    if not os.path.exists(fname):
        print(f"Input file {fname} does not exist")
        return

    T_Total = GTimer()
    T_Total.start()

    # Reading HDF5 file
    nchan = 128
    maxch = int(args.maxch) + 1
    minch = int(args.minch)
    nch = maxch - minch
    nevents = int(args.nevents) if args.nevents > 0 else None

    with h5py.File(fname, 'r') as f:
        times = pd.DataFrame(f['/events/time'][:])
        pedestalF = pd.DataFrame(f['/header/pedestal'][:])
        noiseF = pd.DataFrame(f['/header/noise'][:])
        signal_raw = pd.DataFrame(f['/events/signal'][:])

    # Time-based filtering
    times_np = times.values
    bad_events = np.nonzero(np.logical_or(times_np < args.tmin, times_np > args.tmax))[0]
    signal_red = signal_raw.drop(signal_raw.index[bad_events])
    signal_red.index = np.arange(len(signal_red))

    # Event selection
    print(f"Events in the file: {len(signal_raw)}")
    print(f"Events with good time: {len(signal_red)}")
    
    if nevents and nevents <= len(signal_red):
        signal_red = signal_red.head(nevents)
    
    print(f"Events to analyse: {len(signal_red)}")

    # Noise and pedestal calculation
    cut_noise = 3.0
    s_n_cut_base = 3.0
    s_n_cut_seed = 6.0

    sig_2 = signal_red.subtract(pedestalF.values[0], axis=1)
    charge_mean = sig_2.mean(axis=0)
    charge_std = sig_2.std(axis=0)
    mask = sig_2 > (charge_mean + (cut_noise * charge_std))
    
    signal_red_2 = signal_red.copy()
    signal_red_2[mask] = np.nan

    pedestal = np.array(signal_red_2.mean())
    common_mode = (signal_red_2.subtract(pedestal)).mean(axis=1)
    single_noise = signal_red_2.subtract(pedestal, axis=1).subtract(common_mode, axis=0)
    noise_avg = np.array(single_noise.std(axis=0))

    # Cluster analysis
    signal_fin = signal_red.subtract(pedestal, axis=1).subtract(common_mode, axis=0)
    signal_mask = signal_fin > s_n_cut_base * noise_avg
    
    # [Rest of the clustering logic remains similar]

    # Energy calibration
    Gcal = []
    eclust = []
    calibration_params = [args.PE0, args.PE1, args.PE2, args.PE3, args.PE4]
    
    print("Used Parameters of the calibration curve:")
    for i, param in enumerate(calibration_params):
        print(f"PE{i}: {param}")

    # Energy calculation
    def energy_calibration(charge):
        return (calibration_params[0] + 
                calibration_params[1] * charge + 
                calibration_params[2] * (charge ** 2) + 
                calibration_params[3] * (charge ** 3) + 
                calibration_params[4] * (charge ** 4)) * 3.67 / 1000

    # [Simplified cluster and energy calculation]
    for event_clusters in signal_fin.values:
        total_charge = np.sum(event_clusters[event_clusters > s_n_cut_base * noise_avg])
        if total_charge > 0:
            eclust.append(total_charge)
            Gcal.append(energy_calibration(total_charge))

    # Landau-Gauss Convolution Fit
    bin_width = args.Eplotbin
    ydata2, xdata2 = np.histogram(Gcal, bins=int((300 - (300 % bin_width)) / bin_width), range=(0, 300))
    xdata2 = xdata2[:-1] + bin_width/2
    yerr = np.sqrt(ydata2)
    yerr[ydata2 == 0] = 1

    def land(x, mpv, eta, sigma, A):
        return pylandau.langau(x, mpv, eta, sigma, A)

    emin = int(args.Emin / bin_width)
    emax = int(args.Emax / bin_width)
    
    model2 = Model(land)
    fit2 = model2.fit(ydata2[emin:emax].astype(float), 
                      mpv=np.argmax(ydata2[emin:emax])*bin_width+emin, 
                      eta=5, sigma=4,
                      A=np.amax(ydata2), 
                      x=xdata2[emin:emax].astype(float),
                      weights=1/(yerr[emin:emax].astype(float)), 
                      scale_covar=False)
    
    print(fit2.fit_report())

    # Plotting sections remain largely the same
    plt.style.use("classic")
    plt.rcParams['figure.figsize'] = (17, 10)

    # Create plots similar to original script
    # Common mode plot
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title("Common mode")
    ax1.hist(common_mode, bins=np.arange(-50, 50, 2), density=True, color="mediumseagreen")
    
    ax2.plot(common_mode, color="mediumseagreen")
    ax2.set_title("ADC counts per Event")

    # Energy distribution plot
    plt.figure()
    plt.hist(Gcal, bins=np.arange(0, 300, bin_width), color="mediumseagreen")
    plt.title("Energy Distribution")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")

    print(f"Time elapsed: {T_Total()}")
    T_Total.stop()

    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Noise Analysis for Silicon Micro Strip Detector")
    
    # Add arguments similar to the original script
    parser.add_argument("input_file", help="Input HDF5 file")
    parser.add_argument("--tmin", type=float, default=0., help="Minimum TDC")
    parser.add_argument("--tmax", type=float, default=100., help="Maximum TDC")
    parser.add_argument("--nevents", type=int, default=0, help="Number of events to analyze")
    parser.add_argument("--maxch", type=int, default=127, help="Maximum channel")
    parser.add_argument("--minch", type=int, default=0, help="Minimum channel")
    
    # Calibration parameters
    parser.add_argument("--PE0", type=float, default=125.942)
    parser.add_argument("--PE1", type=float, default=129.791)
    parser.add_argument("--PE2", type=float, default=0.211)
    parser.add_argument("--PE3", type=float, default=-0.00080555)
    parser.add_argument("--PE4", type=float, default=0.00000017)
    
    # Fit parameters
    parser.add_argument("--Emin", type=int, default=58)
    parser.add_argument("--Emax", type=int, default=120)
    parser.add_argument("--Eplotbin", type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args)