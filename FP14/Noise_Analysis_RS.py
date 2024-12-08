'''
Info: This source code includes the analysis of the experimental results of the silicon micro strip detector experiment.

    In detail the following functions are implemented:
    -> calculation and visualization of noise, pedestal and common mode shift
    -> visualization of noise and pedestal estimated by DAC
    -> visualization of the pedestal and noise residuals vs. manually calculated
    -> hit map: number of counts for each channel
    -> cluster analysis: cluster algorithm to find the clusters of charge in the data and the number of clusters per event
    and the size of the clusters
    -> energy calibration: the energy calibration is executed automatically when a file named "calibration.txt" exists
        in the project directory The polynomial of fourth degree fitted to the data is plotted
    -> visualization of the resulting charge and energy distributon
    -> Landau-Gauss convolution and Moyal-function are fitted to the energy data
    

About the code: This code has been written in the summer term 2017 in the frame of the project lab (PP: Projekt Praktikum)
in the master studies physics by Philipp Pagel and Jens Roggel.


Date: 
June 2017
'''


import os
import sys
from   pylab import normpdf, Rectangle
from   scipy.stats import norm as gaus
from scipy.optimize import curve_fit
import h5py as hdf
import pandas as pd
import time
import math
from optparse import OptionParser
from numpy import sum
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import PolynomialModel
from lmfit import Model
import pylandau


# GLib Timer
#########################################
class GTimer(object):
    def __init__(self):
        self._start = 0
        self._end = 0
        self._running = False

    #def start(self):
    #   self._start = time.time()

    def __call__(self):
        if self._running:
            return time.time() - self._start

        else:
            return 0.0

    def start(self):
        self._start = time.time()
        self._running = True

    def stop(self):
        self._running = False

    def reset(self):
        if self._running:
            self._start = time.time()
#########################################


def main(fname,options):
    if not os.path.exists(fname):
        print("Input file", fname,"does not exist")
        return

    T_Total = GTimer()
    T_Total.start()     #start of the timer

    # Einlesen der HDF5-Datei als Pandas DataFrames und Variablendefinitionen
    ###############################################
    nchan = 128
    maxch =int(options.maxch)+1
    minch =int(options.minch)
    nch = maxch-minch
    nevents = int(options.nevents)
    times = pd.DataFrame((hdf.File(fname)['/events/time'][()])
    pedestalF = (pd.DataFrame((hdf.File(fname)['/header/pedestal'])[()]))
    noiseF = (pd.DataFrame((hdf.File(fname)['/header/noise'])[()]))
    signal_raw = pd.DataFrame((hdf.File(fname)['/events/signal'])[()])

    cut_noise = 3.0
    #s_n_cut_base = options.sncut
    s_n_cut_base = 3.0
    s_n_cut_seed = 6.0
    ##########################################################################

    #Events filtern, deren Zeit zu klein ist
    ############################################
    tmin = options.tmin
    tmax = options.tmax
    times_np = times.values
    bad_events = np.array(np.nonzero(np.logical_or(times_np[:] < tmin, times_np[:] > tmax))) # indizes aller Events mit t<tmin und t>tmax
    nevts = len(signal_raw) - len(bad_events[0]) #Anzahl "guter" Events
    signal_red = signal_raw.drop(signal_raw.index[bad_events[0]]) #Events ohne "schlechte" Events
    signal_red.index = np.arange(0, signal_red.index.size)  #index aktualisieren
    ###########################################################################################


    #Definiere Anzahl der zu analysierenden Events
    ################################################
    print("Events in the file. ..................... All Events=", len(signal_raw))
    print("Events with good time. .................. Good Events=", nevts)
    if nevents > 0 and nevents <= nevts:
        nevts = int(nevents)
        signal_red = signal_red.head(nevts)
    print("Events to analyse. ...................... Processed Events=", nevts)
    #####################################################################


    #Channels mit Signal auf NaN setzen
    ##########################################################
    sig_2 = signal_red.subtract(pedestalF.values[0], axis = 1) #Daten ohne pedestal
    charge_mean = sig_2.mean(axis=0)   # mean der daten o. pedestal
    charge_std = sig_2.std(axis=0)     # std der daten ohne pedestal
    mask = sig_2 > (charge_mean + (cut_noise * charge_std)) #Maske fur Bedingung
    signal_red_2=signal_red.copy(deep=True)  ## Arbeitskopie erstellen, wichtig: richtige Kopie, nicht "=", d.h. keine Referenz auf die Speicheradresse!
    signal_red_2[mask] = np.nan       #ausmaskieren der Daten
    ##########################################################

    #Noise wie zuvor berechnen
    ########################################################
    pedestal = np.array(signal_red_2.mean())
    common_mode = (signal_red_2.subtract(pedestal)).mean(axis=1) #D(k)
    std_common = (signal_red_2.subtract(pedestal)).std(axis=1)

    single_noise = signal_red_2.subtract(pedestal, axis=1).subtract(common_mode,axis=0) #Signal -Common Mode Shift - Pedestal
    noise_40=single_noise[39].values
    noise_avg = np.array(single_noise.std(axis=0))
    ########################################################

    #S/N-Cut und Cluster
    ###########################################################
    signal_fin = signal_red.subtract(pedestal, axis=1).subtract(common_mode,axis=0) # Daten ohne Noise und Common Mode


    signal_mask = signal_fin > s_n_cut_base * noise_avg  # Maske der Signal enthaltenden Datenpunkte
    cluster_mask = pd.DataFrame(np.zeros((nevts,nchan))).add(np.arange(nchan),axis=1)[signal_mask] # Alle signale enthalten Zeilenindex, ansonsten NaN
    cluster_mask[np.isnan(cluster_mask)]=-1 # Alle signale enthalten Zeilenindex, ansonsten -1


    split_signal=[]
    for i in range(len(signal_mask)):
        #split_signal.append(np.split(cluster_mask.values[i],np.where(np.diff(signal_mask.values[i])!=0)[0]+1)) #Cluster sowie zwischenraeume in einzelne Events schreiben in einzelne Arrays schieben
        temp_list = np.split(cluster_mask.values[i],np.where(np.diff(signal_mask.values[i])!=0)[0]+1) #Cluster sowie zwischenraeume in einzelne Events schreiben in einzelne Arrays schieben
        split_signal.append([i.astype(int) for i in temp_list])
    cluster_list = []  #Enthaelt fuer jedes Event die Indizes der Luecken zwischen den Clustern

    for i in range(nevts):
        cluster_list[:] = []

        for j in range(len(split_signal[i])):

            if split_signal[i][j][0]==-1:
                cluster_list.append(j)     #wenn Cluster ohne Signal --> Hinzufuegen zu cluster_list

        split_signal[i]=np.delete(split_signal[i], cluster_list, axis=0)  # split_signal enthaelt fuer jedes Event
                                                                            # arrays, die die Indizes der Cluster enthalten
    ###########Bis hier: einfacher Clusterfinder ohne "Seed". Jetzt: Cluster entfernen, die keinen Seed besitzen,
    #um staerkere Bedingung zu fordern###################

    if options.advanced_cluster == 1:
        signal_mask_seed = signal_fin > s_n_cut_seed * noise_avg  # Maske der Signal enthaltenden Datenpunkte
        for i in range(nevts):
            cluster_list[:] = []
            for j in range(len(split_signal[i])):
                if np.sum(signal_mask_seed.iloc[i].values[split_signal[i][j][0]:(split_signal[i][j][-1] + 1)]) == 0:
                    cluster_list.append(j)
            split_signal[i] = np.delete(split_signal[i], cluster_list, axis=0)

    #Clusteranalyse und Berechnung der Gesamtenergie
    #**********************************************************************************
    #**********************************************************************************

    #definition of variables
    clusters = []
    single_cluster_size = []
    eclust = []
    Gcal = []
    # use default parameters or parameters from the parser
    PE0 = options.PE0
    PE1 = options.PE1
    PE2 = options.PE2
    PE3 = options.PE3
    PE4 = options.PE4
    print("Used Parameters of the calibration curve:")
    print("PE0: ", PE0)
    print("PE1: ", PE1)
    print("PE2: ", PE2)
    print("PE3: ", PE3)
    print("PE4: ", PE4)



    #Cluster analyzing and calculation of deposited charge
    ####################################################

    charge_txt_name = fname + "_charge.txt"
    charge_txt_file = open(charge_txt_name,'w')
    for i in range(len(split_signal)):# Loop over all events
        if options.double_hits == 0:
            if len(split_signal[i]) > 1: continue       # Nur ein Cluster pro Event!
        clusters.append(len(split_signal[i]))       # append number of clusters per event
        for j in range (len(split_signal[i])):      # Loop over clusters in event
            single_cluster_size.append(len(split_signal[i][j]))  # append size of clusters per event
            energie = 0

            for k in range(len(split_signal[i][j])):    # loop over indices saved in the arrays
                number = split_signal[i][j][k]          # indices of channel with signal
                energie += signal_fin[number][i]        # sum ADC counts of each cluster
            eclust.append(energie)
            charge_txt_file.write(str(energie) + '\n')

            Elec = PE0 + (PE1 * energie) + (PE2 * (energie ** 2)) + (PE3 * (energie ** 3)) + (PE4 * (energie ** 4)) # calculate the deposited energy using the callibration
            Elec = Elec * 3.67  # multiply with the mean band gap in silicon
            if Elec > 0.:
                Elec = Elec / 1000  # energy in keV

            Gcal.append(Elec)   # append energy calculated by calibration data
    charge_txt_file.close()

    #******************************************************************
    #******************************************************************
    # Fit LanGau Distribution
    bin_width = options.Eplotbin
    ydata2 = np.histogram(Gcal, bins = (300-np.int(300%bin_width)) / bin_width  , range = (0,(300-np.int(300%bin_width))))[0]
    xdata2 = np.arange(0,(300-np.int(300%bin_width)),bin_width)
    yerr = np.sqrt(ydata2)
    yerr[ydata2 == 0] = 1
    def land(x, mpv, eta, sigma, A):
        return pylandau.langau(x, mpv, eta, sigma, A)

    emin2 = options.Emin
    emax2 = options.Emax

    emin = int(emin2 / bin_width)
    emax = int(emax2 / bin_width)
    model2 = Model(land)
    fit2 = model2.fit(ydaxta2[emin:emax].astype(np.float), mpv = np.argmax(ydata2[emin:emax])*bin_width+emin, eta = 5, sigma = 4,
                      A = np.amax(ydata2), x=xdata2[emin:emax].astype(np.float),
                      weights = 1/(yerr[emin:emax].astype(np.float)), scale_covar = False)
    print(fit2.fit_report())

    # Plot the the distribution of the energy with a LanGau fit to the data
    fig9, ax9 = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,1] } )
    ax9[0].axvspan(emin2, emax2, facecolor='forestgreen', alpha=0.1, zorder = 1,label = 'Range of fitted Data')
    ax9[0].set_title('Fit with Landau-Gauss-Convolution')
    ax9[0].set_xlabel("Energy / keV")
    ax9[0].set_ylabel("Counts")
    ax9[0].plot(xdata2, land(xdata2.astype(np.float), fit2.values['mpv'], fit2.values['eta'], fit2.values['sigma'], fit2.values['A']), '-r', label = 'Landau*Gauss fit', zorder = 3 )
    ax9[0].errorbar(xdata2, ydata2, yerr=yerr, fmt = 'o', color = (0,0.7,0), label = 'Data', zorder = 2, mec = 'darkgreen', ecolor = 'green')
    ax9[0].tick_params(direction='in')
    handles, labels = ax9[0].get_legend_handles_labels()
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    handles.append(extra)
    handles.append(extra)
    labels.append(u'MPV = (%.1f $\pm$ %.1f) keV ' % (fit2.values['mpv'], fit2.params['mpv'].stderr))
    labels.append(u'$\chi^2_{red}$ $\simeq$ %.2f' % (fit2.redchi))
    ax9[0].legend(labels=labels, handles = handles, loc='best')
    ax9[1].scatter(xdata2[emin:emax], fit2.residual, color = 'royalblue', label = 'Residuals', edgecolor='darkblue', linewidth='1.5')
    ax9[1].axhline(0, color = 'blue')
    ax9[1].legend(loc='best')
    ax9[1].tick_params(direction='in')
    plt.tight_layout()
    # fit2.plot()


    #******************************************************************
    #******************************************************************
    # Fit Moyal-function to the data
    def fMoyal(x, norm, mode, width):
        # Approximate the landau distribution with the Moyal function

        L = (x - mode) / width
        return norm * np.exp(-0.5 * (L + np.exp(-L))) / np.sqrt(2.0 * np.pi)

    moyal_model = Model(fMoyal)
    fit3 = moyal_model.fit(ydata2[emin:emax].astype(np.float), mode = np.argmax(ydata2[emin:emax])*bin_width+emin, width = 5,
                           norm = np.amax(ydata2), x=xdata2[emin:emax].astype(np.float),
                           weights = 1/(yerr[emin:emax].astype(np.float)), scale_covar = False)
    print(fit3.fit_report())

    # Plot of the energy distribution and Moyal-Fit
    fig10, ax10 = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    ax10[0].axvspan(emin2, emax2, facecolor='forestgreen', alpha=0.1, zorder=1,label = 'Range of fitted Data')
    ax10[0].set_title('Fit with Moyal-Approximation')
    ax10[0].set_xlabel("Energy / keV")
    ax10[0].set_ylabel("Counts")
    ax10[0].plot(xdata2, fMoyal(xdata2, fit3.values['norm'], fit3.values['mode'], fit3.values['width']), '-r', label='Moyal Fit', zorder=3)
    ax10[0].errorbar(xdata2, ydata2, yerr=yerr, fmt='o', color=(0, 0.7, 0), label='Data', zorder=2, mec='darkgreen',
                    ecolor='green')
    ax10[0].tick_params(direction='in')
    handles2, labels2 = ax10[0].get_legend_handles_labels()
    handles2.append(extra)
    handles2.append(extra)
    labels2.append(u'MPV = (%.1f $\pm$ %.1f) keV ' % (fit3.values['mode'], fit3.params['mode'].stderr))
    labels2.append(u'$\chi^2_{red}$ $\simeq$ %.2f' % (fit3.redchi))

    ax10[0].legend(labels=labels2, handles = handles2 ,loc='best')
    ax10[1].scatter(xdata2[emin:emax], fit3.residual, color='royalblue', label='Residuals', edgecolor='darkblue',
                   linewidth='1.5')
    ax10[1].axhline(0, color='blue')
    ax10[1].legend(loc='best')
    ax10[1].tick_params(direction='in')
    plt.tight_layout()



    #******************************************************************
    #******************************************************************

    #Plotten der Ergebnisse
    #########################################################


    plt.rcParams['figure.figsize'] = 17, 10
    plt.style.use("classic")

    #plt.style.use("ggplot")
    ######################################################

    #common mode shift per channel
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("Common mode")
    ax[0].set_xlabel("Channel number")
    ax[0].set_ylabel("ADC counts")
    ax[0].tick_params(direction='in')
    #[i.set_linewidth(100) for i in ax[0].spines.itervalues()]
    mu, sigma = gaus.fit(common_mode)
    n, bins, patches = ax[0].hist(common_mode, np.arange(-50, 50, 2), normed=1, color="mediumseagreen",
                                  label=r'$\mu$=%.3f $\sigma$=%.1f' % (mu, sigma))
    y = normpdf(bins, mu, sigma)
    l = ax[0].plot(bins, y, 'r-', linewidth=2)
    p = Rectangle((0, 0), 1, 1, fc="r")
    ax[0].legend([p], [r'$\mu$=%.3f $\sigma$=%.1f' % (mu, sigma)], loc=1)

    # ADC counts per Event
    ax[1].set_xlabel("Event number")
    ax[1].set_ylabel("ADC counts")
    X = np.arange(0, common_mode.size, 1)
    ax[1].set_xlim(0, 1000)
    ax[1].plot(X, common_mode, color="mediumseagreen")
    ax[1].hlines(mu, 0, 1000)
    ax[1].hlines(mu + sigma, 0, 1000)
    ax[1].hlines(mu - sigma, 0, 1000)
    ax[1].tick_params(direction='in')
    plt.tight_layout()

    #############################################################

    #pedestal from analysis
    fig2, ax2 = plt.subplots(2,1)
    X = np.arange(0, 128, 1)
    ax2[0].set_title("Pedestal")
    ax2[0].set_xlabel("Channel number")
    ax2[0].set_ylabel("ADC counts")
    ax2[0].set_ylim(450, 550)
    ax2[0].set_xlim(0, 128)
    ax2[0].bar(X, pedestal,1, color="mediumseagreen")
    ax2[0].tick_params(direction='in')
    # noise for each channel
    ax2[1].set_title("Noise vs Channel")
    ax2[1].set_xlabel("Channel number")
    ax2[1].set_ylabel("ADC counts")
    ax2[1].set_ylim(0, 7)
    ax2[1].set_xlim(0, 128)
    ax2[1].tick_params(direction='in')
    ax2[1].bar(X, noise_avg, 1, color="mediumseagreen")
    meanN = noise_avg[minch:maxch].mean()
    sigmaN = noise_avg[minch:maxch].std()
    ax2[1].hlines(meanN, 0, 128)
    ax2[1].hlines(meanN + sigmaN, 0, 128, color = "red")
    ax2[1].hlines(meanN - sigmaN, 0, 128,color = "red")

    p = Rectangle((0, 0), 1, 1, fc="g")
    ax2[1].legend([p], [r'$\mu$=%.3f $\sigma$=%.1f' % (meanN, sigmaN)], loc='best')
    plt.tight_layout()

    #########################################################

    # pedestal estimated by DAC
    fig3, ax3 = plt.subplots(2, 1)
    X = np.arange(0,128,1)
    ax3[0].set_title("Pedestal (estimation from DAC)")
    ax3[0].set_xlabel("Channel number")
    ax3[0].set_ylabel("ADC counts")
    ax3[0].set_ylim(450, 550)
    ax3[0].set_xlim(0, 128)
    ax3[0].tick_params(direction='in')
    ax3[0].bar(X, pedestalF.values[0], 1, color="mediumseagreen")

    #noise estimated  by DAC
    ax3[1].set_title("Noise (estimation from DAC)")
    ax3[1].set_xlabel("Channel number")
    ax3[1].set_ylabel("ADC counts")
    ax3[1].set_ylim(0, 7)
    ax3[1].set_xlim(0, 128)
    ax3[1].tick_params(direction='in')
    ax3[1].bar(X, noiseF.values[0], 1, color="mediumseagreen")
    plt.tight_layout()

    #############################

    # pedestal residual
    fig4, ax4 = plt.subplots(2, 1)
    ax4[0].set_title("Pedestal residuals")
    ax4[0].set_xlabel("Channel number")
    ax4[0].set_ylabel("ADC counts")
    ax4[0].set_ylim(-3, 3)
    ax4[0].set_xlim(0, 128)
    ax4[0].tick_params(direction='in')
    ax4[0].bar(X, pedestalF.values[0]-pedestal, 1, color="mediumseagreen")

    # noise residual
    ax4[1].set_ylim(0, 6)
    ax4[1].set_title("Noise residuals")
    ax4[1].set_xlabel("Channel number")
    ax4[1].set_ylabel("ADC counts")
    ax4[1].set_ylim(-5, 5)
    ax4[1].set_xlim(0, 128)
    ax4[1].tick_params(direction='in')
    ax4[1].bar(X, noiseF.values[0]-noise_avg, 1, color="mediumseagreen")
    plt.tight_layout()

    ####################################################

    # Hit Map -> Number of hits for each channel in all events
    fig5, ax5 = plt.subplots(1, 1)
    X = np.arange(0, 128, 1)
    ax5.set_xlim(0, 128)
    ax5.set_title("Hit map (Channel)")
    ax5.set_ylabel('Number of entries')
    ax5.set_xlabel('Channel number')
    ax5.tick_params(direction='in')
    ax5.bar(X, np.sum(signal_mask, axis=0), 1, color="mediumseagreen")
    plt.tight_layout()

    print("Number of triggered Channels in total: ", np.sum(np.sum(signal_mask, axis=0)))
    print("Total number of Clusters: ", len(single_cluster_size))

    # Number of Clusters
    X = np.arange(-0.5, 4.5, 1)
    fig6, ax6 = plt.subplots(2, 1)
    ax6[0].set_xlim(0, 128)
    ax6[0].set_title("Number of clusters per event")
    ax6[0].set_xlabel('Number of clusters')
    ax6[0].set_xlim((-0.5, 4.5))
    ax6[0].tick_params(direction='in')
    ax6[0].hist(clusters, X, color="mediumseagreen",log=True)

    # Clusters size
    X = np.arange(-0.5, 9.5, 1)
    ax6[1].set_title("Cluster size")
    ax6[1].set_xlabel('Number of channels')
    ax6[1].set_xlim((-0.5, 9.5))
    ax6[1].tick_params(direction='in')
    ax6[1].hist(single_cluster_size, X, color="mediumseagreen",log=True)
    plt.tight_layout()

    ##############################

    # charge distribution
    X = np.arange(0, 600, 1)
    fig7, ax7 = plt.subplots(2, 1)
    # plt.title("Signal (ADC)")
    ax7[0].set_ylabel('Number of entries')
    ax7[0].set_xlabel('Charge (ADCs)')
    ax7[0].set_ylim(1, 10000)
    ax7[0].tick_params(direction='in')
    ax7[0].hist(eclust, X, color="mediumseagreen", log=True)

    # plt.title(options.label)

    # energy distribution
    bin_width_h = 6
    X = np.arange(0, 300, bin_width_h)
    ax7[1].set_ylabel('Number of entries')
    ax7[1].set_xlabel('Energy (KeV)')
    ax7[1].tick_params(direction='in')
    ax7[1].hist(Gcal, X, color="mediumseagreen")
    plt.tight_layout()

    ######################################################################

    #ADCbin=int(options.ADCbin)
    #ADCmax=int(options.ADCmax)
    #ADCmin=int(options.ADCmin)

    print("Time elapsed: ", T_Total())
    T_Total.stop()
	
   
    #Show the plot
    ################
    plt.show()
    ################


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("--tmin",
                      dest="tmin", action="store", type="float",
                      help="Minimum TDC to be considered in the analysis (default=0.)",
                      default=0.
                      )
    parser.add_option("--tmax",
                      dest="tmax", action="store", type="float",
                      help="Maximum TDC to be considered in the analysis (default=100.)",
                      default=100.
                      )
    parser.add_option("--nevents",
                      dest="nevents", action="store", type="float",
                      help="Number of events to analyce (DEFAULY ALL)",
                      default=0.
                      )
    parser.add_option("--maxch",
                      dest="maxch", action="store", type="float",
                      help="Maximum channel for noise analyce (DEFAULY=127)",
                      default=127.
                      )
    parser.add_option("--minch",
                      dest="minch", action="store", type="float",
                      help="Minimum channel for noise analice (DEFAULY=0)",
                      default=0.
                      )

    #not activated/used
    parser.add_option("--s/n",
                      dest="sncut", action="store", type="float",
                      help="Signal/Noise cut  (default=5)",
                      default=5.
                      )
    parser.add_option("--PE0",
                      dest="PE0", action="store", type="float",
                      help=" Parameter 0 to convert ADCs to Energy (default=-270.13)",
                      default=125.942
                      )
    parser.add_option("--PE1",
                      dest="PE1", action="store", type="float",
                      help=" Parameter 1 to convert ADCs to Energy (default=160.904)",
                      default=129.791
                      )
    parser.add_option("--PE2",
                      dest="PE2", action="store", type="float",
                      help=" Parameter 2 to convert ADCs to Energy (default=0.174026)",
                      default=0.211
                      )
    parser.add_option("--PE3",
                      dest="PE3", action="store", type="float",
                      help=" Parameter 3 to convert ADCs to Energy (default=-0.000734166)",
                      default=-0.00080555
                      )
    parser.add_option("--PE4",
                      dest="PE4", action="store", type="float",
                      help=" Parameter 4 to convert ADCs to Energy (default=0.00000187504)",
                      default=0.00000017
                      )
    # not activated/used
    parser.add_option("--ADCmin",
                      dest="ADCmin", action="store", type="float",
                      help="Minimum ADC for the plot  (default=0)",
                      default=0.
                      )
    # not activated/used
    parser.add_option("--ADCmax",
                      dest="ADCmax", action="store", type="float",
                      help="Maximum ADC for the plot  (default=600)",
                      default=600.
                      )
    # not activated/used
    parser.add_option("--ADCbin",
                      dest="ADCbin", action="store", type="float",
                      help="Binning for ADC plot  (default=6)",
                      default=6.
                      )
    parser.add_option("--Emin",
                      dest="Emin", action="store", type="int",
                      help="Minimum energy for the fit  (default=70)",
                      default=58
                      )
    parser.add_option("--Emax",
                      dest="Emax", action="store", type="int",
                      help="Maximum energy for the fit  (default=200)",
                      default=120
                      )
    parser.add_option("--Eplotbin",
                      dest="Eplotbin", action="store", type="int",
                      help="Binning for Energy plot  (default=1)",
                      default=1
                      )
    # not activated/used
    parser.add_option("--calib_fname",
                      dest="calib_fname", action="store", type="string",
                      help="Filename of Calibration run txt file",
                      default='0'
                      )
    parser.add_option("--double_hits",
                      dest="double_hits", action="store", type="int",
                      help="0: dont evaluate Events with more than one Cluster. 1: do it.",
                      default=1
                      )
    parser.add_option("--advanced_cluster",
                      dest="advanced_cluster", action="store", type="int",
                      help="0: no advanced clustering (saves time)",
                      default=1
                      )
    (options, args) = parser.parse_args()

    try:
        main(args[0], options)
    except KeyError:
        print("I need an input file")
