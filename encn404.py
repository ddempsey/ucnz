#encn404.py
from ipywidgets import*
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def _rolling_window(Tm, Tsd, Ti, ts):
    # read in some data
    #ts=pd.read_csv('traffic_data.csv',parse_dates=[0]).set_index('time')['density']

    # plot the raw data
    f,(ax1,ax2,ax3)=plt.subplots(1,3, figsize=(16,4))
    for ax in [ax1,ax2,ax3]:
        ts.plot(style='k-', lw=0.5, ax=ax, label='raw data')

    # Calculate a rolling mean
    #Tm=150
    ts.rolling(window=Tm).mean().plot(style='b',ax=ax1)

    # Calculate a rolling standard deviation
    #Tsd=30
    ts.rolling(window=Tsd).std().plot(style='b', ax=ax2.twinx())

    # Calculate a rolling X-day harmonic 
    def rolling_fft(x, ti):
        fft = np.fft.fft(x)/len(x)
        psd = np.abs(fft)**2/2
        period_of_interest = ti
        ts=1./(np.fft.fftfreq(len(x)))
        i=np.argmin(abs(ts-ti))
        return psd[i]

    #Ti=30   # harmonic (days)
    ts.rolling(window=240).apply(rolling_fft, args=(Ti,)).plot(style='b', ax=ax3.twinx())

    for ax in [ax1,ax2,ax3]:
        ax.set_ylabel('traffic density')
    ax1.set_title(f'feature 1: {Tm:d}-day average')
    ax2.set_title(f'feature 2: {Tsd:d}-day std. dev.')
    ax3.set_title(f'feature 3: {Ti:d}-day harmonic')
def rolling_window():    
    Tm=IntSlider(value=150, min=20, max=240, step=10, description='$T_m$', continuous_update=False)
    Tsd=IntSlider(value=30, min=20, max=120, step=10, description='$T_{sd}$', continuous_update=False)
    Ti=IntSlider(value=30, min=10, max=50, step=5, description='$T_i$', continuous_update=False)
    ts=pd.read_csv('traffic_data.csv',parse_dates=[0]).set_index('time')['density']
    io=interactive_output(_rolling_window, {'Tm':Tm,'Tsd':Tsd,'Ti':Ti,'ts':fixed(ts)})
    return VBox([HBox([Tm, Tsd, Ti]),io])
