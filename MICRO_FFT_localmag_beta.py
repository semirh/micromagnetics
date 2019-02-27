# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:50:27 2015

Program written by MF

Program for analyzing a micromagetic simulation using both the average magnetization (avmag) 
and the magnetization snapshots (ie. the magnetization of each cell at each time step).
This progam will:
- import the average magnetization (avmag) as a function of tme and do a FFT of it.
- find the frequency peaks in the avmag FFT
- plot the power density spectrum (PSD) of the avmag and its peaks
- import the magnetization snapshops and do a FFT on each cell
- find the peaks that correspond to the ones found in the avmag (in case they 
are slightly different due to different FFT precision)
- change the data from an array of 6 columns to a disk-shaped array.
- plot the local power density at each frequency peak
The longest step in this program is reading the magnetization snapshots 
(ie opening and reading thousands of files on the hard drive).

@author: MF
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pickle
import os
import re
import numpy.fft as fft
import peakutils

#%% Global variables

top_dir = r'C:\Data_Simulation\summer\simulations\2015_oct_holidays\standard_test'        #no slash at the end
save_plot = 0
max_peaks_to_analyze = 8
pa = 3      #rows for plotting. The entire first row is for the PSD of the avmag
pb = 4      #columns for plotting

#%% Define the function to be used in each directory:

#%% Give working directory and find average magnetization file and magnetization snashop files

def local_fft(data_directory):
#if 1==1:                           #for testing purposes
    
    number_peaks = max_peaks_to_analyze    
    os.chdir(data_directory)
    file_list = []
    get_geometry = False
    
    for file in os.listdir(os.getcwd()):
        if file.endswith('00.dat'):             #find the average magnetization file
            if file.startswith('para65x65'):
                av_mag_file = file      
        if file.endswith('0.Mdat'):
            if get_geometry == False:          #find the first magnetization snapshot to get sample geometry
                geometry = np.genfromtxt(file, usecols=(0,1,2))
                get_geometry = True
            iternum = int(re.findall('\d+', file.split('_')[4])[0])     #get iteration number for sorting later
            file_list.append((iternum,file))    #find all magnetization snapshots
    
    file_list=[(x,y) for x,y in file_list if x>1999950]    #keeps snapshots only if simulations longer than 100ns
    file_list.sort()
    fail_list=[(x,y) for x,y in file_list if x<1999950]     #for printing list of simulations that are too short
    
    length_ok = False
    if len(file_list)>10:
        length_ok = True
    
    #if length_ok and file_list[-1][0]-file_list[0][0] > 3000000: #6000000:   
    #does the analysis only if at least 30ns of data AFTER 100ns is available
    #and only if there are at least 11 snapshots
    
        print('OK - ' + os.path.basename(data_directory))
        
    #%% Import average magnetization file. Do the FTT on it.
    
        data = np.genfromtxt(av_mag_file, usecols=(1,3,4,5), skip_header=1)
        Np = 20000
        N = len(data[-Np:,0])
        dt = (data[100,0]-data[0,0])*1e-12/100
        freq_av = fft.fftfreq(N, d=dt)
        fft_mx = np.abs(fft.fft(data[-Np:,1]))*dt
        fft_my = np.abs(fft.fft(data[-Np:,2]))*dt
        fft_mz = np.abs(fft.fft(data[-Np:,3]))*dt
        
#        fft_mall = np.sqrt(fft_mx**2+fft_my**2+fft_mz**2)
        fft_mall = fft_mz
        fft_mall[0] = 0
        
    #%% Find the frequency peaks of average FFT. To save time, the peaks are first found using the 
    #    average magnetization FFT. Then the FFT of each cell is performed, and then 
    #    for each peak found earlier, we look at the neighboring points to see if it:
    #    moved slightly (due to possible different resolution of the average mag FFT 
    #    vs the FFT of each cell) and adjust if necessary. 
    #    All of this is to avoid searching for peaks for every frequency for every cell,
    #    which may (or not) take a long time (I haven't checked)
        
    #    find peaks of the FFT of the average magnetization
    #    peaks = peak_finder(freq_av[1:N/2],fft_mall[1:N/2],10,0)    

        indexes = peakutils.indexes(fft_mall[1:N/2], thres=0.05, min_dist=0)   
        print(indexes)
        peaks = [[freq_av[n+1],fft_mall[n+1]] for n in indexes if n>5]     #the PeakUtils package counts the indexes differently
        peaks=np.array(peaks)
        peaks = peaks[peaks[:,1].argsort()[::-1]]       #sort from highest to lowest peak intensity
        
        if len(peaks) < number_peaks:           #in case there are not enough peaks
            number_peaks = len(peaks)
    
    #%% Plot FFT and peaks of average magnetization
    
        fig = plt.figure()     #for plotting everything in one figure
        ax0 = plt.subplot2grid((pa,pb),(0,0), colspan=4)
        
    #    plt.plot(freq_av[:N/2],fft_mx[:N/2], label='<mx>')
    #    plt.plot(freq_av[:N/2],fft_my[:N/2], label='<my>')
    #    plt.plot(freq_av[:N/2],fft_mz[:N/2], label='<mz>')
        plt.plot(freq_av[:N/2],fft_mall[:N/2],label='<|m|>')
        plt.plot(peaks[:,0],peaks[:,1], 'o', label= 'Peaks')
    
        labels = [str(peaks[n][0]/1e9) for n in range(number_peaks)]    #peak labels
        labels[0] += ' GHz'
        for label, x, y in zip(labels, peaks[:,0], peaks[:,1]):         #label the peaks
            plt.annotate(
                label, 
                xy = (x, y), xytext = (+10, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
                
        plt.xlim(0,9e9)
        plt.ylim(0,np.amax(fft_mall[:N/2]*1.1))
        plt.title(os.getcwd(), fontsize=16)
        plt.legend()

        
        if max(peaks[:,1])< 1e-16:
            ax1 = plt.subplot(pa,pb,5)
            plt.annotate('no oscillations', xy = (0.5,0.5))
            mng = plt.get_current_fig_manager() 
            mng.window.showMaximized()
            if save_plot == 1:
                plt.savefig('local_fft_' + os.path.basename(os.getcwd()) + '.png')
            return None
            print('no oscillations')
    
    #%% Import all magnetization snapshot data. By far the LONGEST step in this whole script
    
        d3d_array = np.dstack([np.genfromtxt(file[1], usecols=(3,4,5)) for file in file_list])
        
    #%% Do FFT for each cell. 
    
        cell_peaks = []
        for i in range(len(d3d_array)):
            N = d3d_array.shape[2]
            dt = 10000*5e-15        #timestep
            freq = fft.fftfreq(N, d=dt)     #frequency bins
            fft_all = np.abs(fft.fftn(d3d_array[i,0:3,:]))*dt
#            fft_sum = np.sqrt(fft_all[0,1:N/2]**2+fft_all[1,1:N/2]**2+fft_all[2,1:N/2]**2)
            fft_sum = fft_all[2,1:N/2]
    
    #%%Find the real peak for each cell, add to list
    #        because for example a peak detected at 1.62 GHz using average magnetization
    #        data (20000 points) might be detected at 1.60 GHz using local magnetization 
    #        snapshots (2000 points)
    
            peak_values = []
            for n in range(number_peaks):
                peak_index = np.abs(freq - peaks[n,0]).argmin()     #get index of peak closest to the one expected
                window = fft_sum[peak_index-5:peak_index+5]         #create a window of frequencies around that index
                real_peak = np.argmax(window)+peak_index-5          #find the real peak frequency
                peak_values.append(fft_sum[real_peak])              #append the frequencies for the top X peaks
            
            cell_peaks.append(peak_values)                          #append each cells's top X peaks
            
    #%% Change data representation from x,y,z,mx,my,mz to simply [mx,my,mz], where the position 
    # of this vector in the array corresponds to the physical position x,y,z of the data point
    # and then plot, for each of the top X peaks
        
        plot_index = 5
        for n in range(number_peaks):
        
            a = np.full((max(geometry[:,0])+1,max(geometry[:,1])+1), np.nan)    #put nan ('not a number') values outside the disk to hide them when plotting later
            
            for i in  range(0,  geometry.shape[0]):
                a[int(round(geometry[i,0])),int(round(geometry[i,1]))] = cell_peaks[i][n]   #change of representation  from (x,y,z,mx,my,mz) to just (mx,my,z), where the position of the tuple in the array is the position of the cell in the disk
            
            masked_peaks = ma.masked_invalid(a)     #remove 'nan' points from the array 
            masked_peaks = masked_peaks/np.amax(masked_peaks)       #normalize the array to the max value of the array. colors/values should therefore cannot be directly compared from one graph (frequency) to another
    
            
        #%% Plotting
            
            regex = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')      #create regex string to find simulation parameter from average magnetization file name. should upgrade by looking directly at the inputfile
            params = re.findall(regex, file_list[0][1])                 #use the regex to find relevant info
            params[2] = str(float(params[2])*1000)
            field_dir = re.findall(r'[xyz]', os.path.basename(os.getcwd()))  
            
            
            axn = plt.subplot(pa,pb,plot_index)
            circle=plt.Circle((32.5,32.5),33, color='k',fill=False,clip_on=False)   #create black circle around disk
            fig.gca().add_artist(circle)
            plt.pcolor(masked_peaks, cmap='hot')        #plot a heatmap of the intensity of the frequency peak
            plt.clim(0,1)
            plt.xlim(0, 65)
            plt.ylim(0, 65)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
    #        plt.colorbar()
            plt.title(str(peaks[n][0]/1e9) + ' GHz',
                      fontsize=10, y=1.01)
                      
            plot_index +=1
            
        mng = plt.get_current_fig_manager() 
        mng.window.showMaximized()
#        plt.tight_layout(pad=1.5)
        
        if save_plot == 1:
            plt.savefig('local_fft_' + os.path.basename(os.getcwd()) + '.png')
        
    else:
        print('Insufficient simulation time')

#%% End of function

#%% Call the function

for dir_path, dirnames, files in os.walk(top_dir,topdown=True):
    for name in files:
        if name.endswith('0.Mdat'):
            if save_plot == 1:
                plt.close('all')
            local_fft(dir_path)
            print(dir_path)
            break