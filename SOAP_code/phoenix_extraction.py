import numpy as np
import matplotlib.pyplot as plt
import bz2
import pandas as pd
from scipy.interpolate import interp1d
import configparser
config = configparser.ConfigParser()
import os
from pathlib import Path
from astropy.io import fits


def vac2air(wavelength):
    '''
    # Transform wavelength in vacuum to air
    # See VALD website here
    # http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    # The formula comes from Birch and Downs (1994, Metrologia, 31, 315)
    '''
    s = 1.e4 / wavelength # s = 10^4 / lamda_vac, lamda_vac is in Angstrom
    n_air = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return wavelength / n_air


def air2vac(wavelength):
    '''
    # Transform wavelength in air to vacuum
    # See VALD website here
    # http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    # The formula comes from N. Piskunov
    '''
    s = 1.e4 / wavelength # s = 10^4 / lamda_vac, lamda_vac is in Angstrom
    n_air = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return wavelength * n_air


def PHOENIX_name(temp, logg_val, FeH_val, alpha_val):

    if FeH_val <= 0.0:
        prefix_subdir = 'Z-%.1f' % np.abs(FeH_val)
        if alpha_val == 0.0:

            name = '/lte0' + str(int(temp)) + '-%.2f-%.1f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (logg_val, np.abs(FeH_val))
        elif alpha_val > 0.0:

            name = '/lte0' + str(int(temp)) + '-%.2f-%.1f.Alpha=+%.2f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (logg_val, np.abs(FeH_val), alpha_val)
        else:

            name = '/lte0' + str(int(temp)) + '-%.2f-%.1f.Alpha=%.2f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (logg_val, np.abs(FeH_val), alpha_val)

    else:
        prefix_subdir = 'Z+%.1f' % np.abs(FeH_val)
        name = '/lte0' + str(int(temp)) + '-%.2f+%.1f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (logg_val, np.abs(FeH_val))

    return prefix_subdir, name




config_name = 'config_conti_phoenix.cfg'
config.read(config_name)

tstar  = float(config.get('star','Tstar' ))
tspot  = tstar - float(config.get('star','Tdiff_spot' ))
tfaculae = tstar + float(config.get('star','Tdiff_faculae' ))

Teff_array = np.array([tstar, tspot, tfaculae])



grav = float(config.get('star','logg' ))
wvmin = float(config.get('star','minwave' ))
wvmax = float(config.get('star','maxwave' ))
FeH = 0.0
alpha = 0.0

input_prefix = config.get('data_io','input_prefix')
out_path = input_prefix+config.get('data_io','input_dir_source')


rassine_path = config.get('data_io','Rassine_directory')



logg_array = np.linspace(0.0,6.0,13)
temp_array = np.zeros(73)
FeH_array = np.concatenate( (np.linspace(-4.0, -3.0, 2), np.linspace(-2.0, 1.0, 7)) )
FeH_array = FeH_array.round(decimals=2)


alpha_array = -0.2+np.arange(8)*0.2
alpha_array = alpha_array.round(decimals=2)


for j in np.arange(73):
    if j < 48:
        temp_array[j] = 2300.0 + 100.0*(j)
    else:
        temp_array[j] = 7000.0 + 200.0*(j-49)


# print(temp_array.min())
index_logg = np.argmin(np.abs(logg_array - grav))
index_FeH = np.argmin(np.abs(FeH_array - FeH))
index_alpha = np.argmin(np.abs(alpha_array - alpha))




prefix_phoenix = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
prefix_phoenix_wv = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'

mask_type = np.array(['F9', 'G2', 'G8', 'G9', 'K2',])
Teff_array = np.array([6050, 5778, 5480, 5380, 5100])


Temp  = int(config.get('star','Tstar'))
Teff_array = np.array([Temp])

for teff_id in np.arange(Teff_array.size):


    tstar  = Teff_array[teff_id]

    tspot  = tstar - int(config.get('star','Tdiff_spot' ))
    tfaculae = tstar + int(config.get('star','Tdiff_faculae' ))

    T_array = np.array([tstar, tspot, tfaculae])


    for t_id in np.arange(T_array.size):
        temp = T_array[t_id]
        print(';;;;;;;;;;;;;;;;;;')
        print(temp)


        logg_val = logg_array[index_logg]
        FeH_val = FeH_array[index_FeH]
        if FeH_val > 0.0:
            alpha_val = 0.0
        else:
            alpha_val = alpha_array[index_alpha]

        index_temp = np.argmin(np.abs(temp_array - temp))
        if temp > temp_array[index_temp]:

            temp_low = temp_array[index_temp]
            temp_high = temp_array[index_temp+1]

        else:

            temp_low = temp_array[index_temp-1]
            temp_high = temp_array[index_temp]

        wave_name = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

        sub_dir_low, name_low = PHOENIX_name(temp_low, logg_val, FeH_val, alpha_val)
        sub_dir_high, name_high = PHOENIX_name(temp_high, logg_val, FeH_val, alpha_val)
        wave_file = Path(out_path+wave_name)
        name_low_file = Path(out_path+name_low)
        name_high_file = Path(out_path+name_high)



        if not wave_file.is_file():
            print('Start to download wavelength data...')
            os.system('curl '+prefix_phoenix_wv+wave_name+' -o '+out_path+wave_name)
        if not name_low_file.is_file():
            print('Start to download spectral data...')
            os.system('curl '+prefix_phoenix+sub_dir_low+name_low+' -o '+out_path+name_low)
        if not name_high_file.is_file():
            print('Start to download spectral data...')
            os.system('curl '+prefix_phoenix+sub_dir_high+name_high+' -o '+out_path+name_high)

        wave_data_vac = fits.getdata(out_path+wave_name)
        waveIndex = (wave_data_vac >= (wvmin-500)) & (wave_data_vac <= (wvmax+500))
        wavelength_vac = wave_data_vac[waveIndex]

        spec_low = fits.getdata(out_path+name_low)[waveIndex]
        spec_high = fits.getdata(out_path+name_high)[waveIndex]


        wavelength_air = vac2air(wavelength_vac)
        waveIndex_new = (wavelength_air >= wvmin) & (wavelength_air <= wvmax)
        wavelength_air_new = wavelength_air[waveIndex_new]
        spec_low_air = spec_low[waveIndex_new]
        spec_high_air = spec_high[waveIndex_new]
        flux_temp = np.zeros(wavelength_air_new.size)

        for i in np.arange(wavelength_air_new.size):
            flux_temp[i] = spec_low_air[i] + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_air[i] - spec_low_air[i])


        data_name = out_path+'New_Phoenix_'+str(int(temp))+'.p'
        spec_df = pd.DataFrame({'wave':wavelength_air_new,'flux':flux_temp})
        spec_df.to_pickle(data_name)

        cmds = 'python3 '+rassine_path+'Rassine.py '+out_path+'New_Phoenix_'+str(int(temp))+'.p '+out_path+' '+rassine_path
        print(cmds)
        os.system(cmds)
        print(';;;;;;;;;;;;;;;;;;')
