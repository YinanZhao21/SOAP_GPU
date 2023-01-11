import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib import rc,rcParams
import os
import pandas
import string
from scipy.interpolate import CubicSpline
matplotlib.rc('axes.formatter', useoffset=False)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
import configparser
config = configparser.ConfigParser()


def mask_templates(Teff_star):

    mask_type = np.array(['F9', 'G2', 'G8', 'G9', 'K2', 'K5', 'K6', 'M0', 'M2', 'M3', 'M4', 'M5'])
    Teff_template_array = np.array([6050, 5770, 5480, 5380, 5100, 4400, 4300, 3800, 3400, 3250, 3100, 2800])

    index = np.argmin(np.abs(Teff_template_array - Teff_star))
    mask_template_name = mask_type[index]+'_mask.txt'

    return mask_template_name


def bisector_measurement_line(wavelength, flux, noise, line_cen):

    spectr = {'master_deriv':[], 'master_bisector':[], 'master_velocity':[]}

    line_wl = line_cen
    wl_range = 2

    wl_line_range = 0.1
    c_lum = 299792458

    # Bisector para
    bis_step = 0.015
    min_bis_depth = 0.18
    max_bis_depth = 0.9+bis_step/2.


    master_continuum = np.mean(np.sort(flux)[-50:])

    select_line_master = (wavelength > line_wl-wl_line_range) & (wavelength < line_wl+wl_line_range)

    master_flux_norm = flux[select_line_master]/master_continuum


    min_master_flux = np.min(master_flux_norm)


    wave_min_flux = wavelength[select_line_master][np.where(master_flux_norm == min_master_flux)[0][0]]

    left_wing = wavelength[select_line_master] < wave_min_flux
    right_wing = wavelength[select_line_master] >= wave_min_flux

    depth = np.arange(max(min(master_flux_norm[left_wing]), min(master_flux_norm[right_wing]))+0.05,
                   min(max(master_flux_norm[left_wing]), max(master_flux_norm[right_wing]))-0.05, bis_step)


    depth_left = np.concatenate([depth,np.array([np.max(depth)+0.01])])[::-1]
    depth_right = np.copy(depth)
    depth_all = np.concatenate([depth_left, depth_right])

    left_interp = interp1d(master_flux_norm[left_wing], wavelength[select_line_master][left_wing])
    right_interp = interp1d(master_flux_norm[right_wing], wavelength[select_line_master][right_wing])
    wave_resamp_master = np.concatenate([left_interp(depth_left), right_interp(depth_right)])


    spectr['master_deriv'].append((depth_all[1:] - depth_all[:-1]) / (wave_resamp_master[1:] - wave_resamp_master[:-1]))
    spectr['master_velocity'].append((wave_resamp_master[1:]-line_wl)/line_wl*c_lum)
    spectr['master_bisector'] = ((spectr['master_velocity'][-1][:len(depth)]+spectr['master_velocity'][-1][-len(depth):][::-1])/2)[::-1]


    return spectr, depth

c = 299792458.



def calculate_CCF_1(velocity_array, wavelength, wavelength_line, weight_line, wavelength_extend):

    mask_template = np.zeros((velocity_array.size, wavelength.size))
    index_old_begin =20000
    index_old_end = 20000

    begin_wave = wavelength_extend.min()

    for i in np.arange(velocity_array.size):


        wavelength_line_shift = wavelength_line*(1+velocity_array[i]/c)

        mask_corr = calc_mask(wavelength_extend, wavelength_line_shift, weight_line, begin_wave)
        mask_corr = mask_corr[index_old_begin:-index_old_end]

        mask_template[i,:] = mask_corr


    return mask_template


def calculate_CCF_2(velocity_array, spectrum, mask_template):

    CCF = np.zeros(velocity_array.size)

    for i in np.arange(velocity_array.size):

        ccf_vel = np.sum(spectrum*mask_template[i,:])
        CCF[i] = ccf_vel

    return CCF


def Delta_wavelength(v,wavelength0):
    c = 299792458.
    beta = v/c
    delta_wavelength = wavelength0 * (np.sqrt((1+beta)/(1-beta))-1)
    return delta_wavelength


def calc_mask(wavelength_extend ,wavelength_line_shift, weight_line, begin_wave, mask_width=820,hole_width=0):


    hole_width = np.array([Delta_wavelength(mask_width, wavelength_line_shift[i]) for i in np.arange(len(wavelength_line_shift))])

    begining_mask_hole = wavelength_line_shift-hole_width/2.
    end_mask_hole = wavelength_line_shift+hole_width/2.

    index_begining_mask_hole = []
    index_end_mask_hole = []

    freq_step_before_mask_hole = []
    freq_step_after_mask_hole = []
    bg_wave = wavelength_extend.min()

    for i in np.arange(len(wavelength_line_shift)):

        aa = int(np.ceil((begining_mask_hole[i] - begin_wave)/0.005))
        bb = int(np.ceil((end_mask_hole[i] - begin_wave)/0.005)-1)


        index_begining_mask_hole.append(aa)
        index_end_mask_hole.append(bb)


        freq_step_before_mask_hole.append(wavelength_extend[aa] - wavelength_extend[aa-1])
        freq_step_after_mask_hole.append(wavelength_extend[bb+1] - wavelength_extend[bb])


    mask = np.zeros(wavelength_extend.size)
    a = np.array(index_begining_mask_hole)
    b = np.array(index_end_mask_hole)

    freq_step_before_mask_hole = np.array(freq_step_before_mask_hole)
    freq_step_after_mask_hole = np.array(freq_step_after_mask_hole)


    fraction_pixel_before_mask_hole = np.abs(wavelength_extend[a] - begining_mask_hole)/freq_step_before_mask_hole
    fraction_pixel_after_mask_hole  = np.abs(wavelength_extend[b] - end_mask_hole)/freq_step_after_mask_hole

    for i in np.arange(a.size):

        mask[a[i]:b[i]] = [weight_line[i]]*(b[i]-a[i])
        mask[a[i]-1] = weight_line[i]*fraction_pixel_before_mask_hole[i]
        mask[b[i]] = weight_line[i]*fraction_pixel_after_mask_hole[i]

    return mask


def bisector_measurement(vrad, CCF):

    np.savetxt('CCF_data.txt', np.array([vrad,CCF]).T )


    np.savetxt('CCF_error.txt', np.ones(vrad.size).T )

    cmds = "./BIS_FIT2 "+str(int(vrad.size))
    os.system(cmds)
    bis_c = np.fromfile('ccf_bis.bin',dtype='double')
    depth_c = np.fromfile('ccf_depth.bin',dtype='double')

    return bis_c, depth_c



def BIS_inject_spectral_active(wavelength, flux, bis_coeffs_up, bis_coeffs_low, dep_cut_faculae, vel_cut_faculae, velo_shift_faculae, scaling_factor):

    wave_new = np.zeros(wavelength.size)
    flux_out = np.zeros(wavelength.size)
    for kk in np.arange(flux.size):
        if (flux[kk] < dep_cut_faculae) and (flux[kk] > 0.5):

            rv_shift = (np.poly1d(bis_coeffs_up)(flux[kk])+velo_shift_faculae)*scaling_factor
            wav_shift = Delta_wavelength(rv_shift,wavelength[kk])
            wave_new[kk] = wavelength[kk] + wav_shift

        elif (flux[kk] <= 0.5):

            rv_shift = (np.poly1d(bis_coeffs_low)(flux[kk])+velo_shift_faculae)*scaling_factor
            wav_shift = Delta_wavelength(rv_shift,wavelength[kk])

            wave_new[kk] = wavelength[kk] + wav_shift

        elif (flux[kk] > dep_cut_faculae):

            rv_shift = (vel_cut_faculae+velo_shift_faculae)*scaling_factor
            wav_shift = Delta_wavelength(rv_shift,wavelength[kk])
            wave_new[kk] = wavelength[kk] + wav_shift
        else:
            wave_new[kk] = wavelength[kk]

    spec_func = interp1d(wave_new, flux)
    index = (wavelength < wave_new.max()) & (wavelength > wave_new.min())
    flux_out[index] = spec_func(wavelength[index])

    return flux_out





def BIS_inject_spectral_quiet(wavelength, flux, bis_coeffs, scaling_factor):



    wave_new = np.zeros(wavelength.size)
    flux_out = np.zeros(wavelength.size)
    for kk in np.arange(flux.size):

        rv_shift = np.poly1d(bis_coeffs)(flux[kk])*scaling_factor
        wav_shift = Delta_wavelength(rv_shift,wavelength[kk])

        wave_new[kk] = wavelength[kk] + wav_shift

    spec_func = interp1d(wave_new, flux)
    index = (wavelength < wave_new.max()) & (wavelength > wave_new.min())
    flux_out[index] = spec_func(wavelength[index])

    return flux_out


def cubic_func(teff,a,b):

    S = a*( (teff - 4400)/1000)**3 +b

    return S




def Line_removal(wavelength, spec_seed_spot):


    line_list = np.array([5250.2084, 5250.6453, 5434.5232, 6173.3344, 6301.5008])
    depth_array = np.linspace(0.0,1.0,100)
    line_centers = []
    vel_lines = []
    depth_lines = []

    for ids, lines in enumerate(line_list):

        index = (wavelength > (lines - 5.0) ) & (wavelength < (lines + 5.0))

        spectrum,depth = bisector_measurement_line(wavelength[index], spec_seed_spot[index], spec_seed_spot[index]*0+10**-3, lines)


        line_centers.append(lines)
        vel_lines.append(spectrum['master_bisector'])
        depth_lines.append(depth)


    vels_s = np.concatenate(vel_lines)
    depths_s = np.concatenate(depth_lines)
    vels_cut_s =vels_s.reshape(-1)
    depth_cut_s = depths_s.reshape(-1)

    step_size_BIS = 0.05

    bis_size = (0.8-0.2)/step_size_BIS
    vel_arr_cut = []
    dep_arr_cut = []

    for i_s in np.arange(bis_size):

        index_c =( depths_s >  0.2+i_s*step_size_BIS ) & (depths_s < 0.2+(i_s+1)*step_size_BIS)
        if np.argwhere(index_c).size > 0:
            vel_arr_cut.append(np.mean(vels_cut_s[index_c]))
            dep_arr_cut.append(np.mean(depth_cut_s[index_c]))


    bis_coeffs_s = np.polyfit(np.array(dep_arr_cut), np.array(vel_arr_cut) , deg=2, rcond=None, full=False, w=None, cov=False)
    vel_interpo_s = np.poly1d(bis_coeffs_s)(depth_array)

    return bis_coeffs_s




def BIS_removal_spectral(wavelength, flux, bis_coeffs1, min_b, max_b):


    wave_new = np.zeros(wavelength.size)
    flux_out = np.zeros(wavelength.size)
    for kk in np.arange(flux.size):
        if (flux[kk] < max_b) and (flux[kk] > min_b):
            rv_shift = -1*np.poly1d(bis_coeffs1)(flux[kk])

            wav_shift = Delta_wavelength(rv_shift,wavelength[kk])
            wave_new[kk] = wavelength[kk] + wav_shift


        else:
            wave_new[kk] = wavelength[kk]

    spec_func = interp1d(wave_new, flux)
    index = (wavelength < wave_new.max()) & (wavelength > wave_new.min())
    flux_out[index] = spec_func(wavelength[index])

    return flux_out




config_name = 'config_conti_phoenix.cfg'
config.read(config_name)



input_prefix = config.get('data_io','input_prefix')
input_dir = input_prefix+config.get('data_io','input_dir_source')
wave_array_name = config.get('data_io','wave_array_name')
BIS_dir = input_prefix+config.get('data_io','BIS_prefix')
CCF_windows= float(config.get('data_io','CCF_window'   ))
CCF_size = float(config.get('data_io','CCF_size'   ))



output_dir = input_prefix+config.get('data_io','output_dir')
wave_data = output_dir+wave_array_name
wavelength = np.fromfile(wave_data,dtype='double')


vrad_ccf2 = np.arange(-1*CCF_windows,CCF_windows+1.,CCF_size)



nb_zeros_on_sides = 5
period_appodisation = int(((len(vrad_ccf2)-1)/2.))
len_appodisation = int(period_appodisation/2.)
a = np.arange(len(vrad_ccf2))
b = 0.5*np.cos(2*np.pi/period_appodisation*a-np.pi)+0.5
appod = np.concatenate([np.zeros(nb_zeros_on_sides),b[:len_appodisation],np.ones(len(vrad_ccf2)-period_appodisation-2*nb_zeros_on_sides),b[:len_appodisation][::-1],np.zeros(nb_zeros_on_sides)])


vector_size = int(wavelength.size)
step_size = round(wavelength[3]-wavelength[2],5)

wave_extend = 100.0





coeff_frame = np.load(BIS_dir+'CB_scale_liebing.npz', allow_pickle=True)
coeff_slope = coeff_frame['coeff_slope']
coeff_offset = coeff_frame['coeff_offset']


diffbis_coeffs_frame = np.load(BIS_dir+'coeff_BIS_faculae.npz', allow_pickle=True)


mu_array_faculae = diffbis_coeffs_frame['mu_array']
coeff_up_fac_array = diffbis_coeffs_frame['coeff_up_fac']
coeff_low_fac_array = diffbis_coeffs_frame['coeff_low_fac']
dep_cut_fac_array = diffbis_coeffs_frame['dep_cut_fac']
velo_cut_fac_array = diffbis_coeffs_frame['velo_cut_fac']
velo_shift_fac_array = diffbis_coeffs_frame['velo_shift_fac']



depth = np.linspace(0.0,1.0,100)
coeff_frame = np.load(BIS_dir+'coeff_mu_v1.npz',allow_pickle=True)
mu_coeff = coeff_frame['coeff_obs']
mu_array = coeff_frame['mus']


vel_interpo_mu2 = np.poly1d(mu_coeff[0]) (depth)
vel_interpo_mu8 = np.poly1d(mu_coeff[7]) (depth)

index = np.argmin(vel_interpo_mu8-vel_interpo_mu2)

Temp  = int(config.get('star','Tstar'))


tstar_array = np.array([Temp])

for i_teff in np.arange(tstar_array.size):



    tstar  = tstar_array[i_teff]

    tspot  = tstar - int(config.get('star','Tdiff_spot' ))
    tfaculae = tstar + int(config.get('star','Tdiff_faculae' ))

    T_eff  = np.array([tstar, tspot, tfaculae])

    mask_template = input_prefix+config.get('data_io','mask_prefix')+mask_templates(tstar)


    templates = np.loadtxt(mask_template)
    freq_line = templates[:,0]
    contrast_line = templates[:,1]
    index_lines = (freq_line > wavelength.min()) & (freq_line < wavelength.max()) & (contrast_line > 0.1)
    freq_line = freq_line[index_lines]
    contrast_line =contrast_line[index_lines]

    scale1= cubic_func(tstar, coeff_slope, coeff_offset)/np.abs((vel_interpo_mu8[index]-vel_interpo_mu2[index]))

    colors = plt.cm.jet(np.linspace(1,0,mu_array.size))

    factor_rescale = 10**15


    no_points = int(wave_extend/step_size)
    wavelength_before = np.linspace(np.min(wavelength)-step_size*no_points,np.min(wavelength), no_points)
    wavelength_after = np.linspace(wavelength[-1], wavelength[-1]+step_size*no_points, no_points)
    wavelength_extend = np.concatenate((wavelength_before,wavelength,wavelength_after))

    mask_template = calculate_CCF_1(vrad_ccf2,wavelength,freq_line,contrast_line, wavelength_extend)





    for i_t in np.arange(T_eff.size):
        print(i_t, T_eff[i_t])

        spec_name = input_dir+'/RASSINE_New_Phoenix_'+str(int(T_eff[i_t]))+'.p'

        seed_frame = pd.read_pickle(spec_name)
        spec_normed = seed_frame['flux']/seed_frame['output']['continuum_cubic']
        spec_contin = seed_frame['output']['continuum_cubic']

        spec_wave = seed_frame['wave']

        spec_func_contin = interp1d(spec_wave, spec_contin)
        spec_seed_contin = spec_func_contin(wavelength)/factor_rescale

        spec_func = interp1d(spec_wave, spec_normed)
        spec_seed = spec_func(wavelength)

        coeff_lines = Line_removal(wavelength, spec_seed)

        vel_inter = np.poly1d(coeff_lines)(depth)

        CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, spec_seed, mask_template)
        CCF_quiet_Sun /= np.max(CCF_quiet_Sun)
        CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)
        bis_c, depth_c = bisector_measurement(vrad_ccf2, CCF_quiet_Sun)
        depth_array_fit = np.linspace(0,1.0,100)


        spec_strong = spec_seed.copy()

        for iter in np.arange(1):


            index_cp = (depth < 0.85) & (depth > depth_c.min())
            vel_off = np.mean(bis_c[(depth_c < 0.85) & (depth_c > depth_c.min())]) - np.mean(vel_inter[index_cp])


            coeff_lines[-1] = coeff_lines[-1] +vel_off



            flux_new = BIS_removal_spectral(wavelength, spec_strong, coeff_lines, 0, 1)



            CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, flux_new, mask_template)
            CCF_quiet_Sun /= np.max(CCF_quiet_Sun)
            CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)




            bis_c1, depth_c1 = bisector_measurement(vrad_ccf2, CCF_quiet_Sun)

            bis_cut =  0.8
            span_c1 = bis_c1[depth_c1 < bis_cut].max()-bis_c1[depth_c1 < bis_cut].min()
            span_c = bis_c[depth_c < bis_cut].max() - bis_c[depth_c< bis_cut].min()

            spec_strong = flux_new.copy()


        if (np.abs(T_eff[i_t]-tstar) <= 0.1):

            seed_cube = np.zeros((mu_array.size,vector_size))
            seed_cube_contin = np.zeros((mu_array.size,vector_size))
            seed_cube_contin_conv = np.zeros((mu_array.size,vector_size))

            for ids, mus in enumerate(mu_array):
                print(ids,mus)
                flux_inj = BIS_inject_spectral_quiet(wavelength, flux_new, mu_coeff[ids], scale1)

                CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, flux_inj, mask_template)
                CCF_quiet_Sun /= np.max(CCF_quiet_Sun)
                CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)


                seed_cube[ids,:] = flux_inj[:vector_size]
                seed_cube_contin_conv[ids,:] = flux_inj[:vector_size]  *  spec_seed_contin[:vector_size]
                seed_cube_contin[ids,:] = spec_seed_contin[:vector_size]



                bis_c1, depth_c1 = bisector_measurement(vrad_ccf2, CCF_quiet_Sun)
                vel_interpo = np.poly1d(mu_coeff[ids])(depth_c1)




            out_name = output_dir+'/new_noconti_convec_T_eff_'+str(int(T_eff[i_t]))+'.bin'
            seed_cube.tofile(out_name)

            out_name1 = output_dir+'/new_conti_convec_T_eff_'+str(int(T_eff[i_t]))+'.bin'
            seed_cube_contin_conv.tofile(out_name1)

            out_name2 = output_dir+'/new_conti_T_eff_'+str(int(T_eff[i_t]))+'.bin'
            seed_cube_contin.tofile(out_name2)

        else:
            seed_cube = np.zeros((mu_array.size,vector_size))
            seed_cube_contin = np.zeros((mu_array.size,vector_size))
            seed_cube_contin_conv = np.zeros((mu_array.size,vector_size))

            print('Inject BIS difference')


            for ids, mus in enumerate(mu_array):
                print(ids,mus)

                index_min_fac = np.argmin(np.abs(mu_array_faculae - mus) )
                print('use '+str(mu_array_faculae[index_min_fac]))

                flux_inj = BIS_inject_spectral_active(wavelength, flux_new, coeff_up_fac_array[index_min_fac,:], coeff_low_fac_array[index_min_fac,:], \
                                        dep_cut_fac_array[index_min_fac], velo_cut_fac_array[index_min_fac],velo_shift_fac_array[index_min_fac], scale1)

                seed_cube[ids,:] = flux_inj[:vector_size]
                seed_cube_contin_conv[ids,:] = flux_inj[:vector_size]  *  spec_seed_contin[:vector_size]
                seed_cube_contin[ids,:] = spec_seed_contin[:vector_size]

                CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, flux_inj, mask_template)
                CCF_quiet_Sun /= np.max(CCF_quiet_Sun)
                CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)


                bis_c1, depth_c1 = bisector_measurement(vrad_ccf2, CCF_quiet_Sun)
                vel_interpo = np.poly1d(mu_coeff[-1])(depth_c1)




            out_name = output_dir+'/new_noconti_convec_T_eff_'+str(int(T_eff[i_t]))+'.bin'
            seed_cube.tofile(out_name)

            out_name1 = output_dir+'/new_conti_convec_T_eff_'+str(int(T_eff[i_t]))+'.bin'
            seed_cube_contin_conv.tofile(out_name1)

            out_name2 = output_dir+'/new_conti_T_eff_'+str(int(T_eff[i_t]))+'.bin'
            seed_cube_contin.tofile(out_name2)


    os.system('rm -r ccf*bin')
    os.system('rm -r CCF*txt')
