import numpy as np
import matplotlib.pyplot as plt
import os
import time
import configparser
from CCF_func import *

def asy_func(Amax, tmax, D, A, t):

    t_part = -1*(t-tmax)**2
    A_part = D*(1+A*(t-tmax))
    f = Amax*np.exp(t_part/A_part)

    return f

def spot_evo(Amax,d_rate, t):


    curve_out = Amax - d_rate * t

    return curve_out


def LB_boundary(t_eff):

    if t_eff > 5778:
        boundary_LB = np.array([0,1])
    elif (t_eff <=5778) and (t_eff >= 5240):
        boundary_LB = np.array([0,1])

    elif (t_eff <5240) and (t_eff >= 3750):
        boundary_LB = np.array([1,2])
    elif t_eff < 3750:
        boundary_LB = np.array([1,2])
    return boundary_LB

def LB_params(T_star, G2_LB_template, K0_LB_template, M0_LB_template):
    mu_array = np.linspace(0.1,1,10)
    lb = np.zeros(mu_array.size)
    boundary_LB = LB_boundary(T_star)
    for id in np.arange(10):

        LB_derived_array = np.array([G2_LB_template[id], K0_LB_template[id], M0_LB_template[id]])
        params = np.polyfit(temp_derived_array[boundary_LB], LB_derived_array[boundary_LB], 1)
        funcs = np.poly1d(params)
        lb[id] = funcs(T_star)

    params_LB = np.polyfit(mu_array, lb, 3)

    return params_LB



def mask_templates(Teff_star):

    mask_type = np.array(['F9', 'G2', 'G8', 'G9', 'K2', 'K5', 'K6', 'M0', 'M2', 'M3', 'M4', 'M5'])
    Teff_template_array = np.array([6050, 5770, 5480, 5380, 5100, 4400, 4300, 3800, 3400, 3250, 3100, 2800])

    index = np.argmin(np.abs(Teff_template_array - Teff_star))
    mask_template_name = mask_type[index]+'_mask.txt'

    return mask_template_name


c = 299792458.
config = configparser.ConfigParser()


config_name = 'config_hpc.cfg'
config.read(config_name)

GRID        = int(config.get('main','grid' ))
NRHO        = int(config.get('main','nrho' ))
INST_RESO   = int(config.get('main','instrument_reso' ))
rot_num       = float(config.get('main','rot_num' ))
rot_step   = float(config.get('main','rot_step' ))


if INST_RESO > 0:
    inst_profile_FWHM = c/float(INST_RESO)
    inst_profile_sigma = inst_profile_FWHM/(2*np.sqrt(2*np.log(2)))
RAD_Sun     = int(config.get('star','radius_sun' ))
RAD         = float(config.get('star','radius' )) * RAD_Sun
PROT        = float(config.get('star','prot' ))
omega_ratio = float(config.get('star','W_ratio' ))
incl   = float(config.get('star','I'))
limba1 = float(config.get('star','limb1'   ))
limba2 = float(config.get('star','limb2'   ))


Temp  = int(config.get('star','Tstar'))



Temp_diff_spot = int(config.get('star','Tdiff_spot'))
Temp_diff_faculae = int(config.get('star','Tdiff_faculae'))


N_act   = int(config.get('star','N_act'))

input_prefix = config.get('data_io','input_prefix')
input_dir = input_prefix+config.get('data_io','input_dir')
output_prefix = config.get('data_io','output_prefix')+'_phoenix'+str(Temp)
maps_dir = output_prefix+config.get('data_io','maps_dir')
raw_dir = output_prefix+config.get('data_io','raw_dir')
final_spec_dir = output_prefix+config.get('data_io','spec_dir')
file_name_prefix = config.get('data_io','file_prefix')


solar_name =  'new_noconti_convec_T_eff_'+str(Temp)+'.bin'
spot_name = 'new_noconti_convec_T_eff_'+str(Temp-Temp_diff_spot)+'.bin'
faculae_name = 'new_noconti_convec_T_eff_'+str(Temp+Temp_diff_faculae)+'.bin'

solar_SED_spectra_name = 'new_conti_convec_T_eff_'+str(Temp)+'.bin'
spot_SED_name = 'new_conti_T_eff_'+str(Temp-Temp_diff_spot)+'.bin'
faculae_SED_name = 'new_conti_T_eff_'+str(Temp+Temp_diff_faculae)+'.bin'


mu_array_name = config.get('data_io','mu_array_name')
wave_array_name = (config.get('data_io','wave_array_name'))


CCF_windows= float(config.get('data_io','CCF_window'   ))
CCF_size = float(config.get('data_io','CCF_size' ))
keyword = config.get('data_io','keyword')

solar_data = input_dir+solar_name
spot_data = input_dir+spot_name
faculae_data = input_dir+faculae_name

solar_SED_spectra_data = input_dir+solar_SED_spectra_name
spot_SED_data = input_dir+spot_SED_name
faculae_SED_data = input_dir+faculae_SED_name

mu_array_data = input_dir+mu_array_name
wave_data = input_dir+wave_array_name


wavelength_step = np.fromfile(wave_data,dtype='double')
step_size = round(wavelength_step[3]-wavelength_step[2],5)
vector_size = int(wavelength_step.size)

no_thread = int(config.get('data_io','no_thread'))
no_block = int(vector_size/no_thread)

spec_step = np.fromfile(mu_array_data,dtype='double').size
convec_no_spec = int(spec_step+1)

lb_frame = np.load(input_dir+'Limb_brightening_info.npz')

LB_G2_array = lb_frame['LB_G2']
LB_K0_array = lb_frame['LB_K0']
LB_M0_array = lb_frame['LB_M0']
temp_derived_array = lb_frame['Temp_array']

params_LB = LB_params(Temp, LB_G2_array, LB_K0_array, LB_M0_array)[::-1]

LB0 = params_LB[0]
LB1 = params_LB[1]
LB2 = params_LB[2]
LB3 = params_LB[3]


if keyword =='real':

    if not os.path.isdir(raw_dir):
        os.mkdir(raw_dir)

    if not os.path.isdir(final_spec_dir):
        os.mkdir(final_spec_dir)

    phase_out_txt = output_prefix+'/'+file_name_prefix+'_'+'phase_locator.txt'
    phase_out = np.loadtxt(phase_out_txt)


else:

    if os.path.isdir(output_prefix):
        os.system('rm -r '+output_prefix)

    if not os.path.isdir(output_prefix):
        os.mkdir(output_prefix)

    if not os.path.isdir(maps_dir):
        os.mkdir(maps_dir)

    if not os.path.isdir(raw_dir):
        os.mkdir(raw_dir)

    if not os.path.isdir(final_spec_dir):
        os.mkdir(final_spec_dir)


    long_list = np.zeros(N_act)
    lat_list = np.zeros(N_act)
    size_list = np.zeros(N_act)
    type_list = np.zeros(N_act)

    for act_id in np.arange(N_act):
        keys = 'active_regions_'+str(int(act_id+1))
        long_list[act_id] = float(config.get(keys,'longitude'))
        lat_list[act_id] = float(config.get(keys,'latitude'))
        size_list[act_id] = float(config.get(keys,'size'))
        type_list[act_id] = float(config.get(keys,'type'))


    phase_out = np.arange(0.0,rot_num,rot_step)

    phase_out_txt = output_prefix+'/'+file_name_prefix+'_'+'phase_locator.txt'
    np.savetxt(phase_out_txt, phase_out.T)
    spot_info = np.zeros((phase_out.size,2))

########add size evolution here, this is only valid when N = 1:
    t_size = (phase_out - phase_out.min())*PROT
    Amax = size_list**2
    # size_curve = asy_func(Amax, 10.0, 20.0, 0.09, t_size)**0.5
    # size_curve = spot_evo(Amax, 0.0004,t_size)**0.5
    size_curve = spot_evo(Amax, -0.002,t_size)**0.5

    plt.plot(t_size, size_curve)
    plt.show()

    for id in np.arange(phase_out.size):


        lat_array = lat_list
        long_array = long_list
        type_array = type_list

        size_array = size_curve[id].reshape(1)

        out_cube = np.stack((lat_array, long_array, size_array, type_array),axis=1)


        out_txt = maps_dir+'maps'+str(int(id))+'_for_all.txt'
        np.savetxt(out_txt, out_cube)

        spot_info[id,0] = id
        spot_info[id,1] = type_array.size


    f_txt = maps_dir+'final_info.txt'
    np.savetxt(f_txt, spot_info)




start = time.time()
cmds0 = "./SOAP_initialize "+str(vector_size)+" "+str(PROT)+" "+str(omega_ratio)+" "+str(incl)+" "+str(RAD)+" "+str(limba1)+" "+str(limba2)+" "+str(GRID)+" "+str(inst_profile_sigma)+" "+str(step_size)+" "+str(step_size)+" "+str(50.0)\
        +" "+str(no_block)+" "+str(no_thread)+" "+solar_data+" "+output_prefix+"/"+file_name_prefix+" "+str(convec_no_spec)+" "+mu_array_data+" "+wave_data+" "+str(LB0)+" "+str(LB1)+" "+str(LB2)+" "+str(LB3)

os.system(cmds0)

cmds1 = "./SOAP_initialize "+str(vector_size)+" "+str(PROT)+" "+str(omega_ratio)+" "+str(incl)+" "+str(RAD)+" "+str(limba1)+" "+str(limba2)+" "+str(GRID)+" "+str(inst_profile_sigma)+" "+str(step_size)+" "+str(step_size)+" "+str(50.0)\
        +" "+str(no_block)+" "+str(no_thread)+" "+solar_SED_spectra_data+" "+output_prefix+"/"+file_name_prefix+"_SED"+" "+str(convec_no_spec)+" "+mu_array_data+" "+wave_data+" "+str(LB0)+" "+str(LB1)+" "+str(LB2)+" "+str(LB3)

os.system(cmds1)


cmds2 = "./SOAP_integration "+str(GRID)+" "+str(NRHO)+" "+str(int(phase_out.size))+" "+str(Temp)+" "+str(Temp_diff_spot)\
        +" "+str(incl)+" "+str(Temp_diff_faculae)+" "+str(phase_out_txt)+" "+str(faculae_data) \
        +" "+str(vector_size)+" "+raw_dir+" "+maps_dir\
        +" "+str(no_block)+" "+str(no_thread)+" "+solar_data+" "+spot_data+" "+file_name_prefix+" "+str(step_size)+" "+str(convec_no_spec)+" "+wave_data\
        +" "+solar_SED_spectra_data+" "+spot_SED_data+" "+faculae_SED_data+" "+output_prefix+"/"



os.system(cmds2)

cmds3 = "./SOAP_resolution "+str(int(phase_out.size))+" "+str(vector_size)+" "+str(inst_profile_sigma)+" "+raw_dir+" "+final_spec_dir\
        +" "+str(no_block)+" "+str(no_thread)+" "+file_name_prefix+" "+output_prefix+"/"+file_name_prefix

os.system(cmds3)


start2 = time.time()
print("GPU Calculation time (s):", start2 - start)

start3 = time.time()
print('Start to calculate RVs')
frame_size = phase_out.size
cutoff = 200


vrad_ccf2 = np.arange(-1*CCF_windows,CCF_windows+1.,CCF_size)



cutoff_new = 10
wave_extend = 100.0


wavelength = np.fromfile(output_prefix+"/"+file_name_prefix+'_quiet_sun_wave.bin',dtype='double')[cutoff:-1*cutoff]
wavelength_quiet = wavelength

fstar_quiet_sun =np.fromfile(output_prefix+"/"+file_name_prefix+'_quiet_sun_lower_spec.bin',dtype='double')[cutoff:-1*cutoff]
quiet_spec = fstar_quiet_sun[cutoff_new:-1*cutoff_new]


fstar_quiet_sun_SED =np.fromfile(output_prefix+"/"+file_name_prefix+'_SED_quiet_sun_lower_spec.bin',dtype='double')[cutoff:-1*cutoff]
quiet_spec_SED = fstar_quiet_sun_SED[cutoff_new:-1*cutoff_new]


wavelength = wavelength_quiet[cutoff_new:-1*cutoff_new]

mask_template = input_prefix+config.get('data_io','mask_prefix')+mask_templates(Temp)

templates = np.loadtxt(mask_template)
freq_line = templates[:,0]
contrast_line = templates[:,1]
index_lines = (freq_line > wavelength.min()) & (freq_line < wavelength.max()) & (contrast_line > 0.1)
freq_line_G2 = freq_line[index_lines]
contrast_line_G2 =contrast_line[index_lines]

no_points = int(wave_extend/step_size)
wavelength_before = np.linspace(np.min(wavelength)-step_size*no_points,np.min(wavelength), no_points)
wavelength_after = np.linspace(wavelength[-1], wavelength[-1]+step_size*no_points, no_points)
wavelength_extend = np.concatenate((wavelength_before,wavelength,wavelength_after))
mask_template = calculate_CCF_1(vrad_ccf2,wavelength,freq_line_G2,contrast_line_G2, wavelength_extend)

CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, quiet_spec, mask_template)
CCF_quiet_Sun /= np.max(CCF_quiet_Sun)

CCF_quiet_Sun_SED = calculate_CCF_2(vrad_ccf2, quiet_spec_SED, mask_template)
CCF_quiet_Sun_SED /= np.max(CCF_quiet_Sun_SED)



nb_zeros_on_sides = 5
period_appodisation = int(((len(vrad_ccf2)-1)/2.))
len_appodisation = int(period_appodisation/2.)
a = np.arange(len(vrad_ccf2))
b = 0.5*np.cos(2*np.pi/period_appodisation*a-np.pi)+0.5
appod = np.concatenate([np.zeros(nb_zeros_on_sides),b[:len_appodisation],np.ones(len(vrad_ccf2)-period_appodisation-2*nb_zeros_on_sides),b[:len_appodisation][::-1],np.zeros(nb_zeros_on_sides)])

CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)
CCF_quiet_Sun_SED  = 1-((-CCF_quiet_Sun_SED+1)*appod)

np.savetxt('CCF_data.txt', np.array([vrad_ccf2,CCF_quiet_Sun]).T )
np.savetxt('CCF_error.txt', np.ones(vrad_ccf2.size).T )

cmds = "./BIS_FIT2 "+str(int(vrad_ccf2.size))
os.system(cmds)
model_c = np.fromfile('ccf_model.bin',dtype='double')
par_c = np.fromfile('ccf_parameter.bin',dtype='double')
bis_c = np.fromfile('ccf_bis.bin',dtype='double')
depth_c = np.fromfile('ccf_depth.bin',dtype='double')


types_array = ['tot', 'flux', 'bconv']

for type_id in np.arange(3):
    types = types_array[type_id]

    max_num =frame_size+1

    ccf_cube = np.zeros((max_num, vrad_ccf2.size))
    model_cube = np.zeros((max_num, vrad_ccf2.size))
    continuum_array = np.zeros(max_num)
    contrast_array = np.zeros(max_num)
    span_array = np.zeros(max_num)
    vrad_array = np.zeros(max_num)
    fwhm_array = np.zeros(max_num)
    depth_cube= np.zeros((max_num, 91))
    bis_cube = np.zeros((max_num, 91))

    data_cube_tot = np.zeros((max_num, wavelength.size))
    for i in np.arange(max_num):

        print(i,' over ',frame_size,' ',types)
        spec_name = file_name_prefix+'_Spec_'+types+'_%s_.bin' % int(i)
        lower_name = file_name_prefix+'_lower_spec_'+types+'_%s_.bin' % int(i)

        if os.path.isfile(final_spec_dir+lower_name):


            fstar_active_sun_tmp = np.fromfile(final_spec_dir+lower_name,dtype='double')[cutoff:-1*cutoff]
            fstar_active_sun = fstar_active_sun_tmp


        else:
            if types == 'bconv':
                fstar_active_sun = fstar_quiet_sun
            else:
                fstar_active_sun = fstar_quiet_sun_SED


        spec_active = fstar_active_sun[cutoff_new:-1*cutoff_new]
        data_cube_tot[i,:] = spec_active

        CCF_active_sun = calculate_CCF_2(vrad_ccf2, spec_active, mask_template)
        CCF_active_sun /=np.max(CCF_active_sun)
        CCF_active_sun  = 1-((-CCF_active_sun+1)*appod)

        np.savetxt('CCF_data.txt', np.array([vrad_ccf2,CCF_active_sun]).T )
        np.savetxt('CCF_error.txt', np.ones(vrad_ccf2.size).T )

        cmds = "./BIS_FIT2 "+str(int(vrad_ccf2.size))
        os.system(cmds)


        model_c = np.fromfile('ccf_model.bin',dtype='double')
        par_c = np.fromfile('ccf_parameter.bin',dtype='double')
        bis_c = np.fromfile('ccf_bis.bin',dtype='double')
        depth_c = np.fromfile('ccf_depth.bin',dtype='double')

        print(i,par_c[3])

        ccf_cube[i,:] = CCF_active_sun
        model_cube[i,:] = model_c
        continuum_array[i] = par_c[0]
        contrast_array[i] = par_c[1]
        span_array[i] = par_c[2]
        vrad_array[i] = par_c[3]
        fwhm_array[i] = par_c[4]
        depth_cube[i,:]= depth_c
        bis_cube[i,:] = bis_c

    vrad_spectrum = vrad_array
    vrad_spectrum -= vrad_spectrum[-1]


    data_out = {}
    data_out['time'] = phase_out
    data_out['RV'] = vrad_spectrum[:-1]
    data_out['ccf'] = ccf_cube
    data_out['model'] = model_cube
    data_out['continuum'] = continuum_array
    data_out['contrast'] =contrast_array
    data_out['span']= span_array
    data_out['fwhm']=fwhm_array
    data_out['depth']=depth_cube
    data_out['bis']=bis_cube
    out_name = output_prefix+'/'+file_name_prefix+'_'+'RV_'+types+'_GPU.npz'
    np.savez(out_name,**data_out)


end = time.time()


print("RV Calculation time (s):", end - start3)
print("Computing time (s):", end - start)


os.system('rm -r ccf*bin')
os.system('rm -r CCF*txt')

data1 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_tot_GPU.npz')
data2 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_flux_GPU.npz')
data3 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_bconv_GPU.npz')



plt.figure(figsize = (10, 6))

plt.plot(data1['time'],data1['RV'],color = 'red', linestyle='None', marker='.',label='tot')
plt.plot(data1['time'],data2['RV'],color = 'blue', linestyle='None', marker='.',label='flux')
plt.plot(data1['time'],data3['RV'],color = 'green', linestyle='None', marker='.',label='bconv')


plt.legend(loc='best')
plt.ylabel('RV [m/s]')
plt.xlabel('Phase')
plt.savefig(output_prefix+'/'+file_name_prefix+'_'+'RV_plots.png')

data_out = {}
data_out['time'] = data1['time']
data_out['tot'] =  data1['RV']
data_out['flux'] =  data2['RV']
data_out['bconv'] =  data3['RV']
out_name = output_prefix+'/'+file_name_prefix+'_GPU_SOAP_RV_data.npz'
np.savez(out_name,**data_out)


cmd_cp = 'cp '+config_name+' '+output_prefix+'/'+file_name_prefix+'_config_file.cfg'
os.system(cmd_cp)
cmd_cp = 'cp  Plot_animation.py '+output_prefix+'/'+file_name_prefix+'_Plot_animation.py'
os.system(cmd_cp)
