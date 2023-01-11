import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib
from matplotlib import rc,rcParams
import glob
import configparser
config = configparser.ConfigParser()

config_name = glob.glob('./*.cfg')[0]
config.read(config_name)


raw_dir = config.get('data_io','raw_dir')
Temp  = int(config.get('star','Tstar'))
file_name_prefix = config.get('data_io','file_prefix')
GRID   = int(config.get('main','grid' ))
incl   = float(config.get('star','I'))
limba1 = float(config.get('star','limb1'   ))
limba2 = float(config.get('star','limb2'   ))
Temp_diff_spot = int(config.get('star','Tdiff_spot'))
Temp_diff_faculae = int(config.get('star','Tdiff_faculae'))
def differential_rot(theta):
    omega0 = 14.523 #degree/day
    omega1 = -2.688
    omega2 = 0

    omega = omega0 + omega1*np.sin(theta*np.pi/180)**2+omega2*np.sin(theta*np.pi/180)**4
    R_sun = 696400 #km
    velo = omega*np.pi/(180*24*3600)*R_sun#*np.cos(theta*np.pi/180)

    return velo

matplotlib.rc('axes.formatter', useoffset=False)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


light_maps = np.zeros((GRID , GRID ))
light_maps[:,:] = np.nan

differential_maps = np.zeros((GRID , GRID ))
differential_maps[:,:] = np.nan

i = incl * np.pi/180.
delta_grid = 2./GRID
theta_grid = 180.0/GRID

for iz in np.arange(GRID):

    z = -1. + iz*delta_grid
    theta_z = -90.0+ iz*theta_grid
    omega_d = differential_rot(theta_z)


    for iy in np.arange(GRID):

        y = -1. + iy*delta_grid
        delta_d = y * omega_d * np.sin(i)


        if (y*y+z*z)<=1.0:
            r_cos = (1.-(y*y+z*z) )**0.5
            limb =  1. - limba1*(1-r_cos) - limba2*(1-r_cos)*(1-r_cos)

            light_maps[iz,iy] = limb
            differential_maps[iz,iy] = delta_d


phase_out_txt = file_name_prefix+'_'+'phase_locator.txt'
time_frame = np.loadtxt(phase_out_txt)

time_array = time_frame
x = np.arange(0, GRID)
y = np.arange(0, GRID)



t = np.arange(0, time_array.size)

X3, Y3, T3 = np.meshgrid(x, y, t)
G =np.ones((X3.shape))
M =np.ones((X3.shape))


for i_m in t:
    M[:,:,i_m] = differential_maps
    G[:,:,i_m] = light_maps


for i_point in np.arange(t.size):

    file1 = '.'+raw_dir+file_name_prefix+'_temp_map_'+str(int(i_point))+'_.bin'
    temp_data = np.fromfile(file1,dtype='double').reshape((GRID,GRID))


    for ix in np.arange(GRID):
        for iy in np.arange(GRID):


            if (temp_data[ix,iy] != (Temp-Temp_diff_spot)) and (temp_data[ix,iy] > 0.0):

                G[iy,ix,i_point] = 100.0
                M[iy,ix,i_point] = 100.0


    for ix in np.arange(GRID):
        for iy in np.arange(GRID):


           if temp_data[ix,iy] == (Temp-Temp_diff_spot):

                G[iy,ix,i_point] = np.nan
                M[iy,ix,i_point] =np.nan


frame_name = file_name_prefix+'_GPU_SOAP_RV_data.npz'
RV_frames = np.load(frame_name)
tot_rvs = RV_frames['tot']



print('start to plot...')
def animate(i):

     cax1.set_array(M[:-1, :-1, i].flatten())
     ax1.set_title('Veolcity field (Rotation Phase %.3f)' % + float(time_array[i]))

     cax2.set_data(time_array[i], tot_rvs[i])

fig1 = plt.figure(figsize=(24, 12))
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)


ax1.set(xlim=(0, GRID), ylim=(0, GRID))

cax1 = ax1.pcolormesh(x, y, M[:-1, :-1, 0],shading='auto',cmap=plt.cm.get_cmap('jet'))
cax1.set_clim(vmin=-3, vmax=3)

cax2 = ax2.plot(time_array, tot_rvs,color='black',marker='.',linestyle='None')
cax2 = ax2.plot(time_array, tot_rvs,color='red',marker='o',linestyle='None')[0]
ax2.set_xlabel('Phase')
ax2.set_ylabel('RV [m/s]')
ax2.set_title('Tot RV')



anim = FuncAnimation(
    fig1, animate, interval=100, frames=len(t))



output_gif=file_name_prefix+'_GPU_SOAP_visual.gif'
anim.save(output_gif, writer='pillow')
