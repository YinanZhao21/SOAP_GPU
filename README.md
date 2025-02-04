INTRODUCTION
-----------------------------

SOAP-GPU is a revised SOAP 2.0 code (Dumusque et al. 2014, ApJ, 796, 132) that simulate spectral time series with the effect of active regions (spot, faculae or both).
In addition to the traditional outputs of SOAP 2.0 (the cross-correlation function and extracted parameters: radial velocity, bisector span, full width at half maximum),
SOAP-GPU generates the integrated spectra at each phase for given input spectra and spectral resolution.

Five major improvements are implemented compared to the SOAP 2.0 code.
1) Spectral simulation of stellar activity can be fast performed with GPU acceleration.
2) SOAP-GPU can simulate more complicated active region structures, with superposition between active regions.
3) SOAP-GPU implements more realistic line bisectors, based on solar observations, that varies as function of mu angle for both quiet and active regions.
4) SOAP-GPU can accept any input high resolution observed spectra. The PHOENIX synthetic spectral library are already implemented at the code level which
allows users to simulate stellar activity for stars other than the Sun.
5) SOAP-GPU can simulate realistic spectral time series with either spot number/SDO image as additional inputs.

The code is published in Zhao & Dumusque et al. 2023 and more details are shown in that paper.


INSTALLATION AND RUNNING SOAP-GPU
-----------------------------

The SOAP GPU code is written in C, with python scripts for input pre-processing and output post-processing.
The code has been tested on python 3.7 and CUDA 10.1 and though it is expected to work properly on more recent versions, the user is advised to use those versions in case of troubles.

Necessary softwares required:
-----------------------------

- CUDA Toolkit 10.1 or above (You may also need to install the proper version NVIDIA driver, for more information on installation, please see: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- GSL 2.0 or above.
- RASSINE python code (For more information, please see: https://github.com/MichaelCretignier/Rassine_public).


To install RASSINE you can simply do
------------------------------------

$git clone https://github.com/MichaelCretignier/Rassine_public


To install SOAP-GPU, after unziping the zip file:
-------------------------------------------------

$cd SOAP_GPU

$bash install_SOAP_GPU.sh


To allow RASSINE to properly normalise the spectra of K dwarfs:
------------------------------------------------------------------

Copy the files into the folder SOAP_GPU/Rassine_file/ into the folder of the RASSINE code

$cp Rassine_file/* PATH_TO_RASSINE_CODE/.



Before running SOAP-GPU, there are two config files that should be updated:
---------------------------------------------------------------------------

1) config_conti_phoenix.cfg (this is the config file for PHOENIX inputs).

Change the path of "input_prefix" to the one of your current installation of SOAP-GPU
Change the path of "Rassine_directory" to the one of your current installation of RASSINE

2) config_hpc.cfg (this is the main config file to run SOAP-GPU, Each input variables are described in the config file)

Change the path of "input_prefix" to the one of your current installation of SOAP-GPU
Change the path of "output_prefix" to an existing directory where the outputs of the code will be saved


To run SOAP-GPU:
----------------

1) Run phoenix_extraction.py to download the needed PHEONIX spectra (If you use custom input seed spectra, you can skip this step).
This will download the phoenix spectra for the stellar photosphere, the spot and facula region
with the effective temperature and long as configured in config_conti_phoenix.cfg

2) Bisector injection as function of mu angle for both quiet and active seed spectra.
There are two python scripts:
- CB_injection_faculae.py is the script to inject proper bisectors for the PHOENIX library inputs.

- CB_injection_faculae_obs.py is the script for custom input seed spectra (Note the hardcoded parameters are only valid for the
observed spectra of the Sun taken with the Kitt Peak Fourrier Transform Spectrograph (FTS). User may need to adapt the parameters for other inputs).

3) Run run_all_SOAP.py to launch the SOAP-GPU simulation.
If user plan to run SOAP-GPU on server, please modify run.sbatch instead. The output of sbatch will be written in the file "slurm_SLURMID.out"
Examples for different inputs can be found in the directory /example/.


Outputs:
----------------

All the outputs can be found in the directory 'output_prefix' configured in config_hpc.cfg). Those include:
- A .png figure of the corresponds radial velocity data.
- Three .npz files for corresponding radial velocity data, bisector span, FWHM, CCFs and wavelength range of
the output spectra for the total, flux and convective blueshift effects.
A .npz file can be read using data=numpy.load(FILE.npz) and the keys of the object can be accessed by data.files
- A python script Plot_animation.py that can generate an animated GIF of the active regions present on the disk surface as a function of time.
- The spectra at the resolution configured in config_hpc.cfg can be found in the folder defined by the keyword 'spec_dir' in the config_hpc.cfg file).
Those can be read by doing numpy.fromfile('SPECTRUM_NAME.bin', dtype='double'). The wavelength is given in the .npz file



TROUBLESHOOTING
-----------------------------

If when running the code "phoenix_extraction.py" you get the following error message:
"ImportError: Cannot load backend 'Qt5Agg' which requires the 'qt5' interactive framework, as 'headless' is currently running"
either install Qt5, or comment the line 23 and 12 ("matplotlib.use('Qt5Agg',force=True)") in Rassine.py and Rassine_functions.py
inside your installation if the RASSINE code.


If when running the code "CB_injection_faculae.py" you get the following error message:
"./BIS_FIT2: error while loading shared libraries: libgsl.so.25: cannot open shared object file: No such file or directory"
the code cannot find a GSL library. Make sure GSL is installed


CONTACT
-----------------------------
Yinan Zhao

zhaoyinan2121@gmail.com

Observatoire de Genève
51 Chemin Pegasi
1290 Sauverny, Switzerland
