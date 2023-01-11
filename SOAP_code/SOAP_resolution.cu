#include <iostream>
#include <fstream>
#include <ctime>// include this header
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
using namespace std;

double rndup(double n,int nb_decimal)//round up a double type at nb_decimal
{
    double t;
    t=n*pow(10,nb_decimal) - floor(n*pow(10,nb_decimal));
    if (t>=0.5)
    {
        n*=pow(10,nb_decimal);//where n is the multi-decimal double
        n = ceil(n);
        n/=pow(10,nb_decimal);
    }
    else
    {
        n*=pow(10,nb_decimal);//where n is the multi-decimal double
        n = floor(n);
        n/=pow(10,nb_decimal);
    }
    return n;
}


double Delta_lambda_cpu(double line_width, double lambda_line0)
{
    double c=299792458.;
    double line_spread=0.;
	double beta;


	beta = line_width/c;
	line_spread = -1 * lambda_line0 * (1 - sqrtf((1+beta)/(1-beta)));

    return line_spread;
}


__device__ double Delta_lambda(double line_width, double lambda_line0)
{
    double c=299792458.;
    double line_spread=0.;
	double beta;

	beta = line_width/c;
	line_spread = -1 * lambda_line0 * (1 - sqrt((1+beta)/(1-beta)));

    return line_spread;
}



__global__ void wav_sigma(double *wave, double *wave_out, double sigma){


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    wave_out[index] = Delta_lambda(sigma,wave[index]);
    __syncthreads();
}



__global__ void conv_inst(double *wavelength, double *spec, double *sigmas, double *spec_out1, int wid, int vector_size,double *quiet_sun_gpu){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double pi  =3.141592653589793238463;
    double wavelength_step_median = 0.005;

    if ((index > wid) && (index + wid < vector_size-1)){

        for (int m = index - wid; m <index + wid-1; m++){

            spec_out1[index] += wavelength_step_median*(1.0 - (quiet_sun_gpu[m]-spec[m])) * 1./(sigmas[index]*sqrt(2.*pi))*exp(-(m-index)*(m-index)*0.005*0.005/(2*sigmas[index]*sigmas[index]));

        }


    }


    __syncthreads();
}

__device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__device__ double atomicMax_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__global__ void reduceMaxIdxOptimizedShared(double* input, int size, double* maxOut, int* maxIdxOut)
{
    __shared__ double sharedMax;
    __shared__ int sharedMaxIdx;

    if (0 == threadIdx.x)
    {
        sharedMax = -60000.0;
        sharedMaxIdx = 0;
    }

    __syncthreads();

    double localMax = -60000.0;
    int localMaxIdx = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        if (input[i] != 0.0){
        double val = input[i];


        if ((localMax < val) )
        {
            localMax = val;
            localMaxIdx = i;
        }
}
    }

    atomicMax_double(&sharedMax, localMax);

    __syncthreads();

    if (sharedMax == localMax)
    {
        sharedMaxIdx = localMaxIdx;
    }

    __syncthreads();

    if (0 == threadIdx.x)
    {
        *maxOut = sharedMax;
        *maxIdxOut = sharedMaxIdx;
    }
}

__global__ void reduceMinIdxOptimizedShared(double* input, int size, double* minOut, int* minIdxOut)
{
    __shared__ double sharedMin;
    __shared__ int sharedMinIdx;

    if (0 == threadIdx.x)
    {
        sharedMin = input[0];
        sharedMinIdx = 0;
    }

    __syncthreads();

    double localMin = input[0];
    int localMinIdx = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        double val = input[i];

        if (localMin > val)
        {
            localMin = val;
            localMinIdx = i;
        }
    }

    atomicMin_double(&sharedMin, localMin);

    __syncthreads();

    if (sharedMin == localMin)
    {
        sharedMinIdx = localMinIdx;
    }

    __syncthreads();

    if (0 == threadIdx.x)
    {
        *minOut = sharedMin;
        *minIdxOut = sharedMinIdx;
    }
}

__global__ void spec_diff(double *quiet_sun_gpu, double *active_spec, double *spec_difference, int vector_size){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    spec_difference[index] = quiet_sun_gpu[index] - active_spec[index];

    __syncthreads();
}


__global__ void spec_norm(double *active_spec, double *spec_final_output, double *max_val, double *min_val){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    spec_final_output[index] = 1.0-active_spec[index] *(1-min_val[0])/max_val[0];


    __syncthreads();
}


void lower_resolution_gpu(double *wavelength, double *spectrum, double *spectrum_low_reso, int vector_size, double sigma_resolution, string phi_name, double *quiet_sun_gpu, string out_type,string final_spec_dir, int no_block,
                            int no_thread, string out_put_prefix)

    {


    double window_width=50.;

    double wavelength_step=wavelength[1]-wavelength[0];
    int nb_pixel = rndup((window_width*Delta_lambda_cpu(sigma_resolution,wavelength[0]))/wavelength_step,0);


    double *sigma_wav, *wav_gpu, *spec_gpu,  *spec_out1;
    double *spec_difference, *spec_final_output;
    cudaMalloc( (void**)&sigma_wav, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&wav_gpu, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&spec_gpu, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&spec_out1, vector_size * sizeof(double) ) ;

    cudaMalloc( (void**)&spec_difference, (vector_size) * sizeof(double) ) ;
    cudaMalloc( (void**)&spec_final_output, (vector_size) * sizeof(double) ) ;

    cudaMemcpy( wav_gpu, wavelength, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
    cudaMemcpy( spec_gpu, spectrum, vector_size * sizeof(double),  cudaMemcpyHostToDevice );



    wav_sigma<<<no_block, no_thread>>>(wav_gpu, sigma_wav, sigma_resolution);

    conv_inst<<<no_block, no_thread>>>(wav_gpu, spec_gpu, sigma_wav, spec_out1, nb_pixel, vector_size, quiet_sun_gpu);

    spec_diff<<<no_block, no_thread>>>(quiet_sun_gpu, spec_gpu, spec_difference, vector_size);

    double *max_val_dev,  *max_val,*min_val_dev,  *min_val;
    int *max_idx_dev, *max_idx, *min_idx_dev, *min_idx;

    cudaMalloc( (void**)&max_val_dev, 1 * sizeof(double) ) ;
    cudaMalloc( (void**)&max_idx_dev, 1 * sizeof(int) ) ;
    max_val = (double*)malloc( 1 * sizeof(double) );
    max_idx = (int*)malloc( 1 * sizeof(int) );


    cudaMalloc( (void**)&min_val_dev, 1 * sizeof(double) ) ;
    cudaMalloc( (void**)&min_idx_dev, 1 * sizeof(int) ) ;
    min_val = (double*)malloc( 1 * sizeof(double) );
    min_idx = (int*)malloc( 1 * sizeof(int) );

    reduceMaxIdxOptimizedShared<<<no_block, no_thread>>>(spec_out1, vector_size, max_val_dev, max_idx_dev);
    reduceMinIdxOptimizedShared<<<no_block, no_thread>>>(spec_difference, vector_size, min_val_dev, min_idx_dev);

    cudaMemcpy(max_val, max_val_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(max_idx, max_idx_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost );

    cudaMemcpy(min_val, min_val_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(min_idx, min_idx_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost );


    spec_norm<<<no_block, no_thread>>>(spec_out1, spec_final_output, max_val_dev, min_val_dev);



    cudaMemcpy(spectrum_low_reso, spec_final_output, vector_size * sizeof(double), cudaMemcpyDeviceToHost );

    string dataname_out=final_spec_dir+out_put_prefix+std::string("_lower_spec_")+out_type+std::string("_")+phi_name+std::string("_.bin");
    FILE *file = fopen(dataname_out.c_str(),"wb");
      fwrite(spectrum_low_reso,sizeof(double), no_block*no_thread, file);
      fclose(file);

    cudaFree(sigma_wav ) ;
    cudaFree(wav_gpu) ;
    cudaFree(spec_gpu) ;
    cudaFree(spec_out1) ;
    cudaFree(spec_final_output) ;
    cudaFree(spec_difference) ;

}







int main(int argc, char *argv[]){



    int start_s=clock();
    auto wcts = std::chrono::system_clock::now();


    int npsi = atoi(argv[1]);
    int vector_size = atoi(argv[2]);
    double sigma_resolution = atof(argv[3]);
    string raw_dir = argv[4];
    string final_spec_dir = argv[5];
    int no_block = atoi(argv[6]);
    int no_thread = atoi(argv[7]);
    string out_put_prefix = argv[8];
    string quiet_prefix = argv[9];

    double *quiet_sun_ptr, *quiet_sun_gpu;

    quiet_sun_ptr = (double*)malloc( vector_size * sizeof(double) );
    cudaMalloc( (void**)&quiet_sun_gpu, vector_size * sizeof(double) ) ;

    string quietsunname=quiet_prefix+std::string("_quiet_sun_spec.bin");
    FILE *file_quietsun = fopen(quietsunname.c_str(),"rb");
      fread(quiet_sun_ptr,sizeof(double), no_block*no_thread, file_quietsun);
      fclose(file_quietsun);

    cudaMemcpy( quiet_sun_gpu, quiet_sun_ptr, vector_size * sizeof(double),  cudaMemcpyHostToDevice );


    double *quiet_sun_ptr_SED, *quiet_sun_gpu_SED;

    quiet_sun_ptr_SED = (double*)malloc( vector_size * sizeof(double) );
    cudaMalloc( (void**)&quiet_sun_gpu_SED, vector_size * sizeof(double) ) ;

    string quietsunname_SED=quiet_prefix+std::string("_SED_quiet_sun_spec.bin");
    FILE *file_quietsun_SED = fopen(quietsunname_SED.c_str(),"rb");
      fread(quiet_sun_ptr_SED,sizeof(double), no_block*no_thread, file_quietsun_SED);
      fclose(file_quietsun_SED);

    cudaMemcpy( quiet_sun_gpu_SED, quiet_sun_ptr_SED, vector_size * sizeof(double),  cudaMemcpyHostToDevice );




    double *quiet_wave_ptr;

    quiet_wave_ptr = (double*)malloc( vector_size * sizeof(double) );

    string quietwavename=quiet_prefix+std::string("_quiet_sun_wave.bin");
    FILE *file_quietwave = fopen(quietwavename.c_str(),"rb");
      fread(quiet_wave_ptr,sizeof(double), no_block*no_thread, file_quietwave);
      fclose(file_quietwave);


    string type_tot = "tot";
    string type_flux = "flux";
    string type_bconv = "bconv";


    for (int ipsi=0; ipsi<npsi; ipsi++)
    {
        printf("PHASE = %d over %d\n",ipsi,npsi);
        string phi_name = to_string(ipsi);


        double *spectrum_low_reso_tot, *spectrum_low_reso_flux, *spectrum_low_reso_bconv;

        spectrum_low_reso_tot = (double*)malloc( vector_size * sizeof(double) );
        spectrum_low_reso_flux = (double*)malloc( vector_size * sizeof(double) );
        spectrum_low_reso_bconv = (double*)malloc( vector_size * sizeof(double) );


        double *f_spot_flux, *f_spot_bconv, *f_spot_tot;
        f_spot_flux = (double*)malloc( vector_size * sizeof(double) );
        f_spot_bconv = (double*)malloc( vector_size * sizeof(double) );
        f_spot_tot = (double*)malloc( vector_size * sizeof(double) );


        string dataname_flux=raw_dir+out_put_prefix+std::string("_Spec_flux_")+phi_name+std::string("_.bin");
        FILE *file_flux = fopen(dataname_flux.c_str(),"rb");
          fread(f_spot_flux,sizeof(double), no_block*no_thread, file_flux);
          fclose(file_flux);

        string dataname_bconv=raw_dir+out_put_prefix+std::string("_Spec_bconv_")+phi_name+std::string("_.bin");
        FILE *file_bconv = fopen(dataname_bconv.c_str(),"rb");
        fread(f_spot_bconv,sizeof(double), no_block*no_thread, file_bconv);
        fclose(file_bconv);


        string dataname_tot=raw_dir+out_put_prefix+std::string("_Spec_tot_")+phi_name+std::string("_.bin");
        FILE *file_tot = fopen(dataname_tot.c_str(),"rb");
          fread(f_spot_tot,sizeof(double), no_block*no_thread, file_tot);
          fclose(file_tot);

        lower_resolution_gpu(quiet_wave_ptr, f_spot_tot, spectrum_low_reso_tot, vector_size, sigma_resolution, phi_name, quiet_sun_gpu_SED, type_tot, final_spec_dir, no_block, no_thread, out_put_prefix);
        lower_resolution_gpu(quiet_wave_ptr, f_spot_flux, spectrum_low_reso_flux, vector_size, sigma_resolution, phi_name, quiet_sun_gpu_SED, type_flux, final_spec_dir, no_block, no_thread, out_put_prefix);
        lower_resolution_gpu(quiet_wave_ptr, f_spot_bconv, spectrum_low_reso_bconv, vector_size, sigma_resolution, phi_name, quiet_sun_gpu, type_bconv, final_spec_dir, no_block, no_thread, out_put_prefix);

        free(f_spot_flux);
        free(f_spot_bconv);
        free(f_spot_tot);

        free(spectrum_low_reso_tot);
        free(spectrum_low_reso_flux);
        free(spectrum_low_reso_bconv);



    }




    std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout<<"finished in "<<wctduration.count() <<" seconds [wall clock]" << std::endl;

    int stop_s=clock();
    double t = double (stop_s - start_s);
    cout << "time: " << (t / double(CLOCKS_PER_SEC)) << endl;
    return 0;
}
