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




__device__ double Delta_lambda(double line_width, double lambda_line0)//line_width in RV [m/s]
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

}



__global__ void conv_inst(double *wavelength, double *spec, double *sigmas, double *spec_out1, int wid, int vector_size, double wavelength_step_median){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double pi  =3.141592653589793238463;

    if ((index > wid) && (index + wid < vector_size-1)){

        for (int m = index - wid; m <index + wid-1; m++){

            spec_out1[index] += wavelength_step_median*(1.0 - spec[m]) * 1./(sigmas[index]*sqrt(2.*pi))*exp(-(m-index)*(m-index)*0.005*0.005/(2*sigmas[index]*sigmas[index]));

        }


    }


    __syncthreads();
}




__global__ void spec_grad_2d(double *spectrum, double *spectrum_gradient, int size){


    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (index < size-1){

        spectrum_gradient[index] = spectrum[index+1]-spectrum[index];
    }


}



__global__ void fast_shift(double *lambda, double *lambda_shift, double delta){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double c=299792458.;
    double beta = delta*1000./c;
    double fbeta = -1 * (1 - sqrt((1+beta)/(1-beta)));


    lambda_shift[index] = lambda[index]+Delta_lambda(delta*1000.,lambda[index]);


}



__global__ void fast_interp_new(double *lambda, double *lambda_shift, double *spectrum, double *spec_grad, double *spectrum_out, double delta, double limb, double step_size, int offset){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int index_offset = blockIdx.x * blockDim.x + threadIdx.x+offset;
    double lambda0 = lambda[0];
    double c=299792458.;
    double beta = delta*1000./c;
    double fbeta = -1 * (1 - sqrt((1+beta)/(1-beta)));


    if (delta > 0.0){

        //round up the lower boundary here:
        int new_index = (int) ceil(index*(1+fbeta)+fbeta*lambda0/step_size);
        spectrum_out[new_index] = (spectrum[index_offset]+spec_grad[index_offset]/(lambda_shift[index+1] - lambda_shift[index])*(lambda[new_index] - lambda_shift[index]))*limb;

    }
    else if (delta == 0.0){
        int new_index = index;
        spectrum_out[new_index] = spectrum[index_offset]*limb;


    }
    else if (delta < 0.0){

        //round up the lower boundary here:
        int new_index = (int) ceil(index*(1+fbeta)+fbeta*lambda0/step_size);
        if (new_index >= 0){

            spectrum_out[new_index] = (spectrum[index_offset]+spec_grad[index_offset]/(lambda_shift[index+1] - lambda_shift[index])*(lambda[new_index] - lambda_shift[index]))*limb;
        }


    }

}



__global__ void fast_sum(double *dev_sum_star, double *dev_spec_out){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dev_sum_star[index] = dev_sum_star[index]+dev_spec_out[index];

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
        double val = input[i];

        if ((localMax < val) && (val != 0.0))
        {
            localMax = val;
            localMaxIdx = i;
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


void itot(double period, double omega_ratio, double radius_star, double incl, double limba1, double limba2, int grid,
	  int vector_size, double step_size, double *dev_wav, double *dev_spec, double *dev_sum_star, int no_block, int no_thread, int no_spec, double *mu_boundary, string out_put_prefix,
      double LB_para0, double LB_para1, double LB_para2, double LB_para3)
{

    double delta_grid, theta_grid;
    double y, z, delta, r_cos, limb, theta_z, omega_velo, lb_value;
    double pi  =3.141592653589793238463;


    /* Conversions */
    incl = incl * pi/180. ; // [degree]       --> [radian]
    delta_grid = 2./grid;
    theta_grid = 180.0/grid;

    int spec_idx = no_spec-1;

    double *dev_grad, *dev_vrad, *dev_spec_out;
    cudaMalloc( (void**)&dev_vrad, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&dev_grad, vector_size * no_spec* sizeof(double) ) ;
    cudaMalloc( (void**)&dev_spec_out, vector_size * sizeof(double) ) ;

    dim3 dimBlock(16, 16);
    dim3 dimGrid(no_spec*4*4, 134);

    spec_grad_2d<<<dimGrid,dimBlock>>>(dev_spec, dev_grad, vector_size*no_spec); // launch a 2d kernel to calculate the derivative...

    double *delta_maps, *limb_maps,*mu_maps, *LB_maps;
    int *region_maps;
    int grid_map_size = grid*grid;
    delta_maps = (double*)malloc( grid_map_size * sizeof(double) );
    limb_maps = (double*)malloc( grid_map_size * sizeof(double) );
    mu_maps = (double*)malloc( grid_map_size * sizeof(double) );
    region_maps = (int*)malloc( grid_map_size * sizeof(int) );
    LB_maps = (double*)malloc( grid_map_size * sizeof(double) );

    int index_maps;


    double omega0 = 360.0/period;
    double omega1 = omega0*omega_ratio;

    for (int iy=0; iy<grid; iy++)
    {

        y = -1. + iy*delta_grid;

        for (int iz=0; iz<grid; iz++)
        {
            z = -1. + iz*delta_grid;
            theta_z = -90.0+ iz*theta_grid;
            omega_velo = (omega0 +omega1* pow(sin(theta_z*pi/180), 2))*pi/(180*24*3600)*radius_star;


            delta = y * omega_velo * sin(incl);

            index_maps = iy*grid+iz;

            delta_maps[index_maps] = -999.0;
            limb_maps[index_maps] =  -999.0;
            mu_maps[index_maps] = -999.0;
            LB_maps[index_maps] = -999.0;
            region_maps[index_maps] = 0;

            if ((y*y+z*z)<=1.){



                r_cos = pow(1.-(y*y+z*z),.5);
                limb =  1. - limba1*(1-r_cos) - limba2*(1-r_cos)*(1-r_cos);
                lb_value = LB_para0 + LB_para1* r_cos+ LB_para2*r_cos*r_cos + LB_para3*r_cos*r_cos*r_cos;


                for (int id_mu = 0; id_mu<no_spec; id_mu++)
                {
                    if (r_cos > mu_boundary[id_mu]){
                        spec_idx = id_mu;
                        break;

                    }
                }

                fast_shift<<<no_block,no_thread>>>(dev_wav, dev_vrad,delta);
                fast_interp_new<<<no_block,no_thread>>>(dev_wav, dev_vrad, dev_spec, dev_grad, dev_spec_out, delta, limb, step_size, spec_idx*vector_size);
                fast_sum<<<no_block,no_thread>>>(dev_sum_star, dev_spec_out);


                delta_maps[index_maps] = delta;
                limb_maps[index_maps] =  limb;
                mu_maps[index_maps] = r_cos;
                region_maps[index_maps] = spec_idx;
                LB_maps[index_maps] = lb_value;

            }


        }

    }



    string dataname_mu=out_put_prefix+std::string("mu_map.bin");
    FILE *file_mu = fopen(dataname_mu.c_str(),"wb");
        fwrite(mu_maps,sizeof(double), grid_map_size, file_mu);
        fclose(file_mu);

    string dataname_delta=out_put_prefix+std::string("delta_map.bin");
    FILE *file_delta = fopen(dataname_delta.c_str(),"wb");
        fwrite(delta_maps,sizeof(double), grid_map_size, file_delta);
        fclose(file_delta);

    string dataname_limb=out_put_prefix+std::string("limb_map.bin");
    FILE *file_limb = fopen(dataname_limb.c_str(),"wb");
        fwrite(limb_maps,sizeof(double), grid_map_size, file_limb);
        fclose(file_limb);

    string dataname_region=out_put_prefix+std::string("region_map.bin");
    FILE *file_region = fopen(dataname_region.c_str(),"wb");
        fwrite(region_maps,sizeof(int), grid_map_size, file_region);
        fclose(file_region);

    string dataname_lb_v=out_put_prefix+std::string("LB_map.bin");
    FILE *file_lb_v = fopen(dataname_lb_v.c_str(),"wb");
        fwrite(LB_maps,sizeof(double), grid_map_size, file_lb_v);
        fclose(file_lb_v);

    free(mu_maps);
    free(region_maps);
    free(delta_maps);
    free(limb_maps);
    free(LB_maps);

    cudaFree(dev_spec_out);
    cudaFree(dev_vrad);
    cudaFree(dev_grad);


}

__global__ void spec_norm(double *active_spec, double *spec_final_output, double *max_val, double *min_val){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    spec_final_output[index] = 1.0-active_spec[index] *(1-min_val[0])/max_val[0];


}


void lower_resolution_gpu(double *wavelength, double *spectrum, double *spectrum_low_reso, int vector_size, double sigma_resolution, double wavelength_step_median, double window_width, int no_block, int no_thread, string out_put_prefix){


    double wavelength_step=wavelength[1]-wavelength[0];
    int nb_pixel = rndup((window_width*Delta_lambda_cpu(sigma_resolution,wavelength[0]))/wavelength_step,0);

    double *sigma_wav, *wav_gpu, *spec_gpu,  *spec_out1, *spec_final_output;

    cudaMalloc( (void**)&sigma_wav, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&wav_gpu, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&spec_gpu, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&spec_out1, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&spec_final_output, (vector_size) * sizeof(double) ) ;

    cudaMemcpy( wav_gpu, wavelength, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
    cudaMemcpy( spec_gpu, spectrum, vector_size * sizeof(double),  cudaMemcpyHostToDevice );



    wav_sigma<<<no_block,no_thread>>>(wav_gpu, sigma_wav, sigma_resolution);

    conv_inst<<<no_block,no_thread>>>(wav_gpu, spec_gpu, sigma_wav, spec_out1, nb_pixel, vector_size, wavelength_step_median);

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

    reduceMinIdxOptimizedShared<<<no_block, no_thread>>>(spec_gpu, vector_size, min_val_dev, min_idx_dev);


    cudaMemcpy(max_val, max_val_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(max_idx, max_idx_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost );

    cudaMemcpy(min_val, min_val_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(min_idx, min_idx_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost );


    spec_norm<<<no_block, no_thread>>>(spec_out1, spec_final_output, max_val_dev, min_val_dev);


    cudaMemcpy(spectrum_low_reso, spec_final_output, vector_size * sizeof(double), cudaMemcpyDeviceToHost );

    string dataname=out_put_prefix+std::string("_quiet_sun_lower_spec.bin");
    FILE *file = fopen(dataname.c_str(),"wb");
      fwrite(spectrum_low_reso,sizeof(double), no_block*no_thread, file);
      fclose(file);

}


int main(int argc, char *argv[])
{

    int start_s=clock();

    int vector_size = atoi(argv[1]);
    double period = atof(argv[2]);
    double omega_ratio = atof(argv[3]);
    double incl = atof(argv[4]);
    double radius_star = atof(argv[5]);
    double limba1 = atof(argv[6]);
    double limba2 = atof(argv[7]);

    int grid = atoi(argv[8]);
    double sigma_resolution = atof(argv[9]);
    double step_size = atof(argv[10]);
    double wavelength_step_median = atof(argv[11]);
    double window_width= atof(argv[12]);
    int no_block = atoi(argv[13]);
    int no_thread = atoi(argv[14]);
    string solar_data_name = argv[15];
    string out_put_prefix = argv[16];
    int no_spec = atoi(argv[17]);
    string mus_bound_name = argv[18];
    string solar_wave_name = argv[19];

    double LB_para0 = atof(argv[20]);
    double LB_para1 = atof(argv[21]);
    double LB_para2 = atof(argv[22]);
    double LB_para3 = atof(argv[23]);


    double *wave_array, *spec_array;
    double *dev_wave, *dev_spec;
    double *dev_sum_star, *out_array;

    wave_array = (double*)malloc( vector_size * sizeof(double) );
    spec_array = (double*)malloc( vector_size*no_spec * sizeof(double) );
    out_array = (double*)malloc( vector_size * sizeof(double) );



    FILE *file_solar_cube = fopen(solar_data_name.c_str(),"rb");
      fread(spec_array,sizeof(double), vector_size*no_spec, file_solar_cube);
      fclose(file_solar_cube);

    FILE *file_solar_wave = fopen(solar_wave_name.c_str(),"rb");
        fread(wave_array,sizeof(double), vector_size, file_solar_wave);
        fclose(file_solar_wave);


    cudaMalloc( (void**)&dev_wave, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&dev_spec, vector_size*no_spec * sizeof(double) ) ;


    cudaMemcpy( dev_wave, wave_array, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
    cudaMemcpy( dev_spec, spec_array, vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );


    double *mu_boundary;


    mu_boundary = (double*)malloc( (no_spec-1) * sizeof(double) );

    FILE *file_mu_bound_array = fopen(mus_bound_name.c_str(),"rb");
        fread(mu_boundary,sizeof(double), no_spec-1, file_mu_bound_array);
        fclose(file_mu_bound_array);

    cudaMalloc( (void**)&dev_sum_star, vector_size * sizeof(double) ) ;

    auto wcts = std::chrono::system_clock::now();

    itot(period, omega_ratio, radius_star, incl, limba1, limba2, grid, vector_size, step_size, dev_wave, dev_spec, dev_sum_star, no_block, no_thread, no_spec, mu_boundary, out_put_prefix, LB_para0, LB_para1, LB_para2, LB_para3);


    cudaMemcpy( out_array, dev_sum_star, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );

    double *spectrum_low_reso;
    spectrum_low_reso = (double*)malloc( vector_size * sizeof(double) );
    lower_resolution_gpu(wave_array, out_array, spectrum_low_reso, vector_size, sigma_resolution,wavelength_step_median, window_width, no_block, no_thread, out_put_prefix);



    string dataname=out_put_prefix+std::string("_quiet_sun_spec.bin");
    FILE *file = fopen(dataname.c_str(),"wb");
      fwrite(out_array,sizeof(double), no_block*no_thread, file);
      fclose(file);


    string dataname_wav=out_put_prefix+std::string("_quiet_sun_wave.bin");
    FILE *file_wav = fopen(dataname_wav.c_str(),"wb");
    fwrite(wave_array,sizeof(double), no_block*no_thread, file_wav);
    fclose(file_wav);


    std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout<<"finished in "<<wctduration.count() <<"seconds [wall clock]" << std::endl;

    int stop_s=clock();
    double t = double (stop_s - start_s);
    cout << "time: " << (t / double(CLOCKS_PER_SEC)) << endl;
    return 0;
}
