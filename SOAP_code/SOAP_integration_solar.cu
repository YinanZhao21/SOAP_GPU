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

const double pi  =3.141592653589793238463;


__global__ void spot_phase_gpu(double *xyz, double inclination, int nrho, double phase,double *xyz2)
{

    int index_dem0 = blockIdx.x * blockDim.x + 0;
    int index_dem1 = blockIdx.x * blockDim.x + 1;
    int index_dem2 = blockIdx.x * blockDim.x + 2;

    double psi = -phase*(2*pi); //phase entre
    inclination = inclination*pi/180.;

    double axe[3]  = {cos(inclination),0,sin(inclination)};
    double R[3][3] = {{(1-cos(psi))*axe[0]*axe[0] + cos(psi),
                     (1-cos(psi))*axe[0]*axe[1] + sin(psi)*axe[2],
                     (1-cos(psi))*axe[0]*axe[2] - sin(psi)*axe[1]},
                    {(1-cos(psi))*axe[1]*axe[0] - sin(psi)*axe[2],
                     (1-cos(psi))*axe[1]*axe[1] + cos(psi),
                     (1-cos(psi))*axe[1]*axe[2] + sin(psi)*axe[0]},
                    {(1-cos(psi))*axe[2]*axe[0] + sin(psi)*axe[1],
                     (1-cos(psi))*axe[2]*axe[1] - sin(psi)*axe[0],
                     (1-cos(psi))*axe[2]*axe[2] + cos(psi)}};

     xyz2[index_dem0] = R[0][0]*xyz[index_dem0] + R[0][1]*xyz[index_dem1] + R[0][2]*xyz[index_dem2];
     xyz2[index_dem1] = R[1][0]*xyz[index_dem0] + R[1][1]*xyz[index_dem1] + R[1][2]*xyz[index_dem2];
     xyz2[index_dem2] = R[2][0]*xyz[index_dem0] + R[2][1]*xyz[index_dem1] + R[2][2]*xyz[index_dem2];

    __syncthreads();
}



__global__ void spot_area(double *xlylzl, int nrho, int grid, int *boundaries, int *counton, int *countoff, double *maxminy_array, double *maxminz_array){

    int index_block =  blockIdx.x;
    int index_dem0 = blockIdx.x * blockDim.x + 0;
    int index_dem1 = blockIdx.x * blockDim.x + 1;
    int index_dem2 = blockIdx.x * blockDim.x + 2;

    counton[index_block] = 0;
    countoff[index_block] = 0;

    __syncthreads();

    int visible=0;
    double grid_step = 2./grid;
    double miny=1, minz=1, maxy=-1, maxz=-1;

    if (xlylzl[index_dem0] >= 0){

        counton[index_block] = 1;
        maxminy_array[index_block] = xlylzl[index_dem1];
        maxminz_array[index_block] = xlylzl[index_dem2];
    }
    else {
        countoff[index_block] = 1;
    }

    __syncthreads();

    int sum_on=0, sum_off=0;

    for (int jj=0; jj< nrho;jj++){
        sum_on = sum_on+counton[jj];
        sum_off = sum_off+countoff[jj];
        if (counton[jj] > 0){

            if (maxminy_array[jj]<miny) miny = maxminy_array[jj];
            if (maxminz_array[jj]<minz) minz = maxminz_array[jj];
            if (maxminy_array[jj]>maxy) maxy = maxminy_array[jj];
            if (maxminz_array[jj]>maxz) maxz = maxminz_array[jj];

        }

    }



    if (( sum_on >0)&&( sum_off >0)) {
        if (miny*maxy<0) {
            if (minz<0) minz=-1;
            else maxz=1;
        }

        if (minz*maxz<0) {
            if (miny<0) miny=-1;
            else maxy=1;
        }
    }

    if (sum_on==0) visible = 0;
    else visible = 1;

    boundaries[0] =(int) floor((1.+miny)/grid_step);
    boundaries[1] =(int) floor((1.+minz)/grid_step);
    boundaries[2] =(int) ceil((1.+maxy)/grid_step);
    boundaries[3] =(int) ceil((1.+maxz)/grid_step);

    boundaries[4] = visible;

    __syncthreads();
}




__global__ void spot_init_gpu(double s, double longitude, double latitude, double inclination,
	       int nrho, double *xyz, double *xyz2, double *rho)
{


    int index_block =  blockIdx.x;
    int index_dem0 = blockIdx.x * blockDim.x + 0;
    int index_dem1 = blockIdx.x * blockDim.x + 1;
    int index_dem2 = blockIdx.x * blockDim.x + 2;

    longitude   *= pi/180.;
    latitude    *= pi/180.;
    inclination *= pi/180.;


    double rho_step = 2.*pi/(nrho-1);

    rho[index_block] = -pi+ index_block*rho_step;
    xyz2[index_dem0] = pow(1-s*s,.5);
    xyz2[index_dem1] = s*cos(rho[index_block]);
    xyz2[index_dem2] = s*sin(rho[index_block]);

    double b  = -latitude;
    double g = longitude;
    double b2 = pi/2.-inclination;


    double R[3][3] = {{cos(b2)*cos(g)*cos(b)-sin(b)*sin(b2), -sin(g)*cos(b2), cos(b2)*cos(g)*sin(b)+sin(b2)*cos(b)},
                {sin(g)*cos(b),                          cos(g),          sin(g)*sin(b)},
                {-sin(b2)*cos(g)*cos(b)-cos(b2)*sin(b), sin(b2)*sin(g), -sin(b2)*cos(g)*sin(b)+cos(b2)*cos(b)}};

    xyz[index_dem0] = R[0][0]*xyz2[index_dem0] + R[0][1]*xyz2[index_dem1] + R[0][2]*xyz2[index_dem2];
    xyz[index_dem1] = R[1][0]*xyz2[index_dem0] + R[1][1]*xyz2[index_dem1] + R[1][2]*xyz2[index_dem2];
    xyz[index_dem2] = R[2][0]*xyz2[index_dem0] + R[2][1]*xyz2[index_dem1] + R[2][2]*xyz2[index_dem2];
    __syncthreads();

}


void spot_inverse_rotation(double *xyz, double longitude, double latitude,
			   double inclination, double phase, double *xiyizi)
{

  double g2 = --phase*(2*pi);
  double i = inclination * pi/180.;

  double b  =  latitude  * pi/180.;
  double g  = -longitude * pi/180.;
  double b2 = -(pi/2.-i);

  double R[3][3]  = {{ (1-cos(g2))*cos(i)*cos(i) + cos(g2),
  	                   sin(g2)*sin(i),
  	                   (1-cos(g2))*cos(i)*sin(i)},
                     {-sin(g2)*sin(i),
                       cos(g2),
                       sin(g2)*cos(i)},
                     { (1-cos(g2))*sin(i)*cos(i),
                      -sin(g2)*cos(i),
                       (1-cos(g2))*sin(i)*sin(i) + cos(g2)}};
  double R2[3][3] =  {{cos(b)*cos(g)*cos(b2)-sin(b2)*sin(b), -sin(g)*cos(b), cos(b)*cos(g)*sin(b2)+sin(b)*cos(b2)},
                    {sin(g)*cos(b2),                          cos(g),          sin(g)*sin(b2)},
                    {-sin(b)*cos(g)*cos(b2)-cos(b)*sin(b2), sin(b)*sin(g), -sin(b)*cos(g)*sin(b2)+cos(b)*cos(b2)}};

  double R3[3][3] = {{R2[0][0]*R[0][0]+R2[0][1]*R[1][0]+R2[0][2]*R[2][0],
  	                  R2[0][0]*R[0][1]+R2[0][1]*R[1][1]+R2[0][2]*R[2][1],
  	                  R2[0][0]*R[0][2]+R2[0][1]*R[1][2]+R2[0][2]*R[2][2]},
  	                 {R2[1][0]*R[0][0]+R2[1][1]*R[1][0]+R2[1][2]*R[2][0],
  	                  R2[1][0]*R[0][1]+R2[1][1]*R[1][1]+R2[1][2]*R[2][1],
  	                  R2[1][0]*R[0][2]+R2[1][1]*R[1][2]+R2[1][2]*R[2][2]},
  	                 {R2[2][0]*R[0][0]+R2[2][1]*R[1][0]+R2[2][2]*R[2][0],
  	                  R2[2][0]*R[0][1]+R2[2][1]*R[1][1]+R2[2][2]*R[2][1],
  	                  R2[2][0]*R[0][2]+R2[2][1]*R[1][2]+R2[2][2]*R[2][2]}};

  xiyizi[0] = R3[0][0]*xyz[0] + R3[0][1]*xyz[1] + R3[0][2]*xyz[2];
  xiyizi[1] = R3[1][0]*xyz[0] + R3[1][1]*xyz[1] + R3[1][2]*xyz[2];
  xiyizi[2] = R3[2][0]*xyz[0] + R3[2][1]*xyz[1] + R3[2][2]*xyz[2];

}


__device__ double loi_Planck(double lambda0, double Temp)
{
    double c   = 299792458.;     // vitesse de la lumiere en m/s
    double h   = 6.62606896e-34; //cte de Planck
    double k_b = 1.380e-23;      //cte de Boltzmann
    double scale = 1e-10;
    return 2*h*pow(c,2)*1./pow(lambda0*scale,5)*1./(exp((h*c)/(lambda0*k_b*Temp*scale))-1);
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

    __syncthreads();
}



__global__ void fast_flux(double *dev_spec, double *dev_spec_SED, double *dev_spec_out, double *dev_intensity){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dev_spec_out[index] = dev_spec_out[index]+dev_spec_SED[index] -dev_intensity[index]*dev_spec[index];


}

__global__ void fast_bconv(double *dev_spec,double *dev_spot, double *dev_spec_out){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dev_spec_out[index] = dev_spec_out[index]+dev_spec[index]-dev_spot[index];


}

__global__ void fast_tot(double *dev_spec, double *dev_spot, double *dev_spec_out, double *dev_intensity){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dev_spec_out[index] = dev_spec_out[index]+dev_spec[index]-dev_spot[index]*dev_intensity[index] ;


}

__global__ void fast_intensity(double *dev_wavelength, double temp_cell, double temp_star, double *dev_intensity, double *dev_SED){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dev_intensity[index] = loi_Planck(dev_wavelength[index], temp_cell)/loi_Planck(dev_wavelength[index], temp_star) *dev_SED[index];

}

__global__ void fast_intensity_fac(double *dev_wavelength, double temp_cell, double temp_star, double *dev_intensity, double *dev_SED, double lb_paras){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dev_intensity[index] = (1+lb_paras) *dev_SED[index];

}


void spot_scan(double incl, int grid,
          double s, double longitude, double phase, double latitude,
          int iminy, int iminz, int imaxy,
          int imaxz,double magn_feature_type, int T_star, int T_diff_spot, double *temp_maps, double *spot_bi_maps, int T_diff_faculae)
{

    double y, z;
    double delta_grid=2./grid, r_cos;
    double *xayaza; // actual coordinates
    double *xiyizi; // coordinates transformed back to the initial configuration

    int T_spot,T_plage;


    xayaza             = (double *)malloc(sizeof(double)*3);
    xiyizi             = (double *)malloc(sizeof(double)*3);


    int index_maps;

    for (int iy=iminy; iy<imaxy; iy++)
    {
        y = -1.+iy*delta_grid;

        xayaza[1] = y;

        // z-scan
        for (int iz=iminz; iz<imaxz; iz++)
        {
            z = -1.+iz*delta_grid;

            if (z*z+y*y<1.)
            {
                xayaza[0] = pow(1.-(y*y+z*z),.5);
                xayaza[2] = z;

                spot_inverse_rotation(xayaza,longitude,latitude,incl, phase,xiyizi);

                if (xiyizi[0]*xiyizi[0]>=1.-s*s)
                {
                    index_maps = iy*grid+iz;
                    r_cos = pow(1.-(y*y+z*z),.5);
                    if (magn_feature_type==0.0)
                    {
                        T_spot = T_star-T_diff_spot;
                        temp_maps[index_maps] = T_spot;
                        spot_bi_maps[index_maps] = 0.0;
                    }
                    else
                    {
                        T_plage = T_star+T_diff_faculae+0.9-407.7*r_cos+190.9*pow(r_cos,2);

                        temp_maps[index_maps] = T_plage;

                        spot_bi_maps[index_maps] = 1.0;
                    }


                }
            }
        }
    }


}

void spot_integrate(double *temp_maps, double *delta_maps, double *limb_maps,double *spot_bi_maps, double *lb_val_maps, double *dev_wav, double *dev_vrad, double *dev_spec, double *dev_grad, double *dev_spec_out, double *dev_spec_spot, double *dev_grad_spot, double *dev_spec_faculae, double *dev_grad_faculae,
          double *dev_spec_out_spot, double *dev_f_spot_flux,  double *dev_f_spot_bconv, double * dev_f_spot_tot, int no_block, int no_thread, int disk_size, double *dev_intensity, int temp_star, int temp_faculae, double step_size, int *specid_input_maps,int vector_size,
          double *dev_grad_SED, double *spectra_SED_dev, double *spot_SED_dev, double *faculae_SED_dev, double *dev_spec_SED_out)

{



    int T_norm;

    for (int maps_id=0; maps_id < disk_size; maps_id++)
    {
        if (spot_bi_maps[maps_id] == 0.0)
        {

            T_norm = temp_maps[maps_id];
            fast_shift<<<no_block,no_thread>>>(dev_wav, dev_vrad, delta_maps[maps_id]);
            fast_interp_new<<<no_block,no_thread>>>(dev_wav, dev_vrad, dev_spec, dev_grad, dev_spec_out, delta_maps[maps_id], limb_maps[maps_id],step_size, specid_input_maps[maps_id]*vector_size);
            fast_interp_new<<<no_block,no_thread>>>(dev_wav, dev_vrad, spectra_SED_dev, dev_grad_SED, dev_spec_SED_out, delta_maps[maps_id], limb_maps[maps_id],step_size, specid_input_maps[maps_id]*vector_size);



            fast_intensity<<<no_block,no_thread>>>(dev_wav, temp_maps[maps_id], T_norm, dev_intensity, spot_SED_dev);
            fast_interp_new<<<no_block,no_thread>>>(dev_wav, dev_vrad, dev_spec_spot, dev_grad_spot, dev_spec_out_spot, delta_maps[maps_id], limb_maps[maps_id],step_size, specid_input_maps[maps_id]*vector_size);

            fast_flux<<<no_block,no_thread>>>(dev_spec_out, dev_spec_SED_out, dev_f_spot_flux, dev_intensity);
            fast_bconv<<<no_block,no_thread>>>(dev_spec_out, dev_spec_out_spot, dev_f_spot_bconv);
            fast_tot<<<no_block,no_thread>>>(dev_spec_SED_out, dev_spec_out_spot,  dev_f_spot_tot, dev_intensity);


        }

        if (spot_bi_maps[maps_id] == 1.0)
        {

                T_norm = temp_star+temp_faculae+0.9-216.8;

                fast_shift<<<no_block,no_thread>>>(dev_wav, dev_vrad, delta_maps[maps_id]);
                fast_interp_new<<<no_block,no_thread>>>(dev_wav, dev_vrad, dev_spec, dev_grad, dev_spec_out, delta_maps[maps_id], limb_maps[maps_id],step_size, specid_input_maps[maps_id]*vector_size);
                fast_interp_new<<<no_block,no_thread>>>(dev_wav, dev_vrad, spectra_SED_dev, dev_grad_SED, dev_spec_SED_out, delta_maps[maps_id], limb_maps[maps_id],step_size, specid_input_maps[maps_id]*vector_size);


                fast_intensity<<<no_block,no_thread>>>(dev_wav, temp_maps[maps_id], T_norm, dev_intensity, faculae_SED_dev);
                fast_interp_new<<<no_block,no_thread>>>(dev_wav, dev_vrad, dev_spec_faculae, dev_grad_faculae, dev_spec_out_spot, delta_maps[maps_id], limb_maps[maps_id],step_size, specid_input_maps[maps_id]*vector_size);

                fast_flux<<<no_block,no_thread>>>(dev_spec_out, dev_spec_SED_out, dev_f_spot_flux, dev_intensity);
                fast_bconv<<<no_block,no_thread>>>(dev_spec_out, dev_spec_out_spot, dev_f_spot_bconv);
                fast_tot<<<no_block,no_thread>>>(dev_spec_SED_out, dev_spec_out_spot,  dev_f_spot_tot, dev_intensity);


            }


        }

}
__global__ void spec_grad_2d(double *spectrum, double *spectrum_gradient, int size){




    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (index < size-1){

        spectrum_gradient[index] =spectrum[index+1]-spectrum[index];
    }

}


int main(int argc, char *argv[]){

    int start_s=clock();
    auto wcts = std::chrono::system_clock::now();

    int grid = atoi(argv[1]);
    int nrho = atoi(argv[2]);
    int npsi = atoi(argv[3]);
    int T_star = atoi(argv[4]);
    int T_diff_spot = atoi(argv[5]);
    double inclination = atof(argv[6]);
    int T_diff_faculae = atoi(argv[7]);

    string phase_array_name = argv[8];
    string faculae_data_name = argv[9];

    int vector_size = atoi(argv[10]);
    string out_dir = argv[11];
    string map_dir = argv[12];

    int no_block = atoi(argv[13]);
    int no_thread = atoi(argv[14]);
    string solar_data_name = argv[15];
    string spot_data_name = argv[16];
    string out_put_prefix = argv[17];
    double step_size = atof(argv[18]);

    int no_spec = atoi(argv[19]);
    string solar_wave_name = argv[20];

    string solar_SED_spectra_data_name = argv[21];
    string spot_SED_data_name = argv[22];
    string faculae_SED_data_name = argv[23];
    string map_SOAP_dir = argv[24];

    double *spectra_SED_array, *spot_SED_array, *faculae_SED_array;

    double *spectra_SED_dev, *spot_SED_dev, *faculae_SED_dev;


    spectra_SED_array = (double*)malloc(  vector_size*no_spec * sizeof(double) );
    spot_SED_array = (double*)malloc(  vector_size*no_spec * sizeof(double) );
    faculae_SED_array = (double*)malloc(  vector_size*no_spec * sizeof(double) );

    FILE *file_solar_spectra_SED_cube = fopen(solar_SED_spectra_data_name.c_str(),"rb");
      fread(spectra_SED_array,sizeof(double), vector_size*no_spec, file_solar_spectra_SED_cube);
      fclose(file_solar_spectra_SED_cube);

    FILE *file_spot_SED_cube = fopen(spot_SED_data_name.c_str(),"rb");
    fread(spot_SED_array,sizeof(double), vector_size*no_spec, file_spot_SED_cube);
    fclose(file_spot_SED_cube);

    FILE *file_faculae_SED_cube = fopen(faculae_SED_data_name.c_str(),"rb");
      fread(faculae_SED_array,sizeof(double), vector_size*no_spec, file_faculae_SED_cube);
      fclose(file_faculae_SED_cube);

    cudaMalloc( (void**)&spectra_SED_dev,  vector_size*no_spec * sizeof(double) ) ;
    cudaMalloc( (void**)&spot_SED_dev,  vector_size*no_spec * sizeof(double) ) ;
    cudaMalloc( (void**)&faculae_SED_dev,  vector_size*no_spec * sizeof(double) ) ;

    cudaMemcpy( spectra_SED_dev, spectra_SED_array, vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
    cudaMemcpy( spot_SED_dev, spot_SED_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
    cudaMemcpy( faculae_SED_dev, faculae_SED_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );



    double *wave_array, *spec_array;

    double *wave_array_spot, *spec_array_spot, *spec_array_faculae;

    double *dev_wave, *dev_spec;

    double *dev_wave_spot, *dev_spec_spot, *dev_spec_faculae;

    wave_array = (double*)malloc( vector_size * sizeof(double) );
    spec_array = (double*)malloc(  vector_size*no_spec * sizeof(double) );


    wave_array_spot = (double*)malloc( vector_size * sizeof(double) );
    spec_array_spot = (double*)malloc(  vector_size*no_spec * sizeof(double) );
    spec_array_faculae = (double*)malloc(  vector_size*no_spec * sizeof(double) );

    FILE *file_solar_cube = fopen(solar_data_name.c_str(),"rb");
      fread(spec_array,sizeof(double), vector_size*no_spec, file_solar_cube);
      fclose(file_solar_cube);

    FILE *file_solar_wave = fopen(solar_wave_name.c_str(),"rb");
        fread(wave_array,sizeof(double), vector_size, file_solar_wave);
        fclose(file_solar_wave);


    FILE *file_spot_cube = fopen(spot_data_name.c_str(),"rb");
      fread(spec_array_spot,sizeof(double), vector_size*no_spec, file_spot_cube);
      fclose(file_spot_cube);

    FILE *file_faculae_cube = fopen(faculae_data_name.c_str(),"rb");
        fread(spec_array_faculae,sizeof(double), vector_size*no_spec, file_faculae_cube);
        fclose(file_faculae_cube);

    FILE *file_spot_wave = fopen(solar_wave_name.c_str(),"rb");
        fread(wave_array_spot,sizeof(double), vector_size, file_spot_wave);
        fclose(file_spot_wave);


    cudaMalloc( (void**)&dev_wave, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&dev_spec,  vector_size*no_spec* sizeof(double) ) ;

    cudaMemcpy( dev_wave, wave_array, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
    cudaMemcpy( dev_spec, spec_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );

    cudaMalloc( (void**)&dev_wave_spot, vector_size * sizeof(double) ) ;
    cudaMalloc( (void**)&dev_spec_spot,  vector_size*no_spec * sizeof(double) ) ;
    cudaMalloc( (void**)&dev_spec_faculae,  vector_size*no_spec * sizeof(double) ) ;

    cudaMemcpy( dev_wave_spot, wave_array_spot, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
    cudaMemcpy( dev_spec_spot, spec_array_spot,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
    cudaMemcpy( dev_spec_faculae, spec_array_faculae,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );

    dim3 dimBlock(16, 16);
    dim3 dimGrid(no_spec*4*4, 134);

    double *dev_grad, *dev_grad_spot, *dev_grad_faculae;

    cudaMalloc( (void**)&dev_grad, vector_size*no_spec  * sizeof(double) ) ;
    cudaMalloc( (void**)&dev_grad_spot, vector_size*no_spec * sizeof(double) );
    cudaMalloc( (void**)&dev_grad_faculae, vector_size*no_spec * sizeof(double) );

    spec_grad_2d<<<dimGrid,dimBlock>>>(dev_spec, dev_grad, vector_size*no_spec);
    spec_grad_2d<<<dimGrid,dimBlock>>>(dev_spec_spot, dev_grad_spot, vector_size*no_spec);
    spec_grad_2d<<<dimGrid,dimBlock>>>(dev_spec_faculae, dev_grad_faculae, vector_size*no_spec);



    double *dev_grad_SED;

    cudaMalloc( (void**)&dev_grad_SED, vector_size*no_spec  * sizeof(double) ) ;
    spec_grad_2d<<<dimGrid,dimBlock>>>(spectra_SED_dev, dev_grad_SED, vector_size*no_spec);


    int grid_map_size = grid*grid;

    int *specid_input_maps;
    specid_input_maps = (int*)malloc( grid_map_size * sizeof(int) );
    string dataname_region=map_SOAP_dir+out_put_prefix+std::string("region_map.bin");
    FILE *file_specid_map_input = fopen(dataname_region.c_str(),"rb");
        fread(specid_input_maps,sizeof(int), grid_map_size, file_specid_map_input);
        fclose(file_specid_map_input);


    double *delta_maps;
    delta_maps = (double*)malloc( grid_map_size * sizeof(double) );
    string dataname_delta=map_SOAP_dir+out_put_prefix+std::string("delta_map.bin");
    FILE *file_delta_map_input = fopen(dataname_delta.c_str(),"rb");
        fread(delta_maps,sizeof(double), grid_map_size, file_delta_map_input);
        fclose(file_delta_map_input);

    double *limb_maps;
    limb_maps = (double*)malloc( grid_map_size * sizeof(double) );
    string dataname_limb=map_SOAP_dir+out_put_prefix+std::string("limb_map.bin");

    FILE *file_limb_map_input = fopen(dataname_limb.c_str(),"rb");
        fread(limb_maps,sizeof(double), grid_map_size, file_limb_map_input);
        fclose(file_limb_map_input);


    double *lb_val_maps;
    lb_val_maps = (double*)malloc( grid_map_size * sizeof(double) );
    string dataname_lb_val=map_SOAP_dir+out_put_prefix+std::string("LB_map.bin");

    FILE *file_lb_val_map_input = fopen(dataname_lb_val.c_str(),"rb");
        fread(lb_val_maps,sizeof(double), grid_map_size, file_lb_val_map_input);
        fclose(file_lb_val_map_input);

    double *psi;
    psi = (double*)malloc( npsi * sizeof(double) );

    ifstream input_phase(phase_array_name);
    for(int i = 0; i < npsi; i++)
    {
      input_phase >> psi[i];
    }


    double *maps_id;
    maps_id = (double*)malloc( npsi * sizeof(double) );

    double *maps_grid;
    maps_grid = (double*)malloc( npsi * sizeof(double) );

    string inputmap_names = map_dir+std::string("final_info.txt");
    ifstream inputmaps(inputmap_names);
    for(int i = 0; i < npsi; i++)
    {
      inputmaps >> maps_id[i] >> maps_grid[i];
    }


    for (int ipsi=0; ipsi<npsi; ipsi++)
    {


        int id_name = (int)maps_id[ipsi];
        int map_size = (int)maps_grid[ipsi];

        string phi_name = to_string(id_name);
        string file_names = map_dir+std::string("maps")+phi_name+std::string("_for_all.txt");

        double *lat_maps, *long_maps, *size_maps, *type_maps;
        lat_maps = (double*)malloc( map_size * sizeof(double) );
        long_maps = (double*)malloc( map_size * sizeof(double) );
        size_maps = (double*)malloc( map_size * sizeof(double) );
        type_maps = (double*)malloc( map_size * sizeof(double) );

        ifstream input_files(file_names);

        for(int mmm = 0; mmm < map_size; mmm++)
        {
          input_files >> lat_maps[mmm] >> long_maps[mmm] >> size_maps[mmm] >> type_maps[mmm];
        }



        double *temp_maps, *spot_bi_maps;

        temp_maps = (double*)malloc( grid_map_size * sizeof(double) );
        spot_bi_maps = (double*)malloc( grid_map_size * sizeof(double) );

        for(int kk = 0; kk < grid_map_size; kk++)
        {
          temp_maps[kk] = -9999.0;
        }


        for(int kk = 0; kk < grid_map_size; kk++)
        {
          spot_bi_maps[kk] = -9999.0;
        }


        double *dev_vrad, *dev_spec_out, *dev_spec_out_spot;
        cudaMalloc( (void**)&dev_vrad, vector_size * sizeof(double) ) ;
        cudaMalloc( (void**)&dev_spec_out, vector_size * sizeof(double) ) ;
        cudaMalloc( (void**)&dev_spec_out_spot, vector_size * sizeof(double) ) ;

        double *dev_spec_SED_out;
        cudaMalloc( (void**)&dev_spec_SED_out, vector_size * sizeof(double) ) ;

        double *dev_f_spot_flux, *dev_f_spot_bconv, *dev_f_spot_tot;
        cudaMalloc( (void**)&dev_f_spot_flux, vector_size * sizeof(double) ) ;
        cudaMalloc( (void**)&dev_f_spot_bconv, vector_size * sizeof(double) ) ;
        cudaMalloc( (void**)&dev_f_spot_tot, vector_size * sizeof(double) ) ;

        double *f_spot_flux, *f_spot_bconv, *f_spot_tot;
        f_spot_flux = (double*)malloc( vector_size * sizeof(double) );
        f_spot_bconv = (double*)malloc( vector_size * sizeof(double) );
        f_spot_tot = (double*)malloc( vector_size * sizeof(double) );

        double *dev_intensity;
        cudaMalloc( (void**)&dev_intensity, vector_size * sizeof(double) ) ;

        for(int s_num = 0; s_num < map_size; s_num++)
        {


            double *xyz, *xyz2, *rho;
            cudaMalloc( (void**)&xyz, 3*nrho* sizeof(double) );
            cudaMalloc( (void**)&xyz2, 3*nrho* sizeof(double) );
            cudaMalloc( (void**)&rho, nrho* sizeof(double) );

            double *xyz22;
            cudaMalloc( (void**)&xyz22, 3*nrho* sizeof(double) );


            int *counton, *countoff;
            double *maxminy_array, *maxminz_array;
            int *boundaries, *boundaries_cpu;

            cudaMalloc( (void**)&counton, nrho* sizeof(int) );
            cudaMalloc( (void**)&countoff, nrho* sizeof(int) );

            cudaMalloc( (void**)&maxminy_array, nrho* sizeof(double) );
            cudaMalloc( (void**)&maxminz_array, nrho* sizeof(double) );

            cudaMalloc( (void**)&boundaries, 5*sizeof(int) );
            boundaries_cpu = (int*)malloc( 5*sizeof(int) );


            spot_init_gpu<<<nrho, 3>>>(size_maps[s_num], long_maps[s_num], lat_maps[s_num], inclination, nrho, xyz, xyz2, rho);
            spot_phase_gpu<<<nrho, 3>>>(xyz, inclination, nrho, psi[ipsi], xyz22);
            spot_area<<<nrho, 3>>>(xyz22, nrho, grid, boundaries, counton, countoff,maxminy_array,maxminz_array);


            cudaMemcpy(boundaries_cpu, boundaries, 5*sizeof(int), cudaMemcpyDeviceToHost );



            if (boundaries_cpu[4]==1){

                spot_scan(inclination, grid,size_maps[s_num], long_maps[s_num], psi[ipsi],
                      lat_maps[s_num], boundaries_cpu[0], boundaries_cpu[1], boundaries_cpu[2], boundaries_cpu[3],
                      type_maps[s_num],T_star,T_diff_spot,temp_maps, spot_bi_maps, T_diff_faculae);


            }


            cudaFree(counton );
            cudaFree(countoff );

            cudaFree( maxminy_array );
            cudaFree( maxminz_array );

            cudaFree(boundaries);
            free(boundaries_cpu);
            //
            cudaFree(rho);
            cudaFree(xyz2);
            cudaFree(xyz);
            cudaFree(xyz22);


        }




        spot_integrate(temp_maps, delta_maps, limb_maps, spot_bi_maps, lb_val_maps, dev_wave, dev_vrad, dev_spec, dev_grad,dev_spec_out, dev_spec_spot, dev_grad_spot,dev_spec_faculae,dev_grad_faculae, dev_spec_out_spot,
                       dev_f_spot_flux, dev_f_spot_bconv,dev_f_spot_tot, no_block, no_thread, grid_map_size, dev_intensity, T_star,T_diff_faculae,step_size, specid_input_maps, vector_size, dev_grad_SED, spectra_SED_dev,
                       spot_SED_dev, faculae_SED_dev, dev_spec_SED_out);



        cudaMemcpy( f_spot_flux, dev_f_spot_flux, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
        cudaMemcpy( f_spot_bconv, dev_f_spot_bconv, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
        cudaMemcpy( f_spot_tot, dev_f_spot_tot, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );



        string dataname_flux=out_dir+out_put_prefix+std::string("_Spec_flux_")+phi_name+std::string("_.bin");
        FILE *file_flux = fopen(dataname_flux.c_str(),"wb");
          fwrite(f_spot_flux,sizeof(double), no_block*no_thread, file_flux);
          fclose(file_flux);

        string dataname_bconv=out_dir+out_put_prefix+std::string("_Spec_bconv_")+phi_name+std::string("_.bin");
        FILE *file_bconv = fopen(dataname_bconv.c_str(),"wb");
        fwrite(f_spot_bconv,sizeof(double), no_block*no_thread, file_bconv);
        fclose(file_bconv);


        string dataname_tot=out_dir+out_put_prefix+std::string("_Spec_tot_")+phi_name+std::string("_.bin");
        FILE *file_tot = fopen(dataname_tot.c_str(),"wb");
          fwrite(f_spot_tot,sizeof(double), no_block*no_thread, file_tot);
          fclose(file_tot);
        std::cout<< dataname_tot  << std::endl;

        cudaFree(dev_f_spot_flux);
        cudaFree(dev_f_spot_bconv);
        cudaFree(dev_f_spot_tot);
        free(f_spot_flux);
        free(f_spot_bconv);
        free(f_spot_tot);


        cudaFree(dev_vrad) ;
        cudaFree(dev_spec_out) ;
        cudaFree(dev_spec_out_spot) ;
        cudaFree(dev_spec_SED_out) ;

        string dataname_temp=out_dir+out_put_prefix+std::string("_temp_map_")+phi_name+std::string("_.bin");
        FILE *file_temp = fopen(dataname_temp.c_str(),"wb");
            fwrite(temp_maps,sizeof(double), grid_map_size, file_temp);
            fclose(file_temp);

        string dataname_binary=out_dir+out_put_prefix+std::string("_spot_binary_map_")+phi_name+std::string("_.bin");
        FILE *file_binary = fopen(dataname_binary.c_str(),"wb");
            fwrite(spot_bi_maps,sizeof(double), grid_map_size, file_binary);
            fclose(file_binary);

        free(temp_maps);
        free(spot_bi_maps);

    }

    cudaFree(dev_grad_spot) ;
    cudaFree(dev_grad_faculae) ;
    cudaFree(dev_grad_SED) ;
    cudaFree(dev_grad) ;

    std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout<<"finished in "<<wctduration.count() <<" seconds [wall clock]" << std::endl;

    int stop_s=clock();
    double t = double (stop_s - start_s);
    cout << "time: " << (t / double(CLOCKS_PER_SEC)) << endl;
    return 0;

}
