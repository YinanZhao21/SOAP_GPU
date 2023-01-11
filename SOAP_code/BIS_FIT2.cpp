#include <iostream>
#include <fstream>
#include <ctime>// include this header
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>

using namespace std;



struct gauss_params
	{
		double *x, *y, *err, *fit;
		int n, m;
	};



int gauss_f (const gsl_vector *v, void *p, gsl_vector *f)
	{
		double c = gsl_vector_get(v, 0);
		double k = gsl_vector_get(v, 1);
		double x0 = gsl_vector_get(v, 2);
		double fwhm = gsl_vector_get(v, 3);

		double *x = ((struct gauss_params *) p)->x;
		double *y = ((struct gauss_params *) p)->y;
		double *err = ((struct gauss_params *) p)->err;
		double *fit = ((struct gauss_params *) p)->fit;
		int n = ((struct gauss_params *) p)->n;

		int i;
		double sigma = fwhm/2/sqrt(2*log(2));

		for (i = 0; i < n; i++) {
			fit[i] = c + k * exp(-(x[i]-x0)*(x[i]-x0)/2/sigma/sigma);
			gsl_vector_set (f, i, (fit[i]-y[i])/err[i]);
		}

		return GSL_SUCCESS;
	}



int gauss_df (const gsl_vector *v, void *p, gsl_matrix *J)
	{
		double c = gsl_vector_get(v, 0);
		double k = gsl_vector_get(v, 1);
		double x0 = gsl_vector_get(v, 2);
		double fwhm = gsl_vector_get(v, 3);

		double *x = ((struct gauss_params *) p)->x;
		double *y = ((struct gauss_params *) p)->y;
		double *err = ((struct gauss_params *) p)->err;
		int n = ((struct gauss_params *) p)->n;

		int i;
		double sigma = fwhm/2/sqrt(2*log(2));

		for (i = 0; i < n; i++) {
			double e = exp(-(x[i]-x0)*(x[i]-x0)/2/sigma/sigma);
			gsl_matrix_set (J, i, 0, 1/err[i]);
			gsl_matrix_set (J, i, 1, e/err[i]);
			gsl_matrix_set (J, i, 2, k/err[i]*e*(x[i]-x0)/sigma/sigma);
			gsl_matrix_set (J, i, 3, k/err[i]*e*(x[i]-x0)*(x[i]-x0)/sigma/sigma/sigma/2/sqrt(2*log(2)));
		}

		return GSL_SUCCESS;
	}





void callback(const size_t iter, void *params,
         const gsl_multifit_nlinear_workspace *w)
{
  gsl_vector * x = gsl_multifit_nlinear_position(w);


}



int main(int argc, char *argv[]){


	struct gauss_params p;
    int vector_size = atoi(argv[1]);
    double *x;
    x = (double*)malloc( vector_size * sizeof(double) );
    p.y = (double*)malloc( vector_size * sizeof(double) );
    p.err = (double*)malloc( vector_size * sizeof(double) );

    ifstream input2("CCF_data.txt");

    for(int i = 0; i < vector_size; i++)
    {
    input2 >> x[i] >> p.y[i];
    }

    ifstream input3("CCF_error.txt");

    for(int i = 0; i < vector_size; i++)
    {
    input3 >> p.err[i];
    }

    p.n = vector_size;


	int i, j = 0, info, npar = 4;
	double c, k, x0, fwhm;
	double sig_c, sig_k, sig_x0, sig_fwhm;

    const double xtol = 1.0e-8;
    const double gtol = 1.0e-8;
    const double ftol = 0.0;
    const double max_iter=10000;
    double chisq, chisq0;

    double v_init[npar];
    gsl_matrix *covar = gsl_matrix_alloc (npar, npar);
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace *w;
    gsl_multifit_nlinear_fdf F;
    gsl_rng * r;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();

    gsl_vector *f;
    gsl_matrix *J;



	double weights[p.n];

	p.x = (double *) malloc(p.n*sizeof(double));
	p.fit = (double *) malloc(p.n*sizeof(double));

	for (i = 0; i < p.n; i++)
		p.x[i] = x[i]-x[0];

	c = (p.y[0]+p.y[p.n-1])/2;
	fwhm = (p.x[p.n-1]-p.x[0])/5.;
	k = 0.; x0 = p.x[p.n/2];

    for (i = 0; i < p.n; i++)
        if (fabs(p.y[i]-c) > fabs(k)) { k = p.y[i]-c; x0 = p.x[i]; }
    for (i = 0; i < p.n; i++)
        weights[i]=1.0;

    gsl_vector_view wts = gsl_vector_view_array(weights, p.n);
	gsl_rng_env_setup();
	r = gsl_rng_alloc(gsl_rng_default);

    F.f = &gauss_f;
    F.df = &gauss_df; // set to NULL for finite-difference Jacobian
    //F.fdf = &gauss_fdf;
    F.fvv = NULL; // not using geodesic acceleration
	F.n = p.n;
	F.p = npar;
	F.params = &p;

    v_init[0]= c;
    v_init[1]= k;
    v_init[2]= x0;
    v_init[3]= fwhm;
    gsl_vector_view v = gsl_vector_view_array (v_init, npar);

    /* allocate workspace with default parameters */

    //printf("%i\n",p.n);
    w = gsl_multifit_nlinear_alloc(T, &fdf_params, p.n, npar);
    /* initialize solver with starting point and weights */
    gsl_multifit_nlinear_winit (&v.vector, &wts.vector, &F, w);
    /* compute initial cost function */
    f = gsl_multifit_nlinear_residual(w);
    gsl_blas_ddot(f, f, &chisq0);
    /* solve the system with a maximum of max_iter iterations */
    gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol,callback, NULL, &info, w);
    /* compute covariance of best fit parameters */
    J = gsl_multifit_nlinear_jac(w);
    gsl_multifit_nlinear_covar (J, 0.0, covar);

    /* compute final cost */
    gsl_blas_ddot(f, f, &chisq);


    c = gsl_vector_get(w->x, 0); sig_c = sqrt(gsl_matrix_get(covar,0,0));
    k = gsl_vector_get(w->x, 1); sig_k = sqrt(gsl_matrix_get(covar,1,1));
    x0 = gsl_vector_get(w->x, 2); sig_x0 = sqrt(gsl_matrix_get(covar,2,2));
    fwhm = gsl_vector_get(w->x, 3); sig_fwhm = sqrt(gsl_matrix_get(covar,3,3));


    double *dd;
    dd = (double*)malloc( vector_size * sizeof(double) );

	for (i = 0; i < p.n; i++){
        dd[i] = p.fit[i];
    }

	/* Bisector part : */
	// Declarations...
	int    nstep=100, margin=5, len_depth=nstep-2*margin+1;
	double sigma = fwhm/2./pow(2.*log(2.),.5), vr, v0=x0+x[0];
	double dCCFdRV, d2CCFdRV2, d2RVdCCF2;
	double *norm_CCF, *depth, *bis, *p0, *p1, *p2;

	// Allocations...
	norm_CCF = (double *)malloc(sizeof(double)*p.n);
	depth    = (double *)malloc(sizeof(double)*len_depth);
	bis      = (double *)malloc(sizeof(double)*len_depth);
	p0       = (double *)malloc(sizeof(double)*p.n);
	p1       = (double *)malloc(sizeof(double)*p.n);
	p2       = (double *)malloc(sizeof(double)*p.n);

	// Initialization...
	for (i=0; i<p.n; i++) norm_CCF[i] = -c/k*(1.-p.y[i]/c);
	for (i=0; i<(nstep-2*margin+1); i++) depth[i] = (double )(i+margin)/nstep;

	for (i=0; i<p.n-1; i++) {
	  //if ((max(norm_CCF[i],norm_CCF[i+1]) >= depth[0]) &&
	  //   (min(norm_CCF[i],norm_CCF[i+1]) <= depth[p.n-1])){
	    vr = (x[i]+x[i+1])/2.;
	    dCCFdRV = -(vr-v0)/pow(sigma,2)*exp(-pow((vr-v0),2)/2./pow(sigma,2));
	    d2CCFdRV2 = (pow((vr-v0),2)/pow(sigma,2)-1)/pow(sigma,2)*exp(-pow((vr-v0),2)/2./pow(sigma,2));
	    d2RVdCCF2 = -d2CCFdRV2/pow(dCCFdRV,3);
	    p2[i] = d2RVdCCF2/2.;
	    p1[i] = (x[i+1]-x[i]-p2[i]*(pow(norm_CCF[i+1],2)-pow(norm_CCF[i],2)))/(norm_CCF[i+1]-norm_CCF[i]);
	    p0[i] = x[i]-p1[i]*norm_CCF[i]-p2[i]*pow(norm_CCF[i],2);
		   };//};

	int ind_max = 0, i_b, i_r;
	for (i=0; i<p.n; i++) if (norm_CCF[i]>norm_CCF[ind_max]) ind_max=i;
	for (i=0; i<len_depth; i++) {
	  i_b = ind_max; i_r = ind_max;
	  while ((norm_CCF[i_b] > depth[i]) && (i_b > 1)) i_b--;
	  while ((norm_CCF[i_r+1] > depth[i]) && (i_r < (p.n-2))) i_r++;
	  bis[i] = (p0[i_b]+p0[i_r]) + (p1[i_b]+p1[i_r])*depth[i] + (p2[i_b]+p2[i_r])*pow(depth[i],2);
	  bis[i] /= 2.;
	  //printf("%f\t%f\t%f\t%f\t%i\t%i\n", bis[i], depth[i], p0[i_b], p0[i_r], i_b, i_r);
	};

	int n1=0, n2=0;
	double RV_top=0., RV_bottom=0., span;
	for (i=0; i<len_depth; i++) {
	    if ((depth[i]>=0.1) && (depth[i] <= 0.4)) {
	        n1++;
		RV_top += bis[i];};
	    if ((depth[i]>=0.6) && (depth[i] <= 0.9)) {
	        n2++;
		RV_bottom += bis[i];};
	};
	RV_top    /= n1;
	RV_bottom /= n2;
	span = RV_top-RV_bottom;

    int bis_size = len_depth;
    double *ee;
    ee = (double*)malloc( bis_size * sizeof(double) );

    double *ff;
    ff = (double*)malloc( bis_size * sizeof(double) );

	for (i = 0; i < len_depth; i++)
		ee[i] =1.0- fabs(k)*depth[i];
    for (i = 0; i < len_depth; i++)
		ff[i] = bis[i];

    free(norm_CCF);free(depth);free(bis);free(p0);free(p1);free(p2);
    free(p.x);free(p.fit);

    gsl_matrix_free(covar);




    double *ccf_par;
    ccf_par = (double*)malloc( 5 * sizeof(double) );
    ccf_par[0] = c;
    ccf_par[1] = fabs(k);
    ccf_par[2] = span;
    ccf_par[3] = v0;
    ccf_par[4] = fwhm;

    string dataname_model=std::string("ccf_model.bin");
    FILE *file_model = fopen(dataname_model.c_str(),"wb");
        fwrite(dd,sizeof(double), vector_size, file_model);
        fclose(file_model);


    string dataname_par=std::string("ccf_parameter.bin");
    FILE *file_par = fopen(dataname_par.c_str(),"wb");
        fwrite(ccf_par,sizeof(double), 5, file_par);
        fclose(file_par);

    string dataname_depth=std::string("ccf_depth.bin");
    FILE *file_depth = fopen(dataname_depth.c_str(),"wb");
        fwrite(ee,sizeof(double), len_depth, file_depth);
        fclose(file_depth);

    string dataname_bis=std::string("ccf_bis.bin");
    FILE *file_bis = fopen(dataname_bis.c_str(),"wb");
        fwrite(ff,sizeof(double), len_depth, file_bis);
        fclose(file_bis);

    return 0;
}
