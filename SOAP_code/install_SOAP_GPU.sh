#!/bin/bash
echo "Start to install SOAP_GPU"
nvcc -o SOAP_initialize SOAP_initialize.cu
nvcc -o SOAP_integration SOAP_integration.cu
nvcc -o SOAP_integration_solar SOAP_integration_solar.cu
nvcc -o SOAP_resolution SOAP_resolution.cu
echo "Start to install GSL related code"
g++ BIS_FIT2.cpp -o BIS_FIT2 `gsl-config --cflags --libs`
echo -n "Done"
