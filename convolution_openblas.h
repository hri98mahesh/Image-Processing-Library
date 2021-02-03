#include <iostream>
#include <vector>
#include "mkl.h"
using namespace std;
#ifndef CONVOLUTION_OPENBLAS_H_INCLUDE
#define CONVOLUTION_OPENBLAS_H_INCLUDE
vector<vector<float>> matrix_mult_openblas(vector<vector<float>> input,vector<vector<float>> kernel){
    
    int i, j;
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    double * A = new double(input.size()*input[0].size());
    int k=0;
    for(int i =0;i<input.size();i++){
      for(int j=0;j<input[0].size();j++){
         A[k]=input[i][j];
         k++;
      }
    }
    double * B= new double(kernel.size()*kernel[0].size());
     k=0;
    for(int i =0;i<kernel.size();i++){
      for(int j=0;j<kernel[0].size();j++){
         B[k]=kernel[i][j];
         k++;
      }
    }
    double * C = new double(input.size()*kernel[0].size());
   
    //printf (" Intializing matrix data \n\n")
    for (i = 0; i < input.size()*kernel[0].size(); i++) {
        C[i] = 0.0;
    }
   // printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, input.size(), kernel[0].size(), input[0].size(), alpha, A,input[0].size() , B, kernel[0].size(), beta, C, kernel[0].size());
    vector<vector<float>> output;
    for (i=0; i<input.size(); i++) {
           vector<float> v;
           output.push_back(v);
      for (j=0; j<kernel[0].size(); j++) {
         output[i].push_back(C[j+i*kernel[0].size()]);
      }
    }
    return output;
}
#endif