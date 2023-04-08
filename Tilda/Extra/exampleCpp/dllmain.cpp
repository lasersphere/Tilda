/*
Tilda.Extra.exampleCpp.dllmain.cpp

Created on 29.10.2021

@author: Patrick Mueller

C++ file to be compiled into a DLL. In order to compile, create a new DLL project (outside Tilda)
and replace the dllmain.cpp in the project with this one.
*/

// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

#include <complex>
#include <Eigen/Dense>  // Comment out if you don't have the Eigen library installed.

using namespace Eigen;  // Comment out if you don't have the Eigen library installed.

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

extern "C"  // Pythons ctypes is for C not C++.
{

    __declspec(dllexport) void matrix_multiplication_0(std::complex<double>* a, std::complex<double>* b,
        int* shape_a, int* shape_b, std::complex<double>* ret)
        // Simplest algorithm for matrix multiplication, scales with n^3.
    {   
        int dim_n = shape_a[0];
        int dim_k = shape_a[1];
        int dim_m = shape_b[1];
        std::complex<double> sum(0., 0.);
        for (int n = 0; n < dim_n; ++n) {
            for (int m = 0; m < dim_m; ++m) {
                sum = (0., 0.);
                for (int k = 0; k < dim_k; ++k) {
                    sum += a[n * dim_k + k] * b[k * dim_m + m];
                }
                ret[n * shape_b[1] + m] = sum;
            }
        }
    }
    
    __declspec(dllexport) void matrix_multiplication_opt(std::complex<double>* a, std::complex<double>* b,
        int* shape_a, int* shape_b, std::complex<double>* ret)
        // Optimized for loops, still scales with n^3.
    {   
        int dim_n = shape_a[0];
        int dim_k = shape_a[1];
        int dim_m = shape_b[1];
        int nd = 0;
        int md = 0;
        int ndm = 0;
        int nm = 0;
        std::complex<double> sum(0., 0.);
        for (int n = 0; n < dim_n; ++n) {
            nd = n * dim_k;
            ndm = n * dim_m;
            for (int m = 0; m < dim_m; ++m) {
                md = m * dim_k;
                nm = ndm + m;
                ret[nm] = (0.f, 0.f);
                for (int k = 0; k < dim_k; ++k) {
                    ret[nm] += a[nd + k] * b[md + k];
                }
            }
        }
    }

    // Comment out if you don't have the Eigen library installed.
    __declspec(dllexport) std::complex<double>* matrix_multiplication_eigen(std::complex<double>* a, std::complex<double>* b,
        int* shape_a, int* shape_b, std::complex<double>* ret)
        // Using the Eigen library. It is also slower than numpy. :|
    {
        Map<MatrixXcd> a1(a, shape_a[0], shape_a[1]);
        Map<MatrixXcd> b1(b, shape_b[0], shape_b[1]);
        Map<MatrixXcd> ret1(ret, shape_a[0], shape_b[1]);
        ret1 = a1 * b1;
    }

};
