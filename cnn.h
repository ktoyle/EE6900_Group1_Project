//=============================================================================
// File:        cnn.h
// Description: Header for 1D-CNN forward pass
//=============================================================================
#ifndef CNN_H
#define CNN_H

#include <ap_fixed.h>


//--------------------------------------------------------------------------
// Type conversions for simulation and synthesis 
// (Need to use ap_fixed instead of float for synthesis)
//--------------------------------------------------------------------------

#ifdef CSIM_DEBUG
    typedef float data_t;
    typedef float acc_t;
    typedef float weight_t;
#else
    typedef ap_fixed<16,4> data_t;
    typedef ap_fixed<32,8> acc_t;
    typedef ap_fixed<32,10> weight_t;
#endif

#define WINDOW_SIZE  52
#define N_CHANNELS   10
#define N_CLASSES    23

// Conv layer dimensions
#define C0_IN   10
#define C0_OUT  128
#define C1_IN   128
#define C1_OUT  64
#define C2_IN   64
#define C2_OUT  32
#define C3_IN   32
#define C3_OUT  16
#define C4_IN   16
#define C4_OUT  8
#define KERNEL  3

// Dense layer dimensions
#define D0_IN   8
#define D0_OUT  512
#define D1_IN   512
#define D1_OUT  256
#define D2_IN   256
#define D2_OUT  128
#define D3_IN   128
#define D3_OUT  23

int cnn_forward(
    data_t input[WINDOW_SIZE][N_CHANNELS],
    // Conv weights
    weight_t conv0_w[KERNEL * C0_IN * C0_OUT], weight_t conv0_b[C0_OUT],
    weight_t conv1_w[KERNEL * C1_IN * C1_OUT], weight_t conv1_b[C1_OUT],
    weight_t conv2_w[KERNEL * C2_IN * C2_OUT], weight_t conv2_b[C2_OUT],
    weight_t conv3_w[KERNEL * C3_IN * C3_OUT], weight_t conv3_b[C3_OUT],
    weight_t conv4_w[KERNEL * C4_IN * C4_OUT], weight_t conv4_b[C4_OUT],
    // Dense weights
    weight_t dense0_w[D0_IN * D0_OUT], weight_t dense0_b[D0_OUT],
    weight_t dense1_w[D1_IN * D1_OUT], weight_t dense1_b[D1_OUT],
    weight_t dense2_w[D2_IN * D2_OUT], weight_t dense2_b[D2_OUT],
    weight_t dense3_w[D3_IN * D3_OUT], weight_t dense3_b[D3_OUT]
);

#endif