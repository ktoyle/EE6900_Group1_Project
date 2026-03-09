#include <hls_math.h>
#include "hls_stream.h"
#include <iostream>
#include <cmath>
#include <ap_fixed.h>

//--------------------------------------------------------------------------
// Type conversions for simulation and synthesis 
// (Need to use ap_fixed instead of float for synthesis)
//--------------------------------------------------------------------------
#ifdef  CSIM_DEBUG
    typedef float data_t;
    typedef float acc_t;
#else
    typedef ap_fixed<16,4> data_t;
    typedef ap_fixed<32, 8> acc_t;
#endif

//--------------------------------------------------------------------------
// Helper Definitions
//--------------------------------------------------------------------------

#define N_TAPS 31

//----------------------------------------------------------------------------
// Bell Curve Coefficents for FIR low pass filter
// 500Hz cutoff at 1kHz sampling rate
//----------------------------------------------------------------------------

const data_t lpf_coeffs[N_TAPS] = {
    -0.0017, -0.0029, -0.0038, -0.0031,  0.0000,
     0.0069,  0.0182,  0.0328,  0.0487,  0.0633,
     0.0739,  0.0785,  0.0759,  0.0664,  0.0516,
     0.0345,  0.0181,  0.0053, -0.0022, -0.0044,
    -0.0031,  0.0000,  0.0031,  0.0044,  0.0022,
    -0.0053, -0.0181, -0.0345, -0.0516, -0.0664,
    -0.0759
};

//----------------------------------------------------------------------------
// FIR Low Pass Filter
// Processes one sample at a time, maintains history via static shift register
//----------------------------------------------------------------------------
data_t fir_lpf(data_t input) {
    static data_t shift_reg[N_TAPS] = {0};  // persists between calls

    // Shift old samples along to the right
    for (int i = N_TAPS - 1; i > 0; i--) {
        shift_reg[i] = shift_reg[i - 1];
    }
    
    //load in newest sample
    shift_reg[0] = input;

    // Multiply-accumulate (convolve lpf_coeffs and shift_reg)
    acc_t acc = 0;
    for (int i = 0; i < N_TAPS; i++) {
        acc += shift_reg[i] * lpf_coeffs[i];
    }

    //return filtered data
    return (data_t)acc;
}

//-----------------------------------------------------------------------------
// Top level function that gets synthesized
// Reads samples from stream, filters them, writes to output stream
//-----------------------------------------------------------------------------
void group1_top(hls::stream<float> &in_stream, hls::stream<float> &out_stream) {

    
        // Read one sample
        data_t sample = static_cast<data_t>(in_stream.read());

        // Filter it immediately — shift register maintains history across iterations
        data_t lpf_data = fir_lpf(sample);

        // Write result to output stream
        out_stream.write(static_cast<float>(lpf_data));
    
}