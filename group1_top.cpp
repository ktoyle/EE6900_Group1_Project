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
// Inverted Bell Curve Coefficents for FIR high pass filter
// 20Hz cutoff at 1kHz sampling rate
//----------------------------------------------------------------------------

const data_t hpf_coeffs[N_TAPS] = {
    -0.0016, -0.0020, -0.0029, -0.0044, -0.0066,
    -0.0094, -0.0127, -0.0165, -0.0206, -0.0247,
    -0.0288, -0.0325, -0.0356, -0.0379, -0.0394,
    0.9585, -0.0394, -0.0379, -0.0356, -0.0325,
    -0.0288, -0.0247, -0.0206, -0.0165, -0.0127,
    -0.0094, -0.0066, -0.0044, -0.0029, -0.0020,
    -0.0016
};

//---------------------------------------------------------------------------
// Notch Filter coefficents
// Filter out 60Hz 
//---------------------------------------------------------------------------

const data_t notch_coeffs[N_TAPS] = {
    -0.0013, -0.0009, -0.0004, 0.0006, 0.0025,
    0.0050, 0.0077, 0.0098, 0.0103, 0.0087,
    0.0048, -0.0011,-0.0078, -0.0141, -0.0186,
    0.9898, -0.0186, -0.0141, -0.0078, -0.0011,
    0.0048, 0.0087, 0.0103, 0.0098, 0.0077,
    0.0050, 0.0025, 0.0006, -0.0004, -0.0009,
    -0.0013
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

//----------------------------------------------------------------------------
// FIR High Pass Filter
// Processes one sample at a time, maintains history via static shift register
//----------------------------------------------------------------------------
data_t fir_hpf(data_t input) {
    static data_t shift_reg_hpf[N_TAPS] = {0};  // persists between calls

    // Shift old samples along to the right
    for (int i = N_TAPS - 1; i > 0; i--) {
        shift_reg_hpf[i] = shift_reg_hpf[i - 1];
    }
    
    //load in newest sample
    shift_reg_hpf[0] = input;

    // Multiply-accumulate (convolve lpf_coeffs and shift_reg)
    acc_t acc = 0;
    for (int i = 0; i < N_TAPS; i++) {
        acc += shift_reg_hpf[i] * hpf_coeffs[i];
    }

    //return filtered data
    return (data_t)acc;
}

//----------------------------------------------------------------------------
// FIR Notch Filter
// Processes one sample at a time, maintains history via static shift register
//----------------------------------------------------------------------------
data_t fir_notch(data_t input) {
    static data_t shift_reg_notch[N_TAPS] = {0};  // persists between calls

    // Shift old samples along to the right
    for (int i = N_TAPS - 1; i > 0; i--) {
        shift_reg_notch[i] = shift_reg_notch[i - 1];
    }
    
    //load in newest sample
    shift_reg_notch[0] = input;

    // Multiply-accumulate (convolve lpf_coeffs and shift_reg)
    acc_t acc = 0;
    for (int i = 0; i < N_TAPS; i++) {
        acc += shift_reg_notch[i] * notch_coeffs[i];
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
        
         // Pass data through high pass filter (essentially a band filter now)
        data_t band_data = fir_hpf(lpf_data);

        //Pass data through notch filter to get final filtered data
        data_t filtered_data = fir_notch(band_data);

        // Write result to output stream
        out_stream.write(static_cast<float>(filtered_data));
    
}