//=============================================================================
// Project:     EE6900 FPGA Accelerator Design — Group 1
// File:        group1_top.cpp
// Authors:     Kevin Toyle, Timothy Marchant, Aaron Zaman
//
// Description: Real-time EMG signal processing pipeline implemented in
//              Vitis HLS for deployment on the Xilinx PYNQ-Z2 xc7z020 FPGA.
//
//              The pipeline consists of two stages:
//
//              Stage 1 — Digital Filtering:
//                  Incoming raw EMG samples are passed through a chain of
//                  three FIR filters per channel:
//                    - Low Pass Filter  (500Hz cutoff, 1kHz sample rate)
//                    - High Pass Filter (20Hz  cutoff, 1kHz sample rate)
//                    - Notch Filter     (60Hz, removes US power line noise)
//                  Each of the 10 EMG channels maintains independent filter
//                  history via per-channel shift registers.
//
//              Stage 2 — 1D-CNN Inference:
//                  Filtered samples are accumulated into a (52 x 10) window
//                  buffer. Once full, the buffer is normalized to [0,1] and
//                  passed through a 1D Convolutional Neural Network (1D-CNN)
//                  for gesture classification.
//
//                  CNN weights are loaded from DRAM in small tiles during
//                  inference to fit within the xc7z020's BRAM capacity.
//
// Target:      Xilinx PYZQ-Z2 xc7z020clg400-1
// Tool:        Vitis HLS 2024.1
// Date:        March 2026
// 
// References: 
//              Dataset: https://www.kaggle.com/datasets/mansibmursalin/ninapro-db1-full-dataset
//              Paper 1: https://www.nature.com/articles/sdata201453
//              Paper 2: https://ieeexplore.ieee.org/abstract/document/10083548
//
//=============================================================================

#include <hls_math.h>
#include "hls_stream.h"
#include <ap_fixed.h>
#include "cnn.h"

//--------------------------------------------------------------------------
// Helper Definitions
//--------------------------------------------------------------------------

#define N_TAPS 31 //size of filter coeffs array

//-------------------------------------------------------------------------
// Global Variables
//-------------------------------------------------------------------------

data_t emg_buffer[WINDOW_SIZE][N_CHANNELS] = {0}; //array that holds data for 1D-CNN
int buffer_count = 0; //variable that keeps track of current buffer index

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
// FIR Low Pass Filter — one shift register per channel per filter
// Processes 1 sample and a single channel at time, maintains history via static shift register
//----------------------------------------------------------------------------
data_t fir_lpf(data_t input, int ch) {

    //Inline to create hardware copy when calling the function
    #pragma HLS INLINE

    static data_t shift_reg[N_CHANNELS][N_TAPS] = {0};

    // partition by channel
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=1 
    // partial on taps 
    #pragma HLS ARRAY_PARTITION variable=shift_reg cyclic factor=8 dim=2  

    // PIPELINE: shift all taps in parallel each cycle
    LPF_SHIFT:
    for (int i = N_TAPS - 1; i > 0; i--){
        #pragma HLS PIPELINE II=1
        shift_reg[ch][i] = shift_reg[ch][i-1];
    }

    shift_reg[ch][0] = input;

    acc_t acc = 0;

    // UNROLL: compute all 31 MACs simultaneously
    LPF_ACC:
    for (int i = 0; i < N_TAPS; i++){
        #pragma HLS UNROLL
        acc += shift_reg[ch][i] * lpf_coeffs[i];
    }

    return (data_t)acc;
}


//----------------------------------------------------------------------------
// FIR High Pass Filter — one shift register per channel per filter
// Processes 1 sample and a single channel at time, maintains history via static shift register
//----------------------------------------------------------------------------
data_t fir_hpf(data_t input, int ch) {
    #pragma HLS INLINE
    static data_t shift_reg[N_CHANNELS][N_TAPS] = {0};

    // partition by channel
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=1 
    // partial on taps 
    #pragma HLS ARRAY_PARTITION variable=shift_reg cyclic factor=8 dim=2  

    HPF_SHIFT:
    for (int i = N_TAPS - 1; i > 0; i--){
        #pragma HLS PIPELINE II=1
        shift_reg[ch][i] = shift_reg[ch][i-1];
    }

    shift_reg[ch][0] = input;

    acc_t acc = 0;

    // UNROLL: compute all 31 MACs simultaneously
    HPF_ACC:
    for (int i = 0; i < N_TAPS; i++){
        #pragma HLS UNROLL
        acc += shift_reg[ch][i] * hpf_coeffs[i];
    }

    return (data_t)acc;
}

//----------------------------------------------------------------------------
// FIR Notch Filter — one shift register per channel per filter
// Processes 1 sample and a single channel at time, maintains history via static shift register
//----------------------------------------------------------------------------
data_t fir_notch(data_t input, int ch) {

    #pragma HLS INLINE

    static data_t shift_reg[N_CHANNELS][N_TAPS] = {0};

    // partition by channel
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=1 
    // partial on taps 
    #pragma HLS ARRAY_PARTITION variable=shift_reg cyclic factor=8 dim=2  

    NOTCH_SHIFT:
    for (int i = N_TAPS - 1; i > 0; i--){
        #pragma HLS PIPELINE II=1
        shift_reg[ch][i] = shift_reg[ch][i-1];
    }

    shift_reg[ch][0] = input;

    acc_t acc = 0;

    NOTCH_ACC:
    for (int i = 0; i < N_TAPS; i++){
        #pragma HLS UNROLL
        acc += shift_reg[ch][i] * notch_coeffs[i];
    }

    return (data_t)acc;
}

//---------------------------------------------------------------------------
// Normalize emg_buffer between 0 and 1
// Min and Max are based on 52 data entries currently in the buffer
// Buffer gets reset after going through model
//--------------------------------------------------------------------------
void normalize_buffer(data_t buffer[WINDOW_SIZE][N_CHANNELS]) {

    #pragma HLS INLINE

    NORM_MAX_MIN:
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        // Find min and max for this channel across the window
        data_t ch_min = buffer[0][ch];
        data_t ch_max = buffer[0][ch];

        NORM_TRUE_MAX_MIN:
        for (int t = 1; t < WINDOW_SIZE; t++) {
            #pragma HLS PIPELINE II=1
            if (buffer[t][ch] < ch_min) ch_min = buffer[t][ch];
            if (buffer[t][ch] > ch_max) ch_max = buffer[t][ch];
        }

        // Normalize
        data_t range = ch_max - ch_min;

        NORM_CALC:
        for (int t = 0; t < WINDOW_SIZE; t++) {
            #pragma HLS PIPELINE II=1
            if (range > (data_t)0.0001) {
                buffer[t][ch] = (buffer[t][ch] - ch_min) / range;
            } else {
                buffer[t][ch] = (data_t)0;
            }
        }
    }
}


//-----------------------------------------------------------------------------
// group1_top — Top Level Function (synthesized to hardware)
//
// Weight arrays are passed directly to cnn_forward as DRAM pointers.
// cnn_forward internally tiles the weight loading into small BRAM buffers,
// so no bulk weight buffering is needed here.
//-----------------------------------------------------------------------------
void group1_top(hls::stream<float> in_stream[N_CHANNELS],
                hls::stream<int>   &out_stream,
                weight_t conv0_w[], weight_t conv0_b[],
                weight_t conv1_w[], weight_t conv1_b[],
                weight_t conv2_w[], weight_t conv2_b[],
                weight_t conv3_w[], weight_t conv3_b[],
                weight_t conv4_w[], weight_t conv4_b[],
                weight_t dense0_w[], weight_t dense0_b[],
                weight_t dense1_w[], weight_t dense1_b[],
                weight_t dense2_w[], weight_t dense2_b[],
                weight_t dense3_w[], weight_t dense3_b[]
                
                #ifdef CSIM_DEBUG
                , float debug_logits [N_CLASSES]
                #endif
            
                ) {

    //-------------------------------------------------------------------------
    // HLS Interface Pragmas
    //-------------------------------------------------------------------------
    #pragma HLS INTERFACE axis      port=in_stream
    #pragma HLS INTERFACE axis      port=out_stream
    #pragma HLS INTERFACE m_axi depth=3840   port=conv0_w  bundle=weights
    #pragma HLS INTERFACE m_axi depth=128    port=conv0_b  bundle=weights
    #pragma HLS INTERFACE m_axi depth=24576  port=conv1_w  bundle=weights
    #pragma HLS INTERFACE m_axi depth=64     port=conv1_b  bundle=weights
    #pragma HLS INTERFACE m_axi depth=6144   port=conv2_w  bundle=weights
    #pragma HLS INTERFACE m_axi depth=32     port=conv2_b  bundle=weights
    #pragma HLS INTERFACE m_axi depth=1536   port=conv3_w  bundle=weights
    #pragma HLS INTERFACE m_axi depth=16     port=conv3_b  bundle=weights
    #pragma HLS INTERFACE m_axi depth=384    port=conv4_w  bundle=weights
    #pragma HLS INTERFACE m_axi depth=8      port=conv4_b  bundle=weights
    #pragma HLS INTERFACE m_axi depth=4096   port=dense0_w bundle=weights
    #pragma HLS INTERFACE m_axi depth=512    port=dense0_b bundle=weights
    #pragma HLS INTERFACE m_axi depth=131072 port=dense1_w bundle=weights
    #pragma HLS INTERFACE m_axi depth=256    port=dense1_b bundle=weights
    #pragma HLS INTERFACE m_axi depth=32768  port=dense2_w bundle=weights
    #pragma HLS INTERFACE m_axi depth=128    port=dense2_b bundle=weights
    #pragma HLS INTERFACE m_axi depth=2944   port=dense3_w bundle=weights
    #pragma HLS INTERFACE m_axi depth=23     port=dense3_b bundle=weights
    #pragma HLS INTERFACE s_axilite port=return

    

    // Partition emg_buffer along channel dimension
    #pragma HLS ARRAY_PARTITION variable=emg_buffer complete dim=2 

    //-------------------------------------------------------------------------
    // Stage 1: Read, filter, and accumulate one sample per channel
    //-------------------------------------------------------------------------
    TOP_LOOP:
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        #pragma HLS PIPELINE II=2

        // Read raw sample from this channel's input stream
        data_t raw = static_cast<data_t>(in_stream[ch].read());

        // Apply FIR filter chain: LPF → HPF → Notch
        data_t lpf_data   = fir_lpf(raw, ch);
        data_t band_data  = fir_hpf(lpf_data, ch);
        data_t notch_data = fir_notch(band_data, ch);

        // Store filtered sample in accumulation buffer
        emg_buffer[buffer_count][ch] = notch_data;
    }

    // Advance to the next time step
    buffer_count++;

    //-------------------------------------------------------------------------
    // Stage 2: Once buffer is full, run CNN and output prediction
    //-------------------------------------------------------------------------
    if (buffer_count == WINDOW_SIZE) {

        // Normalize buffer values to [0,1] per channel
        normalize_buffer(emg_buffer);

        // Run the full CNN forward pass — weights are tiled internally
        int gesture = cnn_forward(emg_buffer,
                                  conv0_w, conv0_b,
                                  conv1_w, conv1_b,
                                  conv2_w, conv2_b,
                                  conv3_w, conv3_b,
                                  conv4_w, conv4_b,
                                  dense0_w, dense0_b,
                                  dense1_w, dense1_b,
                                  dense2_w, dense2_b,
                                  dense3_w, dense3_b
                                  #ifdef CSIM_DEBUG
                                    , debug_logits
                                  #endif
                                  );

        // Write the predicted gesture index to the output stream                         
        out_stream.write(gesture);
        
        // Reset buffer counter
        buffer_count = 0;
    }
}