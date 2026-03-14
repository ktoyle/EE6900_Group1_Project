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
//                  The CNN architecture consists of 5 convolutional layers
//                  followed by 4 dense layers, producing one of 23 possible
//                  hand gesture classifications from the NinaPro DB1 dataset.
//                  Weights were pretrained in TensorFlow/Keras and exported
//                  as binary files for hardcoded inference on the FPGA.
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
    static data_t shift_reg[N_CHANNELS][N_TAPS] = {0};

    for (int i = N_TAPS - 1; i > 0; i--)
        shift_reg[ch][i] = shift_reg[ch][i-1];
    shift_reg[ch][0] = input;

    acc_t acc = 0;
    for (int i = 0; i < N_TAPS; i++)
        acc += shift_reg[ch][i] * lpf_coeffs[i];

    return (data_t)acc;
}


//----------------------------------------------------------------------------
// FIR High Pass Filter — one shift register per channel per filter
// Processes 1 sample and a single channel at time, maintains history via static shift register
//----------------------------------------------------------------------------
data_t fir_hpf(data_t input, int ch) {
    static data_t shift_reg[N_CHANNELS][N_TAPS] = {0};

    for (int i = N_TAPS - 1; i > 0; i--)
        shift_reg[ch][i] = shift_reg[ch][i-1];
    shift_reg[ch][0] = input;

    acc_t acc = 0;
    for (int i = 0; i < N_TAPS; i++)
        acc += shift_reg[ch][i] * hpf_coeffs[i];

    return (data_t)acc;
}

//----------------------------------------------------------------------------
// FIR Notch Filter — one shift register per channel per filter
// Processes 1 sample and a single channel at time, maintains history via static shift register
//----------------------------------------------------------------------------
data_t fir_notch(data_t input, int ch) {
    static data_t shift_reg[N_CHANNELS][N_TAPS] = {0};

    for (int i = N_TAPS - 1; i > 0; i--)
        shift_reg[ch][i] = shift_reg[ch][i-1];
    shift_reg[ch][0] = input;

    acc_t acc = 0;
    for (int i = 0; i < N_TAPS; i++)
        acc += shift_reg[ch][i] * notch_coeffs[i];

    return (data_t)acc;
}
//---------------------------------------------------------------------------
// Normalize emg_buffer between 0 and 1
// Min and Max are based on 52 data entries currently in the buffer
// Buffer gets reset after going through model
//--------------------------------------------------------------------------
void normalize_buffer(data_t buffer[WINDOW_SIZE][N_CHANNELS]) {
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        // Find min and max for this channel across the window
        data_t ch_min = buffer[0][ch];
        data_t ch_max = buffer[0][ch];

        for (int t = 1; t < WINDOW_SIZE; t++) {
            if (buffer[t][ch] < ch_min) ch_min = buffer[t][ch];
            if (buffer[t][ch] > ch_max) ch_max = buffer[t][ch];
        }

        // Normalize
        data_t range = ch_max - ch_min;
        for (int t = 0; t < WINDOW_SIZE; t++) {
            if (range > (data_t)0.0001) {  // avoid divide by zero
                buffer[t][ch] = (buffer[t][ch] - ch_min) / range;
            } else {
                buffer[t][ch] = (data_t)0;  // flat signal → all zeros
            }
        }
    }
}


//-----------------------------------------------------------------------------
// group1_top — Top Level Function (synthesized to hardware)
//
// This is the entry point for the HLS design. It is called once per time step
// (once per set of 10 new EMG samples, one per channel).
//
// Each call performs the following:
//   1. Reads one new sample from each of the 10 input streams
//   2. Passes each sample through the FIR filter chain (LPF -> HPF -> Notch)
//   3. Stores the filtered sample in the EMG accumulation buffer
//   4. Once the buffer holds WINDOW_SIZE (52) samples:
//      a. Normalizes the buffer to [0,1] per channel
//      b. Runs the full CNN forward pass
//      c. Writes the predicted gesture class index to the output stream
//      d. Resets the buffer counter for the next window
//
// Parameters:
//   in_stream  — array of 10 input streams, one per EMG channel
//   out_stream — output stream for predicted gesture class (integer 0-22)
//   conv0-4_w  — pretrained conv layer weight arrays (loaded from .bin files)
//   conv0-4_b  — pretrained conv layer bias arrays
//   dense0-3_w — pretrained dense layer weight arrays
//   dense0-3_b — pretrained dense layer bias arrays
//-----------------------------------------------------------------------------
void group1_top(hls::stream<float> in_stream[N_CHANNELS],
                hls::stream<int>   &out_stream,
                float conv0_w[], float conv0_b[],
                float conv1_w[], float conv1_b[],
                float conv2_w[], float conv2_b[],
                float conv3_w[], float conv3_b[],
                float conv4_w[], float conv4_b[],
                float dense0_w[], float dense0_b[],
                float dense1_w[], float dense1_b[],
                float dense2_w[], float dense2_b[],
                float dense3_w[], float dense3_b[]) {

     //-------------------------------------------------------------------------
    // Stage 1: Read, filter, and accumulate one sample per channel
    // Each channel is processed independently through the full FIR chain
    //-------------------------------------------------------------------------
    for (int ch = 0; ch < N_CHANNELS; ch++) {

        // Read raw sample from this channel's input stream
        data_t raw = static_cast<data_t>(in_stream[ch].read());

        // Apply FIR filter chain:
        // LPF removes high frequency noise above 500Hz
        data_t lpf_data   = fir_lpf(raw, ch);

        // HPF removes low frequency drift below 20Hz
        // Combined with LPF this creates a 20Hz-500Hz bandpass
        data_t band_data  = fir_hpf(lpf_data, ch);

        // Notch removes 60Hz US power line interference
        data_t notch_data = fir_notch(band_data, ch);

        // Store fully filtered sample in the accumulation buffer
        // buffer_count tracks the current time step position (0 to 51)
        emg_buffer[buffer_count][ch] = notch_data;
    }

    // Advance to the next time step position in the buffer
    buffer_count++;


    //-------------------------------------------------------------------------
    // Stage 2: Once buffer is full, run CNN and output prediction
    // This fires every WINDOW_SIZE (52) calls to group1_top
    //-------------------------------------------------------------------------
    if (buffer_count == WINDOW_SIZE) {

        // Normalize buffer values to [0,1] per channel
        // This makes the CNN input scale-invariant across subjects
        normalize_buffer(emg_buffer);

         // Run the full CNN forward pass and get predicted gesture class
        int gesture = cnn_forward(emg_buffer,
                                  conv0_w, conv0_b,
                                  conv1_w, conv1_b,
                                  conv2_w, conv2_b,
                                  conv3_w, conv3_b,
                                  conv4_w, conv4_b,
                                  dense0_w, dense0_b,
                                  dense1_w, dense1_b,
                                  dense2_w, dense2_b,
                                  dense3_w, dense3_b);

        // Write the predicted gesture index to the output stream                         
        out_stream.write(gesture);
        
        // Reset buffer counter — next 52 samples will fill a new window
        buffer_count = 0;
    }
}