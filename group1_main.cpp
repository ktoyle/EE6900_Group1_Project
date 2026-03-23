//=============================================================================
// Project:     EE6900 FPGA Accelerator Design - Group 1
// File:        group1_main.cpp
// Description: C simulation testbench for the EMG signal processing pipeline.
//
//              This file is NOT synthesized to hardware. It is only used
//              during Vitis HLS C simulation (csim) to verify that the
//              design logic in group1_top.cpp and cnn.cpp is functionally
//              correct before synthesis.
//
//              The testbench performs the following:
//                1. Loads pretrained CNN weights from .bin files into
//                   float arrays (same approach as the ResNet reference design)
//                2. Generates a synthetic test signal containing three
//                   frequency components:
//                     - 50Hz  (in-band, should pass through filters)
//                     - 600Hz (above LPF cutoff, should be removed)
//                     - 60Hz  (power line noise, should be removed by notch)
//                3. Feeds the signal into group1_top one time step at a time,
//                   simulating a real-time 1kHz EMG stream across 10 channels
//                4. Prints the predicted gesture class for each completed
//                   52-sample window
//
//              Expected output: 9 windows completed (500 samples / 52 = 9)
//              Gesture outputs will all be 0 for the synthetic signal since
//              the model has never seen sine wave data. To properly validate
//              the CNN, real NinaPro EMG data should be used as input.
//
// Tool:        Vitis HLS 2024.1 (csim only)
//=============================================================================

#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "cnn.h"

//--------------------------------------------------------------------------
// Testbench Configuration
// N_SAMPLES: total number of time steps to simulate
//            500 samples at 1kHz = 0.5 seconds of signal
//            produces 9 complete 52-sample windows (500 / 52 = 9)
//--------------------------------------------------------------------------

#define N_SAMPLES   500
#define PI          3.14159265358979f
#define N_CHANNELS  10


//--------------------------------------------------------------------------
// Forward declaration of top level function
// Must match the signature in group1_top.cpp exactly
//--------------------------------------------------------------------------
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
                weight_t dense3_w[], weight_t dense3_b[]);

//--------------------------------------------------------------------------
// CNN Weight Arrays
// Sized to match the pretrained Keras model architecture exactly:
//   Conv layers:  (kernel x in_ch x out_ch) flattened
//   Dense layers: (in_size x out_size) flattened
// All weights are float32 to match the .bin file format
//--------------------------------------------------------------------------
weight_t conv0_w[3*10*128],  conv0_b[128];
weight_t conv1_w[3*128*64],  conv1_b[64];
weight_t conv2_w[3*64*32],   conv2_b[32];
weight_t conv3_w[3*32*16],   conv3_b[16];
weight_t conv4_w[3*16*8],    conv4_b[8];
weight_t dense0_w[8*512],    dense0_b[512];
weight_t dense1_w[512*256],  dense1_b[256];
weight_t dense2_w[256*128],  dense2_b[128];
weight_t dense3_w[128*23],   dense3_b[23];

//-----------------------------------------------------------------------------
// load_weights
//
// Reads raw float32 binary data from a file into a float array.
// The .bin files are saved by the Python training script using
// numpy's tofile() which writes raw float32 bytes with no header.
//
// Parameters:
//   path — path to the .bin weight file
//   buf  — destination float array to load weights into
//   n    — number of float values to read
//-----------------------------------------------------------------------------
void load_weights(const char* path, weight_t* buf, int n) {
    std::ifstream f(path, std::ios::binary);
    f.read((char*)buf, n * sizeof(float));
    f.close();
}

//-----------------------------------------------------------------------------
// read_bin_files
//
// Loads all pretrained CNN weights and biases from .bin files into the
// global float arrays declared above. Must be called before any inference.
//
// The weights were exported from the trained Keras model using:
//   layer.get_weights()[0].astype(np.float32).tofile('weights/name.bin')
//
// File naming convention matches Keras layer names:
//   conv1d_weights.bin    = first Conv1D layer weights
//   conv1d_1_weights.bin  = second Conv1D layer weights
//   dense_weights.bin     = first Dense layer weights
//   etc.
//-----------------------------------------------------------------------------
void read_bin_files() {

    // Load all 5 convolutional layer weights and biases
    load_weights("weights/conv1d_weights.bin",   conv0_w, 3*10*128);
    load_weights("weights/conv1d_bias.bin",      conv0_b, 128);
    load_weights("weights/conv1d_1_weights.bin", conv1_w, 3*128*64);
    load_weights("weights/conv1d_1_bias.bin",    conv1_b, 64);
    load_weights("weights/conv1d_2_weights.bin", conv2_w, 3*64*32);
    load_weights("weights/conv1d_2_bias.bin",    conv2_b, 32);
    load_weights("weights/conv1d_3_weights.bin", conv3_w, 3*32*16);
    load_weights("weights/conv1d_3_bias.bin",    conv3_b, 16);
    load_weights("weights/conv1d_4_weights.bin", conv4_w, 3*16*8);
    load_weights("weights/conv1d_4_bias.bin",    conv4_b, 8);

    // Load all 4 dense layer weights and biases
    load_weights("weights/dense_weights.bin",    dense0_w, 8*512);
    load_weights("weights/dense_bias.bin",       dense0_b, 512);
    load_weights("weights/dense_1_weights.bin",  dense1_w, 512*256);
    load_weights("weights/dense_1_bias.bin",     dense1_b, 256);
    load_weights("weights/dense_2_weights.bin",  dense2_w, 256*128);
    load_weights("weights/dense_2_bias.bin",     dense2_b, 128);
    load_weights("weights/dense_3_weights.bin",  dense3_w, 128*23);
    load_weights("weights/dense_3_bias.bin",     dense3_b, 23);

    std::cout << "Weights loaded successfully\n";
}

int main() {

    // Step 1: Load pretrained weights from .bin files
    read_bin_files();

    // Step 2: Create HLS streams — one input stream per EMG channel
    // plus one output stream for gesture class predictions
    hls::stream<float> in_stream[N_CHANNELS];
    hls::stream<int>   out_stream;

    // Sampling rate — determines frequency of the test signal components
    float fs = 1000.0f;

     // Step 3: Generate synthetic test signal and write to all input streams
    for (int i = 0; i < N_SAMPLES; i++) {
        float t = i / fs;
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            float signal = std::sin(2 * PI * 50  * t)
                         + std::sin(2 * PI * 600 * t)
                         + std::sin(2 * PI * 60  * t);
            in_stream[ch].write(signal);
        }
    }

     // Step 4: Run the filter + CNN pipeline one time step at a time
    // group1_top is called N_SAMPLES times, accumulating filtered samples
    // into a 52-sample buffer. Every 52 calls the CNN fires and a gesture
    // prediction is written to out_stream.
    int windows_completed = 0;


    for (int i = 0; i < N_SAMPLES; i++) {
        // Process one time step across all 10 channels
        group1_top(in_stream, out_stream,
                   conv0_w, conv0_b,
                   conv1_w, conv1_b,
                   conv2_w, conv2_b,
                   conv3_w, conv3_b,
                   conv4_w, conv4_b,
                   dense0_w, dense0_b,
                   dense1_w, dense1_b,
                   dense2_w, dense2_b,
                   dense3_w, dense3_b);

        // Check if a complete window was processed and a prediction was made
        // out_stream only has data when buffer_count hits WINDOW_SIZE (52)
        if (!out_stream.empty()) {
            int gesture = out_stream.read();
            windows_completed++;

            // Print window number and predicted gesture class (0-22)
            std::cout << "Window " << windows_completed
                      << " Gesture: " << gesture << "\n";
        }
    }

     // Summary: should print 9 (500 samples / 52 samples per window = 9)
    std::cout << "\nTotal windows: " << windows_completed << "\n";
    return 0;
}