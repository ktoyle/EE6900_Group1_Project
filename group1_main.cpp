#include <iostream>
#include <cmath>
#include "hls_stream.h"
#include "ap_fixed.h"

// Must match digital_filter.cpp
//typedef ap_fixed<16,12> data_t;

// Forward declare the top level function
void group1_top(hls::stream<float> &in_stream, hls::stream<float> &out_stream);

#define N_SAMPLES 100
#define PI 3.14159265358979f

int group1_main() {
    hls::stream<float> in_stream;
    hls::stream<float> out_stream;

    float fs = 1000.0f;

    // Generate test signal: 50Hz (should pass) + 600Hz (should be filtered out)
    for (int i = 0; i < N_SAMPLES; i++) {
        float t          = i / fs;
        float low_freq   = std::sin(2 * PI * 50  * t);  // should pass through
        float high_freq  = std::sin(2 * PI * 600 * t);  // should be removed
        float signal     = low_freq + high_freq;

        //in_stream.write(static_cast<float>(signal * 1000));  // scale to int
        in_stream.write(signal);  //write the raw float
    }

    // Run the filter
    for (int i = 0; i < N_SAMPLES; i++) {
        group1_top(in_stream, out_stream);
    }

    // Print results
    std::cout << "Sample | Output\n";
    std::cout << "-------+--------\n";
    for (int i = 0; i < N_SAMPLES; i++) {
        float val = out_stream.read();
        std::cout << i << "\t" << val << "\n";
    }

    std::cout << "VERSION: float stream test\n";


    return 0;
}