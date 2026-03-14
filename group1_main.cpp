#include <iostream>
#include <cmath>
#include "hls_stream.h"
#include "ap_fixed.h"

#define N_SAMPLES   500
#define PI          3.14159265358979f
#define WINDOW_SIZE 52
#define N_CHANNELS  10

// Forward declare top level function
void group1_top(hls::stream<float> in_stream[N_CHANNELS], hls::stream<int> &out_stream);

int main() {
    hls::stream<float> in_stream[N_CHANNELS];
    hls::stream<int>   out_stream;

    float fs = 1000.0f;

    // Generate test signal for all 10 channels
    // Each channel gets a slightly different mix to simulate real EMG
    for (int i = 0; i < N_SAMPLES; i++) {
        float t = i / fs;
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            float low_freq   = std::sin(2 * PI * 50  * t);  // should pass
            float high_freq  = std::sin(2 * PI * 600 * t);  // should be removed
            float powerline  = std::sin(2 * PI * 60  * t);  // should be removed
            float signal     = low_freq + high_freq + powerline;
            in_stream[ch].write(signal);
        }
    }

    // Run the filter — one call per time step
    // Every WINDOW_SIZE calls the buffer fills and CNN would fire
     int windows_completed = 0;

    for (int i = 0; i < N_SAMPLES; i++) {
    group1_top(in_stream, out_stream);

    // Check if a window was written to out_stream
        if (!out_stream.empty()) {
            int dummy = out_stream.read();
            windows_completed++;
            std::cout << "Window " << windows_completed << " completed at sample " << i + 1 << "\n";
        }
    }

    std::cout << "\nTotal windows completed: " << windows_completed << "\n";
    std::cout << "Expected: " << N_SAMPLES / WINDOW_SIZE << "\n";

    return 0;
}