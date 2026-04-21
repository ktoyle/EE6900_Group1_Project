//=============================================================================
// Project:     EE6900 FPGA Accelerator Design - Group 1
// File:        group1_main.cpp
// Description: Testbench for the real-time EMG inference pipeline.
//
//              This testbench provides the verification environment for the 
//              `group1_top` HLS module. It simulates the hardware streaming 
//              interface and validates the end-to-end classification accuracy 
//              using pre-recorded NinaPro DB1 sensor data.
//
// Operations performed:
//   1. Weight Initialization: Loads pretrained CNN weights and biases from 
//      binary files into global buffers for hardware simulation.
//   2. Data Ingestion: Parses the NinaPro CSV dataset, filtering out noise 
//      and rest-state samples to generate valid input windows.
//   3. Hardware Streaming: Simulates a 1kHz real-time stream by feeding 
//      normalized EMG samples into the `hls::stream` interface of the 
//      accelerator design.
//   4. Inference & Verification: Executes the 1D-CNN forward pass, calculates 
//      class probabilities via softmax, and performs ground-truth comparison 
//      to report system accuracy and confidence levels.
//
// Notes: 
//   - Requires weights/ directory and Ninapro_DB1_small.csv in the 
//     working directory.
//   - Uses CSIM_DEBUG macro to toggle diagnostic logit output.
//=============================================================================


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "hls_stream.h"
#include "cnn.h"
#include <cstdlib>  

//--------------------------------------------------------------------------
// Testbench Configuration
//--------------------------------------------------------------------------
#define CSV_PATH       "Ninapro_DB1_small.csv"
#define N_SAMPLES      52
#define N_CHANNELS     10
#define EMG_FIRST_COL  1
#define EMG_LAST_COL   10
#define LABEL_COL      35   // restimulus

// Update this to the full path to your project folder
const char* weight_base_path = "C:/Users/kevto/Documents/School/Masters/Spring2026/EE6900_FPGA_Accel_Design/project_workspace/ee6900_group1_project/weights/";

//-----------------------------------------------------------------------------
// Forward declaration — must match group1_top.cpp exactly
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
                    , float debug_logits[N_CLASSES]
                #endif
                );

//-----------------------------------------------------------------------------
// softmax activation
//-----------------------------------------------------------------------------
void softmax(const float logits[N_CLASSES], float probs[N_CLASSES]) {
    float max_logit = logits[0];
    for (int i = 1; i < N_CLASSES; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < N_CLASSES; i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < N_CLASSES; i++) {
        probs[i] /= sum;
    }
}

//-----------------------------------------------------------------------------
// Weight arrays
//-----------------------------------------------------------------------------
weight_t conv0_w[3*10*128],  conv0_b[128];
weight_t conv1_w[3*128*64],  conv1_b[64];
weight_t conv2_w[3*64*32],   conv2_b[32];
weight_t conv3_w[3*32*16],   conv3_b[16];
weight_t conv4_w[3*16*8],    conv4_b[8];
weight_t dense0_w[8*512],    dense0_b[512];
weight_t dense1_w[512*256],  dense1_b[256];
weight_t dense2_w[256*128],  dense2_b[128];
weight_t dense3_w[128*23],   dense3_b[23];


// Function that loads weights from the path
void load_weights(const char* filename, weight_t* buf, int n) {

    // Combine the base path and the filename
    std::string full_path = std::string(weight_base_path) + std::string(filename);
    
    std::ifstream f(full_path, std::ios::binary);

    if (!f.is_open()) {
        std::cerr << "ERROR: Could not open weight file: " << full_path << std::endl;
        exit(1); 
    }
    
    f.read((char*)buf, n * sizeof(float));

    if (f.gcount() != n * sizeof(float)) {
        std::cerr << "ERROR: Failed to read expected amount of data from: " << full_path 
                  << " (Read " << f.gcount() << " bytes, expected " << n * sizeof(float) << ")" << std::endl;
        exit(1);
    }
    f.close();
}


void read_bin_files() {

    load_weights("conv1d_weights.bin",   conv0_w, 3*10*128);
    load_weights("conv1d_bias.bin",      conv0_b, 128);
    load_weights("conv1d_1_weights.bin", conv1_w, 3*128*64);
    load_weights("conv1d_1_bias.bin",    conv1_b, 64);
    load_weights("conv1d_2_weights.bin", conv2_w, 3*64*32);
    load_weights("conv1d_2_bias.bin",    conv2_b, 32);
    load_weights("conv1d_3_weights.bin", conv3_w, 3*32*16);
    load_weights("conv1d_3_bias.bin",    conv3_b, 16);
    load_weights("conv1d_4_weights.bin", conv4_w, 3*16*8);
    load_weights("conv1d_4_bias.bin",    conv4_b, 8);
    load_weights("dense_weights.bin",    dense0_w, 8*512);
    load_weights("dense_bias.bin",       dense0_b, 512);
    load_weights("dense_1_weights.bin",  dense1_w, 512*256);
    load_weights("dense_1_bias.bin",     dense1_b, 256);
    load_weights("dense_2_weights.bin",  dense2_w, 256*128);
    load_weights("dense_2_bias.bin",     dense2_b, 128);
    load_weights("dense_3_weights.bin",  dense3_w, 128*23);
    load_weights("dense_3_bias.bin",     dense3_b, 23);
    std::cout << "Weights loaded successfully\n";
}


//-----------------------------------------------------------------------------
// CSV row data
//-----------------------------------------------------------------------------
struct Row {
    float emg[N_CHANNELS];
    int   restimulus;
};

//-----------------------------------------------------------------------------
// Helper: try to parse a float. Returns true on success.
//-----------------------------------------------------------------------------
static bool try_parse_float(const std::string& s, float& out) {
    if (s.empty()) return false;
    const char* start = s.c_str();
    char* end = nullptr;
    out = std::strtof(start, &end);
    // If strtof didn't advance past start, or value is inf/nan, reject
    return end != start;
}


static bool try_parse_int(const std::string& s, int& out) {
    if (s.empty()) return false;
    const char* start = s.c_str();
    char* end = nullptr;
    long v = std::strtol(start, &end, 10);
    if (end == start) return false;
    out = (int)v;
    return true;
}

//-----------------------------------------------------------------------------
// load_csv_filtered
//-----------------------------------------------------------------------------
std::vector<Row> load_csv_filtered(const char* path) {
    std::vector<Row> rows;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: could not open CSV: " << path << "\n";
        return rows;
    }

    std::string line;
    // Try to detect and skip a header row
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        float dummy;
        if (!try_parse_float(cell, dummy)) {
            // First cell wasn't numeric — row was a header, continue past it
        } else {
            // First cell was numeric — row was data, rewind
            file.clear();
            file.seekg(0, std::ios::beg);
        }
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        Row r;
        r.restimulus = 0;
        bool valid = true;

        while (std::getline(ss, cell, ',')) {
            if (col >= EMG_FIRST_COL && col <= EMG_LAST_COL) {
                float v;
                if (!try_parse_float(cell, v)) { valid = false; break; }
                r.emg[col - EMG_FIRST_COL] = v;
            } else if (col == LABEL_COL) {
                int v;
                if (!try_parse_int(cell, v)) { valid = false; break; }
                r.restimulus = v;
            }
            col++;
        }

        if (valid && r.restimulus > 0) {
            rows.push_back(r);
        }
    }
    file.close();
    return rows;
}

//-----------------------------------------------------------------------------
// find_pure_window — first 52-sample span where all samples share a label
//-----------------------------------------------------------------------------
int find_pure_window(const std::vector<Row>& rows, int start_hint) {
    int n = (int)rows.size();
    for (int i = start_hint; i + N_SAMPLES <= n; i++) {
        int label = rows[i].restimulus;
        bool pure = true;
        for (int j = 1; j < N_SAMPLES; j++) {
            if (rows[i + j].restimulus != label) { pure = false; break; }
        }
        if (pure) return i;
    }
    return -1;
}

int main() {
    // Step 1: Load weights
    read_bin_files();

    // Step 2: Load CSV, keep only non-rest samples
    std::cout << "Loading CSV...\n";
    std::vector<Row> rows = load_csv_filtered(CSV_PATH);
    std::cout << "Loaded " << rows.size() << " non-rest samples\n";
    if ((int)rows.size() < N_SAMPLES) {
        std::cerr << "ERROR: not enough samples — check that "
                  << CSV_PATH << " is in the csim working directory\n";
        return 1;
    }

    // Step 3: Pick a clean 52-sample window
    int start = find_pure_window(rows, 0);
    if (start < 0) {
        std::cerr << "ERROR: no pure 52-sample window found\n";
        return 1;
    }
    int expected_ninapro_label = rows[start].restimulus;
    std::cout << "Window starts at filtered index " << start << "\n";
    std::cout << "Expected NinaPro label: " << expected_ninapro_label << "\n";

    // Step 4: Feed the window into the streams
    hls::stream<float> in_stream[N_CHANNELS];
    hls::stream<int>   out_stream;
    for (int i = 0; i < N_SAMPLES; i++) {
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            in_stream[ch].write(rows[start + i].emg[ch]);
        }
    }

    // Step 5: Run pipeline 52 times — CNN fires on the last call
    float debug_logits[N_CLASSES] = {0};
    for (int i = 0; i < N_SAMPLES; i++) {
        group1_top(in_stream, out_stream,
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
    }

    // Step 6: Read prediction
    if (out_stream.empty()) {
        std::cerr << "ERROR: no prediction produced\n";
        return 1;
    }
    int predicted_class = out_stream.read();
    int predicted_ninapro_label = predicted_class + 1;

    // Step 7: Diagnostic — print raw logits
    std::cout << "\nRaw logits: ";
    for (int i = 0; i < N_CLASSES; i++) {
        std::cout << debug_logits[i] << " ";
    }
    std::cout << "\n";

    // Step 8: Softmax + reporting
    float probs[N_CLASSES];
    softmax(debug_logits, probs);
    float confidence = probs[predicted_class] * 100.0f;

    std::cout << "\nModel output class: " << predicted_class << "\n";
    std::cout << "Predicted NinaPro label: " << predicted_ninapro_label << "\n";
    std::cout << "Confidence: " << confidence << "%\n";
    std::cout << "Expected NinaPro label:  " << expected_ninapro_label << "\n";
    std::cout << (predicted_ninapro_label == expected_ninapro_label
                  ? "MATCH" : "MISMATCH") << "\n";

    std::cout << "\nTop 3 predictions:\n";
    for (int rank = 0; rank < 3; rank++) {
        int best = 0;
        for (int i = 1; i < N_CLASSES; i++) {
            if (probs[i] > probs[best]) best = i;
        }
        std::cout << "  " << (rank + 1) << ". class " << best
                  << " (NinaPro " << (best + 1) << "): "
                  << (probs[best] * 100.0f) << "%\n";
        probs[best] = -1.0f;
    }

    return 0;
}