//=============================================================================
// Project:     EE6900 FPGA Accelerator Design - Group 1
// File:        cnn.cpp
// Description: 1D Convolutional Neural Network (1D-CNN) forward pass for
//              real-time EMG hand gesture classification.
//
//              Tiled implementation: weights are loaded from DRAM into
//              small reusable BRAM buffers in tiles to fit within the
//              xc7z020's limited BRAM capacity.
//
//              Architecture (matches pretrained Keras model):
//                - 5x Conv1D layers with ReLU activation
//                - 5x MaxPool1D layers (pool size 2, stride 2)
//                - 4x Dense (fully connected) layers
//                - Final argmax to select predicted gesture class
//
//              Input:  Normalized EMG window of shape (52, 10)
//              Output: Integer gesture class index (0 to 22)
//=============================================================================

#include "cnn.h"

//-----------------------------------------------------------------------------
// Tile sizes for weight buffering
//
// CONV_TILE: number of output filters processed per tile in conv layers
//   Largest conv weight tile: KERNEL * C1_IN * CONV_TILE = 3*128*8 = 3072
//   At 32 bits each = 12 KB per tile — fits easily in BRAM
//
// DENSE_TILE: number of output neurons processed per tile in dense layers
//   Largest dense weight tile: D1_IN * DENSE_TILE = 512*32 = 16384
//   At 32 bits each = 64 KB per tile — fits in BRAM
//-----------------------------------------------------------------------------
#define CONV_TILE   8
#define DENSE_TILE  32

//-----------------------------------------------------------------------------
// relu
//-----------------------------------------------------------------------------
static inline data_t relu(data_t x) {
    return x > (data_t)0 ? x : (data_t)0;
}

//-----------------------------------------------------------------------------
// conv1d_layer_tiled
//
// Tiled 1D convolution: processes CONV_TILE output filters at a time.
// For each tile, loads weights from DRAM into a local BRAM buffer,
// computes the convolution for those filters, and writes final results
// (with ReLU) directly to the output activation buffer.
//
// Parameters:
//   input       — flattened input activations [in_time x in_ch] (BRAM)
//   in_time     — number of time steps
//   in_ch       — number of input channels
//   output      — flattened output activations [in_time x out_ch] (BRAM)
//   out_ch      — total number of output filters
//   kernel      — full weight array [kernel_size x in_ch x out_ch] (DRAM)
//   kernel_size — convolution kernel size
//   bias        — bias array [out_ch] (DRAM)
//-----------------------------------------------------------------------------
void conv1d_layer_tiled(
    data_t input[], int in_time, int in_ch,
    data_t output[], int out_ch,
    weight_t kernel[], int kernel_size,
    weight_t bias[]
) {
    #pragma HLS INLINE off

    // Local BRAM buffers for one tile of weights
    // Sized for the largest conv layer: 3 * 128 * CONV_TILE = 3072
    weight_t w_buf[KERNEL * C1_IN * CONV_TILE];
    weight_t b_buf[CONV_TILE];

    // Iterate over output filter groups
    CONV_TILE_LOOP:
    for (int fg = 0; fg < out_ch; fg += CONV_TILE) {

        // Determine how many filters in this tile
        int tile_size = (fg + CONV_TILE <= out_ch) ? CONV_TILE : (out_ch - fg);

        //---------------------------------------------------------------------
        // Load weight tile from DRAM into BRAM
        // Remap from full kernel layout [k][in_ch][out_ch]
        // to tile layout [k][in_ch][CONV_TILE]
        //---------------------------------------------------------------------
        LOAD_CONV_W:
        for (int k = 0; k < kernel_size; k++) {
            for (int c = 0; c < in_ch; c++) {
                for (int f = 0; f < tile_size; f++) {
                    #pragma HLS PIPELINE
                    w_buf[k * in_ch * CONV_TILE + c * CONV_TILE + f] =
                        kernel[k * in_ch * out_ch + c * out_ch + fg + f];
                }
            }
        }

        // Load bias tile
        LOAD_CONV_B:
        for (int f = 0; f < tile_size; f++) {
            #pragma HLS PIPELINE
            b_buf[f] = bias[fg + f];
        }

        //---------------------------------------------------------------------
        // Compute convolution for this tile of filters
        //---------------------------------------------------------------------
        CONV_TIME:
        for (int t = 0; t < in_time; t++) {
            CONV_FILTER:
            for (int f = 0; f < tile_size; f++) {
                #pragma HLS PIPELINE

                acc_t acc = (acc_t)b_buf[f];

                CONV_KERNEL:
                for (int k = 0; k < kernel_size; k++) {
                    int t_in = t + k - kernel_size / 2;

                    if (t_in >= 0 && t_in < in_time) {
                        CONV_CHANNEL:
                        for (int c = 0; c < in_ch; c++) {
                            acc += (acc_t)input[t_in * in_ch + c] *
                                   (acc_t)w_buf[k * in_ch * CONV_TILE + c * CONV_TILE + f];
                        }
                    }
                }

                // Write final result with ReLU directly to output
                output[t * out_ch + fg + f] = relu((data_t)acc);
            }
        }
    }
}

//-----------------------------------------------------------------------------
// maxpool1d_layer
//
// 1D max pooling with pool size 2 and stride 2.
//-----------------------------------------------------------------------------
void maxpool1d_layer(
    data_t input[], int in_time,
    data_t output[], int ch
) {
    #pragma HLS INLINE off

    int out_time = in_time / 2;

    MAXPOOL:
    for (int t = 0; t < out_time; t++) {
        MAXPOOL_CALC:
        for (int c = 0; c < ch; c++) {
            #pragma HLS PIPELINE

            data_t a = input[(t*2)   * ch + c];
            data_t b = input[(t*2+1) * ch + c];

            output[t * ch + c] = (a > b) ? a : b;
        }
    }
}

//-----------------------------------------------------------------------------
// dense_layer_tiled
//
// Tiled dense layer: processes DENSE_TILE output neurons at a time.
// For each tile, loads weights from DRAM into a local BRAM buffer,
// computes the full dot product for those neurons, applies ReLU if
// requested, and writes final results to the output array.
//
// Parameters:
//   input      — input activations [in_size] (BRAM)
//   in_size    — number of input neurons
//   output     — output activations [out_size] (BRAM)
//   out_size   — total number of output neurons
//   weights    — full weight matrix [in_size x out_size] (DRAM)
//   bias       — bias array [out_size] (DRAM)
//   apply_relu — apply ReLU activation after linear transform
//-----------------------------------------------------------------------------
void dense_layer_tiled(
    data_t input[], int in_size,
    data_t output[], int out_size,
    weight_t weights[], weight_t bias[],
    bool apply_relu
) {
    #pragma HLS INLINE off

    // Local BRAM buffers for one tile of weights
    // Sized for the largest dense layer: D1_IN * DENSE_TILE = 512*32 = 16384
    weight_t w_buf[D1_IN * DENSE_TILE];
    weight_t b_buf[DENSE_TILE];

    // Iterate over output neuron groups
    DENSE_TILE_LOOP:
    for (int og = 0; og < out_size; og += DENSE_TILE) {

        // Determine how many neurons in this tile
        int tile_size = (og + DENSE_TILE <= out_size) ? DENSE_TILE : (out_size - og);

        //---------------------------------------------------------------------
        // Load weight tile from DRAM into BRAM
        // Remap from [in_size x out_size] to [in_size x DENSE_TILE]
        //---------------------------------------------------------------------
        LOAD_DENSE_W:
        for (int i = 0; i < in_size; i++) {
            for (int o = 0; o < tile_size; o++) {
                #pragma HLS PIPELINE
                w_buf[i * DENSE_TILE + o] = weights[i * out_size + og + o];
            }
        }

        // Load bias tile
        LOAD_DENSE_B:
        for (int o = 0; o < tile_size; o++) {
            #pragma HLS PIPELINE
            b_buf[o] = bias[og + o];
        }

        //---------------------------------------------------------------------
        // Compute dot product for this tile of output neurons
        //---------------------------------------------------------------------
        DENSE_COMPUTE:
        for (int o = 0; o < tile_size; o++) {
            #pragma HLS PIPELINE

            acc_t acc = (acc_t)b_buf[o];

            DENSE_DOT:
            for (int i = 0; i < in_size; i++) {
                acc += (acc_t)input[i] * (acc_t)w_buf[i * DENSE_TILE + o];
            }

            // Write final result to correct position in output
            output[og + o] = apply_relu ? relu((data_t)acc) : (data_t)acc;
        }
    }
}

//-----------------------------------------------------------------------------
// argmax
//-----------------------------------------------------------------------------
int argmax(data_t input[], int size) {
    #pragma HLS INLINE off

    int best_idx    = 0;
    data_t best_val = input[0];

    ARG_MAX:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE
        if (input[i] > best_val) {
            best_val = input[i];
            best_idx = i;
        }
    }
    return best_idx;
}

//-----------------------------------------------------------------------------
// cnn_forward
//
// Full 1D-CNN forward pass using tiled weight loading.
// Weights are loaded from DRAM in small tiles — no need to buffer
// all weights in BRAM simultaneously.
//
// Activation buffers between layers remain fully on-chip in BRAM.
//
// Data flow:
//   Input (52,10) → flatten → 520
//   Conv0+Pool0: (52,10)  → (52,128)  → (26,128)
//   Conv1+Pool1: (26,128) → (26,64)   → (13,64)
//   Conv2+Pool2: (13,64)  → (13,32)   → (6,32)
//   Conv3+Pool3: (6,32)   → (6,16)    → (3,16)
//   Conv4+Pool4: (3,16)   → (3,8)     → (1,8)
//   Dense0: 8   → 512
//   Dense1: 512 → 256
//   Dense2: 256 → 128
//   Dense3: 128 → 23
//   Argmax: 23  → 1
//-----------------------------------------------------------------------------
int cnn_forward(
    data_t input[WINDOW_SIZE][N_CHANNELS],
    weight_t conv0_w[], weight_t conv0_b[],
    weight_t conv1_w[], weight_t conv1_b[],
    weight_t conv2_w[], weight_t conv2_b[],
    weight_t conv3_w[], weight_t conv3_b[],
    weight_t conv4_w[], weight_t conv4_b[],
    weight_t dense0_w[], weight_t dense0_b[],
    weight_t dense1_w[], weight_t dense1_b[],
    weight_t dense2_w[], weight_t dense2_b[],
    weight_t dense3_w[], weight_t dense3_b[]
) {
    //-------------------------------------------------------------------------
    // Flatten 2D input into 1D
    //-------------------------------------------------------------------------
    data_t flat[WINDOW_SIZE * N_CHANNELS];

    FLATTEN_INPUT:
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int c = 0; c < N_CHANNELS; c++) {
            #pragma HLS PIPELINE
            flat[t * N_CHANNELS + c] = input[t][c];
        }
    }

    //-------------------------------------------------------------------------
    // Conv block 0: (52,10) → conv → (52,128) → pool → (26,128)
    //-------------------------------------------------------------------------
    data_t c0[WINDOW_SIZE * C0_OUT];
    data_t p0[26 * C0_OUT];

    conv1d_layer_tiled(flat, WINDOW_SIZE, C0_IN, c0, C0_OUT, conv0_w, KERNEL, conv0_b);
    maxpool1d_layer(c0, WINDOW_SIZE, p0, C0_OUT);

    //-------------------------------------------------------------------------
    // Conv block 1: (26,128) → conv → (26,64) → pool → (13,64)
    //-------------------------------------------------------------------------
    data_t c1[26 * C1_OUT];
    data_t p1[13 * C1_OUT];

    conv1d_layer_tiled(p0, 26, C1_IN, c1, C1_OUT, conv1_w, KERNEL, conv1_b);
    maxpool1d_layer(c1, 26, p1, C1_OUT);

    //-------------------------------------------------------------------------
    // Conv block 2: (13,64) → conv → (13,32) → pool → (6,32)
    //-------------------------------------------------------------------------
    data_t c2[13 * C2_OUT];
    data_t p2[6 * C2_OUT];

    conv1d_layer_tiled(p1, 13, C2_IN, c2, C2_OUT, conv2_w, KERNEL, conv2_b);
    maxpool1d_layer(c2, 13, p2, C2_OUT);

    //-------------------------------------------------------------------------
    // Conv block 3: (6,32) → conv → (6,16) → pool → (3,16)
    //-------------------------------------------------------------------------
    data_t c3[6 * C3_OUT];
    data_t p3[3 * C3_OUT];

    conv1d_layer_tiled(p2, 6, C3_IN, c3, C3_OUT, conv3_w, KERNEL, conv3_b);
    maxpool1d_layer(c3, 6, p3, C3_OUT);

    //-------------------------------------------------------------------------
    // Conv block 4: (3,16) → conv → (3,8) → pool → (1,8)
    //-------------------------------------------------------------------------
    data_t c4[3 * C4_OUT];
    data_t p4[1 * C4_OUT];

    conv1d_layer_tiled(p3, 3, C4_IN, c4, C4_OUT, conv4_w, KERNEL, conv4_b);
    maxpool1d_layer(c4, 3, p4, C4_OUT);

    //-------------------------------------------------------------------------
    // Dense layers (tiled weight loading)
    //-------------------------------------------------------------------------
    data_t d0[D0_OUT], d1[D1_OUT], d2[D2_OUT], d3[D3_OUT];

    dense_layer_tiled(p4, D0_IN, d0, D0_OUT, dense0_w, dense0_b, true);
    dense_layer_tiled(d0, D1_IN, d1, D1_OUT, dense1_w, dense1_b, true);
    dense_layer_tiled(d1, D2_IN, d2, D2_OUT, dense2_w, dense2_b, true);
    dense_layer_tiled(d2, D3_IN, d3, D3_OUT, dense3_w, dense3_b, false);

    //-------------------------------------------------------------------------
    // Return predicted gesture class
    //-------------------------------------------------------------------------
    return argmax(d3, D3_OUT);
}