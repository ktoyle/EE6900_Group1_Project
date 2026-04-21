//=============================================================================
// Project:     EE6900 FPGA Accelerator Design - Group 1 (resource optimized)
// File:        cnn.cpp
// Description: 1D Convolutional Neural Network (1D-CNN) forward pass for
//              real-time EMG hand gesture classification.
//
//              Single-loop tiled implementation: conv and dense layers each
//              use ONE hardware instance that is reused across all layers.
//              Weights are loaded from DRAM in small tiles per layer.
//              Activations ping-pong between two reusable BRAM buffers.
//
//              Input:  Normalized EMG window of shape (52, 10)
//              Output: Integer gesture class index (0 to 22)
//=============================================================================

#include "cnn.h"
#include <cstring> 

//-----------------------------------------------------------------------------
// Tile sizes for weight buffering
//-----------------------------------------------------------------------------
#define CONV_TILE   16 
#define DENSE_TILE  64 

//-----------------------------------------------------------------------------
// Activation buffer size — must hold the largest intermediate tensor
// Largest is conv0 output: 52 * 128 = 6656 elements
//-----------------------------------------------------------------------------
#define ACT_BUF_SIZE  (WINDOW_SIZE * C0_OUT)  // 6656

//-----------------------------------------------------------------------------
// relu activation
//-----------------------------------------------------------------------------
static inline data_t relu(data_t x) {
    return x > (data_t)0 ? x : (data_t)0;
}

//-----------------------------------------------------------------------------
// conv1d_layer_tiled
//
// Tiled 1D convolution: processes CONV_TILE output filters at a time.
// Single hardware instance reused for all 5 conv layers.

//-----------------------------------------------------------------------------

void conv1d_layer_tiled(
    data_t input[], int in_time, int in_ch,
    data_t output[], int out_ch,
    weight_t kernel[], int kernel_size,
    weight_t bias[]
) {
    //No inline to save on resources and reuse hardware block
    #pragma HLS INLINE off

    // Local BRAM buffers for one tile of weights
    weight_t b_buf[CONV_TILE];
    weight_t w_buf[KERNEL][C1_IN][CONV_TILE];

    #pragma HLS ARRAY_PARTITION variable=w_buf cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=3

    CONV_TILE_LOOP:
    for (int fg = 0; fg < out_ch; fg += CONV_TILE) {

        //HLS LOOP_TRIPCOUNT to tell the compiler how many iterations are in loop at compile time
        //This helps with latency count when iteration is a parameter and unknown at compile time
         #pragma HLS LOOP_TRIPCOUNT min=1 max=8

        int tile_size = (fg + CONV_TILE <= out_ch) ? CONV_TILE : (out_ch - fg);

       //Load weight tile from DRAM into BRAM
       LOAD_CONV_W:
        for (int k = 0; k < kernel_size; k++) {
            
             #pragma HLS LOOP_TRIPCOUNT min=3 max=3    // k

            for (int c = 0; c < in_ch; c++) {

                 #pragma HLS PIPELINE II=1 
                 #pragma HLS LOOP_TRIPCOUNT min=10 max=128 // c
                 
                for (int f = 0; f < tile_size; f++) {
                    
                     #pragma HLS UNROLL
                     #pragma HLS LOOP_TRIPCOUNT min=8 max=8    // f
                   
                    // Load
                    w_buf[k][c][f] = kernel[k * in_ch * out_ch + c * out_ch + fg + f];
                }
            }
        }

        LOAD_CONV_B:
        for (int f = 0; f < tile_size; f++) {

            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT min=16 max=16 // 8 8
            
            b_buf[f] = bias[fg + f];
        }

        // Compute convolution for this tile
        CONV_TIME:
        for (int t = 0; t < in_time; t++) {
            #pragma HLS LOOP_TRIPCOUNT min=3 max=52

            CONV_FILTER:
            for (int f = 0; f < tile_size; f++) {
                #pragma HLS LOOP_TRIPCOUNT min=16 max=16

                acc_t acc = (acc_t)b_buf[f];

                CONV_CHANNEL:
                for (int c = 0; c < in_ch; c++) {
                    #pragma HLS LOOP_TRIPCOUNT min=10 max=128
                    #pragma HLS PIPELINE II=1

                    CONV_KERNEL:
                    for (int k = 0; k < kernel_size; k++) {
                        #pragma HLS UNROLL
                        int t_in = t + k - kernel_size / 2;
                        if (t_in >= 0 && t_in < in_time) {
                            acc += (acc_t)input[t_in * in_ch + c] *
                                (acc_t)w_buf[k][c][f];
                        }
                    }
                }

                output[t * out_ch + fg + f] = relu((data_t)acc);
            }
        }

    }
}

//-----------------------------------------------------------------------------
// maxpool1d_layer
//
// 1D max pooling with pool size 2 and stride 2.
// Single hardware instance reused for all 5 pool layers.
//-----------------------------------------------------------------------------
void maxpool1d_layer(
    data_t input[], int in_time,
    data_t output[], int ch
) {
    #pragma HLS INLINE off

    int out_time = in_time / 2;

    MAXPOOL:
    for (int t = 0; t < out_time; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=26

        MAXPOOL_CALC:
        for (int c = 0; c < ch; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=8 max=128
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
// Single hardware instance reused for all 4 dense layers.
//-----------------------------------------------------------------------------
void dense_layer_tiled(
    data_t input[], int in_size,
    data_t output[], int out_size,
    weight_t weights[], weight_t bias[],
    bool apply_relu
) {
    #pragma HLS INLINE off

    // Local BRAM buffers for one tile of weights
    weight_t w_buf[D1_IN * DENSE_TILE];  // 512*32 = 16384
    weight_t b_buf[DENSE_TILE];

    DENSE_TILE_LOOP:
    for (int og = 0; og < out_size; og += DENSE_TILE) {
         #pragma HLS LOOP_TRIPCOUNT min=1 max=8//16  // max: 512/32 = 16 (dense0)
        int tile_size = (og + DENSE_TILE <= out_size) ? DENSE_TILE : (out_size - og);

        // Load weight tile from DRAM into BRAM
        LOAD_DENSE_W:
        for (int i = 0; i < in_size; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=8 max=512 
            for (int o = 0; o < tile_size; o++) {
                #pragma HLS PIPELINE //no II parameter lets the compiler choose the best option
                #pragma HLS LOOP_TRIPCOUNT min=8 max=512  // D0_IN to D1_IN
                w_buf[i * DENSE_TILE + o] = weights[i * out_size + og + o];
            }
        }

        LOAD_DENSE_B:
        for (int o = 0; o < tile_size; o++) {
            #pragma HLS PIPELINE
             #pragma HLS LOOP_TRIPCOUNT min=23 max=64//32
            b_buf[o] = bias[og + o];
        }

        // Compute dot product for this tile
        DENSE_COMPUTE:
        for (int o = 0; o < tile_size; o++) {

            #pragma HLS LOOP_TRIPCOUNT min=23 max=64//32
            #pragma HLS PIPELINE

            acc_t acc = (acc_t)b_buf[o];

            DENSE_DOT:
            for (int i = 0; i < in_size; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=8 max=512
                acc += (acc_t)input[i] * (acc_t)w_buf[i * DENSE_TILE + o];
            }

            output[og + o] = apply_relu ? relu((data_t)acc) : (data_t)acc;
        }
    }
}

//-----------------------------------------------------------------------------
// argmax activation to choose the most probable gesture
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
// Full 1D-CNN forward pass using a single loop for conv layers and
// a single loop for dense layers. This ensures HLS creates only ONE
// hardware instance of conv1d_layer_tiled and ONE of dense_layer_tiled,
// reused across all layers via time-sharing.
//
// Activations ping-pong between buf_a and buf_b:
//   - Conv reads from buf_a, writes to buf_b
//   - Pool reads from buf_b, writes back to buf_a
//   - Next layer repeats
//
// Data flow:
//   Input (52,10) → flatten into buf_a
//   Conv0: buf_a → buf_b    Pool0: buf_b → buf_a
//   Conv1: buf_a → buf_b    Pool1: buf_b → buf_a
//   Conv2: buf_a → buf_b    Pool2: buf_b → buf_a
//   Conv3: buf_a → buf_b    Pool3: buf_b → buf_a
//   Conv4: buf_a → buf_b    Pool4: buf_b → buf_a
//   Dense0: buf_a → buf_b
//   Dense1: buf_b → buf_a
//   Dense2: buf_a → buf_b
//   Dense3: buf_b → buf_a   (or buf_b, doesn't matter — argmax reads it)
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
    #ifdef CSIM_DEBUG
        , float debug_logits[N_CLASSES]        
    #endif
) {
    //-------------------------------------------------------------------------
    // Reusable activation buffers (ping-pong)
    // Sized for the largest intermediate tensor: 52 * 128 = 6656
    //-------------------------------------------------------------------------
    data_t buf_a[ACT_BUF_SIZE];
    data_t buf_b[ACT_BUF_SIZE];
    #pragma HLS ARRAY_PARTITION variable=buf_a cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=buf_b cyclic factor=16 dim=1

    //-------------------------------------------------------------------------
    // Conv layer configuration tables
    //-------------------------------------------------------------------------
    const int conv_in_time[5]  = {52, 26, 13, 6, 3};
    const int conv_in_ch[5]    = {C0_IN, C1_IN, C2_IN, C3_IN, C4_IN};
    const int conv_out_ch[5]   = {C0_OUT, C1_OUT, C2_OUT, C3_OUT, C4_OUT};

    //-------------------------------------------------------------------------
    // Dense layer configuration tables
    //-------------------------------------------------------------------------
    const int dense_in_size[4]  = {D0_IN, D1_IN, D2_IN, D3_IN};
    const int dense_out_size[4] = {D0_OUT, D1_OUT, D2_OUT, D3_OUT};

    //-------------------------------------------------------------------------
    // Flatten 2D input into buf_a
    //-------------------------------------------------------------------------
    FLATTEN_INPUT:
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int c = 0; c < N_CHANNELS; c++) {
            #pragma HLS PIPELINE
            buf_a[t * N_CHANNELS + c] = input[t][c];
        }
    }

    //-------------------------------------------------------------------------
    // Conv + Pool layers: single loop, one hardware instance each
    // Conv reads buf_a → writes buf_b
    // Pool reads buf_b → writes buf_a
    //-------------------------------------------------------------------------
    CONV_LAYERS:
    for (int layer = 0; layer < 5; layer++) {
        //#pragma HLS UNROLL factor= 4
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        // Select weight pointers for this layer via switch
        weight_t* w;
        weight_t* b;

        //Loop that keeps calculations within the same hardware block to help with resources
        switch (layer) {
            case 0: w = conv0_w; b = conv0_b; break;
            case 1: w = conv1_w; b = conv1_b; break;
            case 2: w = conv2_w; b = conv2_b; break;
            case 3: w = conv3_w; b = conv3_b; break;
            case 4: w = conv4_w; b = conv4_b; break;
        }

        // Conv: buf_a → buf_b
        
        conv1d_layer_tiled(buf_a, conv_in_time[layer], conv_in_ch[layer],
                           buf_b, conv_out_ch[layer],
                           w, KERNEL, b);

        // Pool: buf_b → buf_a (halves the time dimension)
        maxpool1d_layer(buf_b, conv_in_time[layer], buf_a, conv_out_ch[layer]);
    }


    //-------------------------------------------------------------------------
    // Dense layers: single loop, one hardware instance
    // Alternates read/write between buf_a and buf_b
    //   Dense0: buf_a → buf_b   (ReLU)
    //   Dense1: buf_b → buf_a   (ReLU)
    //   Dense2: buf_a → buf_b   (ReLU)
    //   Dense3: buf_b → buf_a   (no ReLU — raw logits)
    //-------------------------------------------------------------------------
    DENSE_LAYERS:
    for (int layer = 0; layer < 4; layer++) {

        weight_t* w;
        weight_t* b;

        switch (layer) {
            case 0: w = dense0_w; b = dense0_b; break;
            case 1: w = dense1_w; b = dense1_b; break;
            case 2: w = dense2_w; b = dense2_b; break;
            case 3: w = dense3_w; b = dense3_b; break;
        }

        bool use_relu = (layer < 3);

        // Even layers: buf_a → buf_b
        // Odd layers:  buf_b → buf_a
        if (layer % 2 == 0) {
            dense_layer_tiled(buf_a, dense_in_size[layer],
                              buf_b, dense_out_size[layer],
                              w, b, use_relu);
        } else {
            dense_layer_tiled(buf_b, dense_in_size[layer],
                              buf_a, dense_out_size[layer],
                              w, b, use_relu);
        }
    }

    //Compute confidence interval for decisions when doing csim
    #ifdef CSIM_DEBUG
        if (debug_logits != nullptr) {
            for (int i = 0; i < D3_OUT; i++) {
                debug_logits[i] = (float)buf_a[i];
            }
        }
    #endif

    //-------------------------------------------------------------------------
    // Argmax — after 4 dense layers (even count), final result is in buf_a
    // Dense0: a→b, Dense1: b→a, Dense2: a→b, Dense3: b→a
    // So buf_a has the 23 logits
    //-------------------------------------------------------------------------
    return argmax(buf_a, D3_OUT);
}



