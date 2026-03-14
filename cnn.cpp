//=============================================================================
// Project:     EE6900 FPGA Accelerator Design - Group 1
// File:        cnn.cpp
// Description: 1D Convolutional Neural Network (1D-CNN) forward pass for
//              real-time EMG hand gesture classification.
//
//              Architecture (matches pretrained Keras model):
//                - 5x Conv1D layers with ReLU activation
//                - 5x MaxPool1D layers (pool size 2, stride 2)
//                - 4x Dense (fully connected) layers
//                - Final argmax to select predicted gesture class
//
//              Weights are pretrained in TensorFlow/Keras on the NinaPro DB1
//              dataset and passed in as float arrays loaded from .bin files.
//
//              Input:  Normalized EMG window of shape (52, 10)
//                      52 time steps x 10 electrode channels
//              Output: Integer gesture class index (0 to 22)
//                      representing one of 23 hand gestures
//=============================================================================

#include "cnn.h"

//-----------------------------------------------------------------------------
// relu
//
// Rectified Linear Unit activation function.
// Returns the input value if positive, otherwise returns 0.
// Used after each convolutional and dense layer (except the final output layer).
//
// Parameters:
//   x — input value
//
// Returns:
//   x if x > 0, else 0
//-----------------------------------------------------------------------------
static inline data_t relu(data_t x) {
    return x > (data_t)0 ? x : (data_t)0;
}

//-----------------------------------------------------------------------------
// conv1d_layer
//
// Performs a 1D convolution on a time-series input with same padding.
// Same padding means the output time dimension matches the input time dimension
// by zero-padding the edges of the input.
//
// The convolution slides a kernel of size kernel_size along the time axis,
// computing a dot product across all input channels at each time step.
// A bias is added and ReLU activation is applied to the result.
//
// Parameters:
//   input       — flattened input array of shape (in_time x in_ch)
//   in_time     — number of time steps in the input
//   in_ch       — number of input channels (features per time step)
//   output      — flattened output array of shape (in_time x out_ch)
//   out_ch      — number of output filters (feature maps)
//   kernel      — flattened weight array of shape (kernel_size x in_ch x out_ch)
//   kernel_size — size of the 1D convolution kernel (number of time steps it spans)
//   bias        — bias array of shape (out_ch)
//-----------------------------------------------------------------------------
void conv1d_layer(
    data_t input[], int in_time, int in_ch,
    data_t output[], int out_ch,
    float kernel[], int kernel_size,
    float bias[]
) {
    // Iterate over each output time step
    for (int t = 0; t < in_time; t++) {

        // Iterate over each output filter (feature map)
        for (int f = 0; f < out_ch; f++) {

            // Initialize accumulator with bias for this filter
            acc_t acc = (acc_t)bias[f];

            // Slide the kernel across the time dimension
            for (int k = 0; k < kernel_size; k++) {

                // Compute the corresponding input time index with same padding
                // kernel_size/2 centers the kernel on the current output time step
                int t_in = t + k - kernel_size / 2;

                // Skip if outside input bounds (zero padding)
                if (t_in >= 0 && t_in < in_time) {

                    // Dot product across all input channels at this time step
                    for (int c = 0; c < in_ch; c++) {
                        // input is stored as [time][channel] flattened to 1D
                        // kernel is stored as [k][in_ch][out_ch] flattened to 1D
                        acc += (acc_t)input[t_in * in_ch + c] *
                               (acc_t)kernel[k * in_ch * out_ch + c * out_ch + f];
                    }
                }
            }

            // Apply ReLU and store result
            // output is stored as [time][filter] flattened to 1D
            output[t * out_ch + f] = relu((data_t)acc);
        }
    }
}

//-----------------------------------------------------------------------------
// maxpool1d_layer
//
// Performs 1D max pooling with pool size 2 and stride 2.
// Takes the maximum of every two consecutive time steps, halving the time
// dimension. This reduces computation and increases receptive field in
// subsequent layers.
//
// Parameters:
//   input   — flattened input array of shape (in_time x ch)
//   in_time — number of time steps in the input (must be even)
//   output  — flattened output array of shape (in_time/2 x ch)
//   ch      — number of channels (same for input and output)
//-----------------------------------------------------------------------------
void maxpool1d_layer(
    data_t input[], int in_time,
    data_t output[], int ch
) {
    // Output time dimension is half the input
    int out_time = in_time / 2;

    for (int t = 0; t < out_time; t++) {
        for (int c = 0; c < ch; c++) {
            // Take the two consecutive time steps for this pool window
            data_t a = input[(t*2)   * ch + c];  // even time step
            data_t b = input[(t*2+1) * ch + c];  // odd time step

            // Store the maximum of the two values
            output[t * ch + c] = (a > b) ? a : b;
        }
    }
}

//-----------------------------------------------------------------------------
// dense_layer
//
// Fully connected (dense) layer — every input neuron connects to every
// output neuron via a learned weight. Computes a matrix-vector multiply
// followed by bias addition and optional ReLU activation.
//
// Parameters:
//   input      — input array of shape (in_size)
//   in_size    — number of input neurons
//   output     — output array of shape (out_size)
//   out_size   — number of output neurons
//   weights    — weight matrix of shape (in_size x out_size) flattened to 1D
//                stored in row-major order: weights[i * out_size + o]
//   bias       — bias array of shape (out_size)
//   apply_relu — if true, apply ReLU activation after the linear transform
//                set to false for the final output layer (argmax handles it)
//-----------------------------------------------------------------------------
void dense_layer(
    data_t input[], int in_size,
    data_t output[], int out_size,
    float weights[], float bias[],
    bool apply_relu
) {
    // Iterate over each output neuron
    for (int o = 0; o < out_size; o++) {

        // Initialize accumulator with this neuron's bias
        acc_t acc = (acc_t)bias[o];

        // Dot product of input vector with this neuron's weight row
        for (int i = 0; i < in_size; i++) {
            acc += (acc_t)input[i] * (acc_t)weights[i * out_size + o];
        }

        // Apply ReLU if requested, otherwise store raw logit
        output[o] = apply_relu ? relu((data_t)acc) : (data_t)acc;
    }
}

//-----------------------------------------------------------------------------
// argmax
//
// Returns the index of the highest value in an array.
// Used after the final dense layer to select the predicted gesture class
// from the raw output logits (no need to apply softmax since argmax of
// logits gives the same result as argmax of softmax probabilities).
//
// Parameters:
//   input — array of logit values (one per class)
//   size  — number of classes
//
// Returns:
//   index of the maximum value (predicted gesture class)
//-----------------------------------------------------------------------------
int argmax(data_t input[], int size) {
    int best_idx    = 0;
    data_t best_val = input[0];  // assume first element is best initially

    for (int i = 1; i < size; i++) {
        if (input[i] > best_val) {
            best_val = input[i];  // update best value
            best_idx = i;         // update best index
        }
    }
    return best_idx;
}

//-----------------------------------------------------------------------------
// cnn_forward
//
// Full 1D-CNN forward pass for EMG gesture classification.
// Runs the input EMG window through all 5 convolutional blocks and
// 4 dense layers, returning the predicted gesture class index.
//
// The data flow through the network is:
//   Input (52,10) → flatten → 520 values
//   Conv1 + Pool1: 520 → (52,128) → (26,128)
//   Conv2 + Pool2: (26,128) → (26,64) → (13,64)
//   Conv3 + Pool3: (13,64)  → (13,32) → (6,32)
//   Conv4 + Pool4: (6,32)   → (6,16)  → (3,16)
//   Conv5 + Pool5: (3,16)   → (3,8)   → (1,8) = 8 values
//   Dense0: 8   → 512
//   Dense1: 512 → 256
//   Dense2: 256 → 128
//   Dense3: 128 → 23  (one logit per gesture class)
//   Argmax: 23  → 1   (predicted gesture index)
//
// Parameters:
//   input    — normalized EMG window of shape (WINDOW_SIZE x N_CHANNELS)
//   conv0-4  — pretrained conv layer weights and biases
//   dense0-3 — pretrained dense layer weights and biases
//
// Returns:
//   Predicted gesture class index (0 to N_CLASSES-1)
//-----------------------------------------------------------------------------
int cnn_forward(
    data_t input[WINDOW_SIZE][N_CHANNELS],
    float conv0_w[], float conv0_b[],
    float conv1_w[], float conv1_b[],
    float conv2_w[], float conv2_b[],
    float conv3_w[], float conv3_b[],
    float conv4_w[], float conv4_b[],
    float dense0_w[], float dense0_b[],
    float dense1_w[], float dense1_b[],
    float dense2_w[], float dense2_b[],
    float dense3_w[], float dense3_b[]
) {
    //-------------------------------------------------------------------------
    // Flatten 2D input (WINDOW_SIZE x N_CHANNELS) into 1D array
    // Conv1d_layer expects a flat array indexed as [time * n_channels + channel]
    //-------------------------------------------------------------------------
    data_t flat[WINDOW_SIZE * N_CHANNELS];
    for (int t = 0; t < WINDOW_SIZE; t++)
        for (int c = 0; c < N_CHANNELS; c++)
            flat[t * N_CHANNELS + c] = input[t][c];

    //-------------------------------------------------------------------------
    // Conv block 1
    // Input:  flat (52 x 10) = 520 values
    // After conv:  (52 x 128) — 128 feature maps each of length 52
    // After pool:  (26 x 128) — halved by max pooling
    //-------------------------------------------------------------------------
    data_t c0[WINDOW_SIZE * C0_OUT];  // conv output buffer
    data_t p0[26 * C0_OUT];           // pool output buffer
    conv1d_layer(flat, WINDOW_SIZE, C0_IN, c0, C0_OUT, conv0_w, KERNEL, conv0_b);
    maxpool1d_layer(c0, WINDOW_SIZE, p0, C0_OUT);

    //-------------------------------------------------------------------------
    // Conv block 2
    // Input:  (26 x 128)
    // After conv:  (26 x 64)
    // After pool:  (13 x 64)
    //-------------------------------------------------------------------------
    data_t c1[26 * C1_OUT];
    data_t p1[13 * C1_OUT];
    conv1d_layer(p0, 26, C1_IN, c1, C1_OUT, conv1_w, KERNEL, conv1_b);
    maxpool1d_layer(c1, 26, p1, C1_OUT);

    //-------------------------------------------------------------------------
    // Conv block 3
    // Input:  (13 x 64)
    // After conv:  (13 x 32)
    // After pool:  (6 x 32)  — note: 13/2 = 6 (truncated)
    //-------------------------------------------------------------------------
    data_t c2[13 * C2_OUT];
    data_t p2[6 * C2_OUT];
    conv1d_layer(p1, 13, C2_IN, c2, C2_OUT, conv2_w, KERNEL, conv2_b);
    maxpool1d_layer(c2, 13, p2, C2_OUT);

    //-------------------------------------------------------------------------
    // Conv block 4
    // Input:  (6 x 32)
    // After conv:  (6 x 16)
    // After pool:  (3 x 16)
    //-------------------------------------------------------------------------
    data_t c3[6 * C3_OUT];
    data_t p3[3 * C3_OUT];
    conv1d_layer(p2, 6, C3_IN, c3, C3_OUT, conv3_w, KERNEL, conv3_b);
    maxpool1d_layer(c3, 6, p3, C3_OUT);

    //-------------------------------------------------------------------------
    // Conv block 5
    // Input:  (3 x 16)
    // After conv:  (3 x 8)
    // After pool:  (1 x 8) = 8 values — the compressed feature vector
    //-------------------------------------------------------------------------
    data_t c4[3 * C4_OUT];
    data_t p4[1 * C4_OUT];
    conv1d_layer(p3, 3, C4_IN, c4, C4_OUT, conv4_w, KERNEL, conv4_b);
    maxpool1d_layer(c4, 3, p4, C4_OUT);

    //-------------------------------------------------------------------------
    // Dense layers
    // Progressively transform 8 compressed features into class probabilities
    // Dense 0: 8   → 512  (expand to learn complex combinations)
    // Dense 1: 512 → 256  (compress)
    // Dense 2: 256 → 128  (compress further)
    // Dense 3: 128 → 23   (one logit per gesture class, no ReLU)
    //-------------------------------------------------------------------------
    data_t d0[D0_OUT], d1[D1_OUT], d2[D2_OUT], d3[D3_OUT];

    dense_layer(p4, D0_IN, d0, D0_OUT, dense0_w, dense0_b, true);   // ReLU
    dense_layer(d0, D1_IN, d1, D1_OUT, dense1_w, dense1_b, true);   // ReLU
    dense_layer(d1, D2_IN, d2, D2_OUT, dense2_w, dense2_b, true);   // ReLU
    dense_layer(d2, D3_IN, d3, D3_OUT, dense3_w, dense3_b, false);  // no ReLU — raw logits

    //-------------------------------------------------------------------------
    // Return the index of the highest logit as the predicted gesture class
    // Equivalent to applying softmax then taking argmax, but faster
    //-------------------------------------------------------------------------
    return argmax(d3, D3_OUT);
}
