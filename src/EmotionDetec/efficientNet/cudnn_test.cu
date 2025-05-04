#include <iostream>
#include <cudnn.h>

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << std::endl; \
        return 1; \
    }

int main() {
    cudnnHandle_t cudnn;
    cudnnStatus_t status = cudnnCreate(&cudnn);
    CHECK_CUDNN(status);

    // Define tensor dimensions: (batch_size, channels, height, width)
    int batchSize = 1, channels = 1, height = 5, width = 5;
    int filterHeight = 3, filterWidth = 3;

    float input[batchSize][channels][height][width] = {{{{1, 2, 3, 4, 5},
                                                          {6, 7, 8, 9, 10},
                                                          {11, 12, 13, 14, 15},
                                                          {16, 17, 18, 19, 20},
                                                          {21, 22, 23, 24, 25}}}};

    float filter[channels][1][filterHeight][filterWidth] = {{{{1, 0, -1},
                                                              {1, 0, -1},
                                                              {1, 0, -1}}}};

    float output[batchSize][1][height - filterHeight + 1][width - filterWidth + 1];  // Output dimensions

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    // Create tensor descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, channels, height, width));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, 1, height - filterHeight + 1, width - filterWidth + 1));

    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, channels, 1, filterHeight, filterWidth));

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    float alpha = 1.0f, beta = 0.0f;

    // Perform convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, input_desc, &input, filter_desc, &filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_desc, &output));

    std::cout << "Convolution output: " << std::endl;
    for (int i = 0; i < height - filterHeight + 1; ++i) {
        for (int j = 0; j < width - filterWidth + 1; ++j) {
            std::cout << output[0][0][i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudnnDestroy(cudnn);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return 0;
}
