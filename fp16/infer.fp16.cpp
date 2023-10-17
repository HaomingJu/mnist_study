
#include <array>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <NvInfer.h>
#include "DebugLogger.h"
#include <fstream>
#include <vector>
#include "mnist-reader.h"
#include <assert.h>

int main(int argc, char* argv[]) {

    const char* trt_file = argv[1];
    const char* data_path = argv[2];

    DebugLogger logger("Infer");

    // 1. 引擎准备
    // 1.1 创建IRuntime*
    // -> nvinfer1::IRuntime*
    auto runtime = nvinfer1::createInferRuntime(logger);

    // 1.2 读取engine文件
    std::ifstream trt_handle(trt_file, std::ios::ate | std::ios::binary);
    std::streamsize size = trt_handle.tellg();
    trt_handle.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    trt_handle.read(buffer.data(), size);
    if(size == 0){
        printf("Unable to read engine file");
        return 0;
    }
    trt_handle.close();
    printf("buffer.size:%ld\n", buffer.size());

    // 1.3 反序列化trt文件, 创建ICudaEngine*
    // -> nvinfer1::ICudaEngine*
    auto engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size());

    // 1.4 对engine进行验证，打印调试信息
    int32_t num_io_tensors = engine->getNbIOTensors();
    for(int32_t index = 0; index < num_io_tensors; ++index){
        std::string tensor_name = engine->getIOTensorName(index);
        auto data_type = engine->getTensorDataType(tensor_name.c_str());
        auto io_mode = engine->getTensorIOMode(tensor_name.c_str());
        auto format = engine->getTensorFormat(tensor_name.c_str());

        printf("--- Tensor Name: %s\tData Type: %d\tIO Mode: %d\tFormat: %d\t\n", 
                tensor_name.c_str(), data_type, io_mode, format);
    }

    // 1.5 获取输入和输出张量的尺寸-字节数
    int32_t bytes_input_tensor = 1, bytes_output_tensor = 1;
    auto shape_in = engine->getTensorShape("Input3");
    auto shape_out = engine->getTensorShape("Plus214_Output_0");
    for(int i = 0; i < shape_in.nbDims; ++i){
        bytes_input_tensor *= shape_in.d[i];
    }
    for(int i = 0; i < shape_out.nbDims; ++i){
        bytes_output_tensor *= shape_out.d[i];
    }
    // 因为input tensort和output tensor的数据类型都是kFLOAT - 4Bytes
    bytes_input_tensor *= 4;
    bytes_output_tensor *= 4;
    printf("--- bytes_input_tensor:%d, bytes_output_tensor:%d\n", bytes_input_tensor, bytes_output_tensor);

    // 2. 申请GPU内存
    // 2.1 构建一个stream
    cudaStream_t mStream;
    cudaStreamCreate(&mStream);

    // 2.2 申请GPU内存
    void* input_buffer_gpu = nullptr;
    void* output_buffer_gpu = nullptr;
    cudaError_t err = cudaMallocAsync(&input_buffer_gpu, bytes_input_tensor, mStream);
    if(err != cudaError_t::cudaSuccess) {
        printf("--- INPUT cudaMallocAsync failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    err = cudaMallocAsync(&output_buffer_gpu, bytes_output_tensor, mStream);
    if(err != cudaError_t::cudaSuccess) {
        printf("--- OUTPUT cudaMallocAsync failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    cudaStreamSynchronize(mStream);
    printf("--- input_buffer_gpu:%p, output_buffer_gpu:%p\n", input_buffer_gpu, output_buffer_gpu);

    // 2.3 申请普通内存
    void* input_buffer_host = new char[bytes_input_tensor];
    void* output_buffer_host = new char[bytes_output_tensor];

    // 2.4 准备数据
    std::string label_file = data_path + std::string("/t10k-labels-idx1-ubyte");
    std::string image_file = data_path + std::string("/t10k-images-idx3-ubyte");

    MnistReader mnist_data(label_file.c_str(),image_file.c_str());


    auto context = engine->createExecutionContext();
    context->setInputShape("Input3", shape_in);
    context->setTensorAddress("Input3", input_buffer_gpu);
    context->setTensorAddress("Plus214_Output_0", output_buffer_gpu);

        cudaStream_t inferStream;
        cudaStreamCreate(&inferStream);

    for(int i = 0; i < 10; ++i) {

        MnistReader::DataInfo info = mnist_data[i];

        cudaMemcpyAsync(input_buffer_gpu, info.data_ptr, bytes_input_tensor, cudaMemcpyHostToDevice, inferStream);

        bool status = context->enqueueV3(inferStream);
        if (!status) {
            printf("------- enqueueV3 failed\n");
            return 0;
        }
        cudaMemcpyAsync(output_buffer_host, output_buffer_gpu, bytes_output_tensor, cudaMemcpyDeviceToHost, inferStream);
        cudaStreamSynchronize(inferStream);

        // Softmax 激活函数
        float* output_value = static_cast<float*>(output_buffer_host);
        std::array<float, 10> softmax_value;
        printf("--------------------------------[%d]----------------------\n", info.label);
        float sum = 0.0f;
        for(int i = 0; i < 10; ++i){
            float exp_value = std::exp(output_value[i]);
            sum += exp_value;
            softmax_value[i] = exp_value;
        }
        for(int i = 0; i < 10;  ++i) {
            softmax_value[i] = softmax_value[i] / sum;
        }

        for(int i = 0; i < 10; ++i) {
            const int sz = int(std::round(softmax_value[i] * 100));
            printf("[%d]: ", i);
            for(int j = 0; j < sz; ++j) {
                printf("*");
            }
            printf("\n");
        }

    }

    
    cudaFreeAsync(input_buffer_gpu, mStream);
    cudaFreeAsync(output_buffer_gpu, mStream);
    cudaStreamDestroy(mStream);
    cudaStreamDestroy(inferStream);

    delete[] input_buffer_host;
    input_buffer_host = nullptr;
    delete[] output_buffer_host;
    output_buffer_host = nullptr;
    return 0;
}
