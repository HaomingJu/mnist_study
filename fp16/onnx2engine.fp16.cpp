
#include <NvInferImpl.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <cstdio>
#include <fstream>

#include <NvInfer.h>
#include "DebugLogger.h"

#include <NvOnnxParser.h>
#include <istream>

/*
 * 将onnx模型文件转为TensorRT中序列化之后的模型文件
 *
 * 涉及需要明确的参数有:
 * 1. 精度：            fp16或者fp8(量化)
 * 2. batchSize:        是否为1
 */

int main(int argc, char* argv[]) {
    const char* onnx_model_file = argv[1];
    const char* trt_model_file = argv[2];

    DebugLogger logger("Onnx2TRT", DebugLogger::Severity::kINFO);

    /* 1. 挨个创建 IBuilder -> INetworkDefinition -> IParser -> IBuilderConfig */

    // -> nvinfer1::IBuilder* 
    auto builder = nvinfer1::createInferBuilder(logger);

    // -> nvinfer1::NetworkDefinitionCreationFlags 
    const auto flags = 1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    // -> nvinfer1::INetworkDefinition* 
    auto network = builder->createNetworkV2(flags);

    // -> nvonnxparser::IParser* 
    auto parser = nvonnxparser::createParser(*network, logger);

    // 加载ONNX模型文件
    bool ret = parser->parseFromFile(onnx_model_file, int(DebugLogger::Severity::kERROR));
    if(!ret){
        logger.log(DebugLogger::Severity::kERROR, "Parse onnx model file failed");
        return 0;
    }


    /* 2. 依据模型特性设置对应的参数，方便进行序列化 */

    // 2.1 构建IBuilder的配置对象IBuilderConfig
    // -> nvinfer1::IBuilderConfig*
    auto b_config = builder->createBuilderConfig();

    // 2.2 模型精度设置为FP16, 未进行量化
    b_config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // 2.3 依据硬件特性决定是否设置TF32
    if(builder->platformHasTf32()) {
        b_config->setFlag(nvinfer1::BuilderFlag::kTF32);
    } else {
        b_config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }

    // 2.4 依据模型特性设置DLA相关参数
    if(builder->getNbDLACores()){
        b_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        b_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        b_config->setDLACore(0);
    }
    else{
        b_config->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
    }

    // TODO: 这里要探究一下是否一定需要再b_config中设置stream对象

    // 2.5 按照config配置进行序列化, 并写入到文件中
    // -> nvinfer1::IHostMemory* 
    auto trt_memory_ptr = builder->buildSerializedNetwork(*network, *b_config);

    std::fstream trt_file_handle(trt_model_file, std::ios::out | std::ios::binary);
    trt_file_handle.write(static_cast<char*>(trt_memory_ptr->data()), trt_memory_ptr->size());
    trt_file_handle.close();
    
    return 0;
}
