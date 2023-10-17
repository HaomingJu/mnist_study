
#include <iostream>
#include <cstdio>
#include <NvInfer.h>
#include "DebugLogger.h"

int main(int argc, char* argv[]) {
    const char* onnx_model_file = argv[1];
    DebugLogger logger("DeviceInfo");

    // -> nvinfer1::IBuilder* 
    auto builder = nvinfer1::createInferBuilder(logger);

    int32_t num_dla_core = builder->getNbDLACores();
    int32_t max_threads = builder->getMaxThreads();
    bool has_TF32 = builder->platformHasTf32();
    bool has_fast_fp16 = builder->platformHasFastFp16();
    bool has_fast_int8 = builder->platformHasFastInt8();



    printf("DLA Core: %d\n", num_dla_core);
    printf("Max threads: %d\n", max_threads);
    printf("has TF32: %d\n", has_TF32);
    printf("has fast fp16: %d\n", has_fast_fp16);
    printf("has fast int8: %d\n", has_fast_int8);

    
    return 0;
}
