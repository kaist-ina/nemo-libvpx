//==============================================================================
//
//  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C++ API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//

#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <algorithm>

#include <vpx_mem/vpx_mem.h>
#include <vpx/vpx_nemo.h>
#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "udlExample.hpp"
#include "CreateUserBuffer.hpp"
#include "PreprocessInput.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "main.hpp"

#include <GLES2/gl2.h>
#include "CreateGLBuffer.hpp"
#include <android/log.h>
#define TAG "main.cpp JNI"
#define _UNKNOWN   0
#define _DEFAULT   1
#define _VERBOSE   2
#define _DEBUG    3
#define _INFO        4
#define _WARN        5
#define _ERROR    6
#define _FATAL    7
#define _SILENT       8
#define LOGUNK(...) __android_log_print(_UNKNOWN,TAG,__VA_ARGS__)
#define LOGDEF(...) __android_log_print(_DEFAULT,TAG,__VA_ARGS__)
#define LOGV(...) __android_log_print(_VERBOSE,TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(_DEBUG,TAG,__VA_ARGS__)
#define LOGI(...) __android_log_print(_INFO,TAG,__VA_ARGS__)
#define LOGW(...) __android_log_print(_WARN,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(_ERROR,TAG,__VA_ARGS__)
#define LOGF(...) __android_log_print(_FATAL,TAG,__VA_ARGS__)
#define LOGS(...) __android_log_print(_SILENT,TAG,__VA_ARGS__)

SNPE::SNPE(nemo_dnn_runtime runtime_mode)
{
    {
        switch (runtime_mode) {
            case CPU_FLOAT32:
                runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
                fprintf(stdout, "SNPE: CPU_FLOAT32\n");
                break;
            case GPU_FLOAT32_16_HYBRID:
                runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID;
                fprintf(stdout, "SNPE: GPU_FLOAT32_16_HYBRID\n");
                break;
            case DSP_FIXED8:
                runtime = zdl::DlSystem::Runtime_t::DSP_FIXED8_TF;
                fprintf(stdout, "SNPE: DSP_FIXED8\n");
                break;
            case GPU_FLOAT16:
                runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
                fprintf(stdout, "SNPE: GPU_FLOAT16\n");
                break;
            case AIP_FIXED8:
                runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
                fprintf(stdout, "SNPE: AIP_FIXED8\n");
                break;
            default:
                fprintf(stdout, "SNPE: Unset\n");
                runtime = zdl::DlSystem::Runtime_t::UNSET;
                break;
        }
    }

    fprintf(stdout, "SNPE: Allocate class\n");
}

void *snpe_alloc(nemo_dnn_runtime runtime_mode) {
    return static_cast<void *>(new SNPE(runtime_mode));
}

SNPE::~SNPE(void){
    if (snpe)
    {
        snpe.reset();
    }
}

void snpe_free(void *snpe) {
    delete static_cast<SNPE *>(snpe);
}

int SNPE::check_runtime(void){
    int result;
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    LOGE("SNPE: version %s", Version.asString().c_str());
    fprintf(stdout, "SNPE: Version %s\n", Version.asString().c_str()); //Print Version number

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        fprintf(stdout, "Selected runtime not present. Falling back to CPU.\n");
        return -1;
    }

    fprintf(stdout, "SNPE: Check runtime\n");
    return 0;
}

int snpe_check_runtime(void *snpe){
    return static_cast<SNPE *>(snpe)->check_runtime();
}

int SNPE::init_network(const char *path){
    static std::string dlc;
    static zdl::DlSystem::RuntimeList runtimeList;
    bool useUserSuppliedBuffers = false;
    bool usingInitCaching = false;
    zdl::DlSystem::UDLFactoryFunc udlFunc = UdlExample::MyUDLFactory;
    zdl::DlSystem::UDLBundle udlBundle; udlBundle.cookie = (void*)0xdeadbeaf, udlBundle.func = udlFunc; // 0xdeadbeaf to test cookie
    zdl::DlSystem::PlatformConfig platformConfig;

    //check if dlc is valid file
    std::ifstream dlcFile(path);
    if(!dlcFile){
        LOGE("DLC does not exist");
        fprintf(stdout, "DLC does not exist\n");
        return -1;
    }

    //Open dlc
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(path);
    if (container == nullptr)
    {
        LOGE("Faild to open a dlc file");
        fprintf(stdout, "Failed to open a dlc file\n");
        return -1;
    }

    snpe = setBuilderOptions(container, runtime, runtimeList, udlBundle, useUserSuppliedBuffers, platformConfig, usingInitCaching);
    if(snpe == nullptr){
        LOGE("failed to build a snpe object");
        fprintf(stdout, "Failed build a snpe object\n");
        return -1;
    }

    fprintf(stdout, "SNPE: Init network\n");
    return 0;
}

//TODO (snpe): config best user buffer
int snpe_load_network(void *snpe, const char *path){
    return static_cast<SNPE *>(snpe)->init_network(path);
}

int SNPE::execute_byte(uint8_t *input_buffer, float *output_buffer, int number_of_elements){
    bool execStatus = false;
    zdl::DlSystem::TensorMap outputTensorMap;
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensorFromByteBuffer(snpe, input_buffer, number_of_elements);

    execStatus = snpe->execute(inputTensor.get(), outputTensorMap);

    //Save the execution results if execution successful
    if (!execStatus){
        fprintf(stdout, "Failed to run a model\n");
        return -1;
    }

    saveOutputToBuffer(outputTensorMap, output_buffer);
    fprintf(stdout, "SNPE: Execute a DNN\n");
    return 0;
}

int snpe_execute_byte(void *snpe, uint8_t *input_buffer, float *output_buffer, int number_of_elements){
    return static_cast<SNPE *>(snpe)->execute_byte(input_buffer, output_buffer, number_of_elements);
}

int SNPE::execute_float(float *input_buffer, float *output_buffer, int number_of_elements){
    bool execStatus = false;
    zdl::DlSystem::TensorMap outputTensorMap;
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensorFromFloatBuffer(snpe, input_buffer, number_of_elements);

    execStatus = snpe->execute(inputTensor.get(), outputTensorMap);

    // Save the execution results if execution successful
    if (!execStatus){
        fprintf(stdout, "Failed to run a model\n");
        return -1;
    }

    saveOutputToBuffer(outputTensorMap, output_buffer);
    fprintf(stdout, "SNPE: Execute a DNN\n");
    return 0;
}

int snpe_execute_float(void *snpe, float *input_buffer, float *output_buffer, int number_of_elements){
    return static_cast<SNPE *>(snpe)->execute_float(input_buffer, output_buffer, number_of_elements);
}
