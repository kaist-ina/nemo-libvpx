//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef LOADINPUTTENSOR_H
#define LOADINPUTTENSOR_H

#include <unordered_map>
#include <string>
#include <vector>

#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorMap.hpp"

typedef unsigned int GLuint;
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE>& snpe , std::vector<std::string>& fileLines);
std::tuple<zdl::DlSystem::TensorMap, bool> loadMultipleInput (std::unique_ptr<zdl::SNPE::SNPE> & snpe , std::string& fileLine);

bool loadInputUserBufferFloat(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                                std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                                std::vector<std::string>& fileLines);

bool loadInputUserBufferTf8(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::vector<std::string>& fileLines,
                         zdl::DlSystem::UserBufferMap& inputMap);

void loadInputUserBuffer(std::unordered_map<std::string, GLuint>& applicationBuffers,
                                std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                                const GLuint inputglbuffer);


std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensorFromByteBuffer(std::shared_ptr<zdl::SNPE::SNPE>& snpe , unsigned char * buffer, int buffer_size);
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensorFromFloatBuffer(std::shared_ptr<zdl::SNPE::SNPE>& snpe , float * buffer, int buffer_size);

#endif
