/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "nvtx3/nvToolsExt.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>

static const int NUM_CLASSES_YOLO = 80;
#define OBJECTLISTSIZE 25200
#define BLOCKSIZE  1024
thrust::device_vector<NvDsInferParseObjectInfo> objects_v(OBJECTLISTSIZE);
thrust::device_vector<float> objects_floats(OBJECTLISTSIZE * (NUM_CLASSES_YOLO + 5));


extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);


__global__ void decodeYoloTensor_cuda(NvDsInferParseObjectInfo *binfo/*output*/, void* data, int dimensions, int rows,
                                        int netW, int netH, float Threshold, bool is_fp16){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < rows) {
        float maxProb, bx, by, bw, bh, maxScore;
        int maxIndex;
        if (is_fp16){
            maxProb = __half2float(*(((__half*)data) + idx * dimensions + 4));
            //maxProb < Threshold, directly return
            if (maxProb < Threshold) {
                binfo[idx].detectionConfidence = 0;
                return;
            }
            bx = __half2float(*(((__half*)data) + idx * dimensions + 0));
            by = __half2float(*(((__half*)data) + idx * dimensions + 1));
            bw = __half2float(*(((__half*)data) + idx * dimensions + 2));
            bh = __half2float(*(((__half*)data) + idx * dimensions + 3));
            float class_score;
            maxScore = 0;
            maxIndex = 0;
            for (int j = 0; j < dimensions - 5; j++) {
                class_score = __half2float(*(((__half*)data) + idx * dimensions + 5 + j));
               if (class_score > maxScore) {
                  maxIndex = j;
                  maxScore = class_score;
               }
            }

        } else {
            maxProb = ((float*)data)[idx * dimensions + 4];
            //maxProb < Threshold, directly return
            if (maxProb < Threshold) {
                binfo[idx].detectionConfidence = 0;
                return;
            }
            bx = ((float*)data)[idx * dimensions + 0];
            by = ((float*)data)[idx * dimensions + 1];
            bw = ((float*)data)[idx * dimensions + 2];
            bh = ((float*)data)[idx * dimensions + 3];
            float * classes_scores = ((float*)data) + idx * dimensions + 5;
            maxScore = 0;
            maxIndex = 0;
            for (int j = 0; j < dimensions - 5; j++) {
               if (*classes_scores > maxScore) {
                  maxIndex = j;
                  maxScore = *classes_scores;
               }
               classes_scores++;
            }
        }
        float stride = 1.0;
        float xCenter = bx * stride;
        float yCenter = by * stride;
        float x0 = xCenter - bw / 2;
        float y0 = yCenter - bh / 2;
        float x1 = x0 + bw;
        float y1 = y0 + bh;
        x0 = fminf(netW, fmaxf(float(0.0), x0));
        y0 = fminf(netH, fmaxf(float(0.0), y0));
        x1 = fminf(netW, fmaxf(float(0.0), x1));
        y1 = fminf(netH, fmaxf(float(0.0), y1));
        binfo[idx].left = x0;
        binfo[idx].top = y0;
        binfo[idx].width = fminf(float(netW), fmaxf(float(0.0), x1-x0));
        binfo[idx].height = fminf(float(netH), fmaxf(float(0.0), y1-y0));
        binfo[idx].detectionConfidence = maxProb;
        binfo[idx].classId = maxIndex;
    }
    return;
}
static bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{

    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }
    float* data = (float*)layer.buffer;


    const int dimensions = layer.inferDims.d[1];
    int rows = layer.inferDims.numElements / layer.inferDims.d[1];

    int GRIDSIZE = ((OBJECTLISTSIZE-1)/BLOCKSIZE)+1;
    //find the min threshold
    float min_PreclusterThreshold = *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
        detectionParams.perClassPreclusterThreshold.end()));
    decodeYoloTensor_cuda<<<GRIDSIZE,BLOCKSIZE>>>
        (thrust::raw_pointer_cast(objects_v.data()), data, dimensions, rows, networkInfo.width,
        networkInfo.height, min_PreclusterThreshold, layer.dataType == HALF);
    objectList.resize(OBJECTLISTSIZE);
    thrust::copy(objects_v.begin(),objects_v.end(),objectList.begin());//the same as cudamemcpy
    return true;
}

extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    nvtxRangePush("NvDsInferParseYolo");
    bool ret = NvDsInferParseYolo (
        outputLayersInfo, networkInfo, detectionParams, objectList);

    nvtxRangePop();
    return ret;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo);