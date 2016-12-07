//
//  fftcc_bridge.hpp
//  dic
//
//  Created by Tue Le on 9/26/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef fftcc_bridge_hpp
#define fftcc_bridge_hpp

#include <stdio.h>

#include "fftcc_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void* newFFTCC(
  const int_t iImgWidth,
  const int_t iImgHeight,
  const int_t iROIWidth,
  const int_t iROIHeight,
  const int_t iStartX,
  const int_t iStartY,
  const int_t iSubsetX,
  const int_t iSubsetY,
  const int_t iGridSpaceX,
  const int_t iGridSpaceY,
  const int_t iMarginX,
  const int_t iMarginY);
  
void initializeFFTCC(
  void* fftcc,
  const unsigned char* refImg,
  int_t** iPOIXY,
  real_t** fU,
  real_t** fV,
  real_t** fZNCC);

void algorithmFFTCC(
  void* fftcc,
  const unsigned char* tarImg,
  const int_t* iPOIXY,
  real_t* fU,
  real_t* fV,
  real_t* fZNCC);

void destroyFFTCC(void* fftcc);

int getNumPoiXFFTCC(void *fftcc);
int getNumPoiYFFTCC(void *fftcc);

#ifdef __cplusplus
}
#endif
#endif /* fftcc_bridge_hpp */
