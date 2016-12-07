//
//  fftcc_bridge.cpp
//  dic
//
//  Created by Tue Le on 9/26/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#include "fftcc_bridge.hpp"

#include "fftcc.hpp"

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
  const int_t iMarginY) {
  
  return new FFTCC(
    iImgWidth,
    iImgHeight,
    iROIWidth,
    iROIHeight,
    iStartX,
    iStartY,
    iSubsetX,
    iSubsetY,
    iGridSpaceX,
    iGridSpaceY,
    iMarginX,
    iMarginY);
}

void initializeFFTCC(
  void* fftcc,
  const unsigned char* refImg,
  int_t** iPOIXY,
  real_t** fU,
  real_t** fV,
  real_t** fZNCC) {
  
  ((FFTCC*)fftcc)->initializeFFTCC(refImg, *iPOIXY, *fU, *fV, *fZNCC);
}

void algorithmFFTCC(
  void* fftcc,
  const unsigned char* tarImg,
  const int_t* iPOIXY,
  real_t* fU,
  real_t* fV,
  real_t* fZNCC) {
  
  ((FFTCC*)fftcc)->algorithmFFTCC(tarImg, iPOIXY, fU, fV, fZNCC);
}
  
void destroyFFTCC(void* fftcc) {
  delete ((FFTCC*)fftcc);
}

int getNumPoiXFFTCC(void* fftcc) {
  return ((FFTCC*)fftcc)->getNumPOIsX();
}

int getNumPoiYFFTCC(void* fftcc) {
  return ((FFTCC*)fftcc)->getNumPOIsY();
}
