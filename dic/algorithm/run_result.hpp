//
//  run_result.hpp
//  dic
//
//  Created by Tue Le on 9/11/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef run_result_hpp
#define run_result_hpp

#include <stdio.h>

#include "poi.hpp"

typedef struct POI POI;

struct RunResult {
  int numberX;
  int numberY;
  
  int cropMarginX;
  int cropMarginY;
  
  int imgWidth;
  int imgHeight;
  
  POI** poi;
};

struct FFTCCResult {
  double u, v;
};

typedef struct RunResult RunResult;

#ifdef __cplusplus
extern "C" {
#endif

int runResultGetNumberX(RunResult* result);
int runResultGetNumberY(RunResult* result);

int runResultGetCropMarginX(RunResult* result);
int runResultGetCropMarginY(RunResult* result);

int runResultGetImgWidth(RunResult *result);
int runResultGetImgHeight(RunResult *result);

POI* runResultGetPOI(RunResult* result, int x, int y);

#ifdef __cplusplus
}
#endif

#endif /* run_result_hpp */
