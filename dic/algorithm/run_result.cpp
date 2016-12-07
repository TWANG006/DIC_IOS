//
//  run_result.cpp
//  dic
//
//  Created by Tue Le on 9/11/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#include <cstdio>
#include <iostream>

#include "run_result.hpp"

int runResultGetNumberX(RunResult* result) {
  return result->numberX;
}

int runResultGetNumberY(RunResult* result) {
  return result->numberY;
}

int runResultGetCropMarginX(RunResult *result) {
 return result->cropMarginX;
}

int runResultGetCropMarginY(RunResult *result) {
 return result->cropMarginY;
}

int runResultGetImgWidth(RunResult* result) {
  return result->imgWidth;
}

int runResultGetImgHeight(RunResult* result) {
  return result->imgHeight;
}

POI* runResultGetPOI(RunResult* result, int x, int y) {
  return &result->poi[y][x];
}
