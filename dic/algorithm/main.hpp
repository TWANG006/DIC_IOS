//
//  main.hpp
//  dic
//
//  Created by Tue Le on 9/5/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef main_hpp
#define main_hpp

#include <stdio.h>

#include "run_result.hpp"

RunResult* runWithDefaultParams(
  double** ref,
  double** tar,
  int width,
  int height);

RunResult* runWithParams(
  double** ref,
  double** tar,
  const int imgWidth,
  const int imgHeight,
  const int accuracyOrder,
  const int subsetX,
  const int subsetY,
  const int marginX,
  const int marginY,
  const int gridSpaceX,
  const int gridSpaceY,
  const int maxIteration,
  const double dConvergeCriterion);

FFTCCResult* runFFTCC();

#endif /* main_hpp */
