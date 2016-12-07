//
//  bridge.hpp
//  dic
//
//  Created by Tue Le on 9/6/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef bridge_hpp
#define bridge_hpp

#include <stdio.h>

#include "run_result.hpp"
#include "fftcc_bridge.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RunResult RunResult;

RunResult* bridge(
  double* flattenRef,
  double* flattenTar,
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

#ifdef __cplusplus
}
#endif

#endif /* bridge_hpp */
