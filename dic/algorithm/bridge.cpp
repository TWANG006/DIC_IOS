//
//  bridge.cpp
//  dic
//
//  Created by Tue Le on 9/6/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#include "bridge.hpp"
#include "main.hpp"
#include "util.hpp"

void reconstruct(double* flatten, double** image, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      image[y][x] = flatten[y * width + x];
    }
  }
}

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
  const double dConvergeCriterion) {
  
  double** ref = initialize2D<double>(imgHeight, imgWidth, 0);
  double** tar = initialize2D<double>(imgHeight, imgWidth, 0);
  
  reconstruct(flattenRef, ref, imgWidth, imgHeight);
  reconstruct(flattenTar, tar, imgWidth, imgHeight);
  
  RunResult* result = runWithParams(
    ref,
    tar,
    imgWidth,
    imgHeight,
    accuracyOrder,
    subsetX,
    subsetY,
    marginX,
    marginY,
    gridSpaceX,
    gridSpaceY,
    maxIteration,
    dConvergeCriterion);
  
  destroy(imgHeight, ref);
  destroy(imgHeight, tar);
  
  return result;
}
