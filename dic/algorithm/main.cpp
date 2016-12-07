//
//  main.cpp
//  dic
//
//  Created by Tue Le on 9/5/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#include "main.hpp"
#include "algo.hpp"
#include "util.hpp"

#include <iostream>
#include <cstdio>
#include <iomanip>

void initializePOI(
  const int x,
  const int y,
  const int maxIteration,
  const double dConvergeCriterion,
  POI* poi) {

  poi->X = x;
  poi->Y = y;

  for (int k = 0; k < 6; k++) {
    poi->dP0[k] = 0;
    poi->dP[k] = 0;
    poi->dDP[k] = 0;
  }

  poi->ZNCC = 10;
  poi->ZNSSD = 10;
  poi->iDarkSubset = 0;
  poi->iOutofROI = 0;
  poi->iInvertibleMatrix = 0;
  poi->iMaxIteration = maxIteration;
  poi->iIteration = 1;
  poi->dConvergeCriterion = dConvergeCriterion;
  poi->iProcessed = 0;
}

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
  const double dConvergeCriterion) {

	// Define the size of region of interest (ROI).
  int cropMarginX = accuracyOrder;
  int cropMarginY = accuracyOrder;

  int roiWidth = imgWidth - cropMarginX * 2;
  int roiHeight = imgHeight - cropMarginY * 2;

  // Define the size of subset window for IC-GN algorithm.
  int subsetW = subsetX * 2 + 1;
  int subsetH = subsetY * 2 + 1;

  // Define the size of subset window for FFT-CC algorithm.
  int fftSubW = subsetX * 2;
  int fftSubH = subsetY * 2;

  // Estimate the number of points of interest(POIs).
  int numberX = int(floor((roiWidth - subsetX * 2 - marginX * 2) / double(gridSpaceX))) + 1;
  int numberY = int(floor((roiHeight - subsetY * 2 - marginY * 2) / double(gridSpaceY))) + 1;

  // Data structure initialization.
  double** dBuffX = initialize2D<double>(imgHeight, imgWidth, 0);
  double** dBuffY = initialize2D<double>(imgHeight, imgWidth, 0);

  double**** dTBicubic = initialize4D<double>(roiHeight, roiWidth, 4, 4, 0);

  double** dR = initialize2D<double>(roiHeight, roiWidth, 0);
  double** dRx = initialize2D<double>(roiHeight, roiWidth, 0);
  double** dRy = initialize2D<double>(roiHeight, roiWidth, 0);
  double** dT = initialize2D<double>(roiHeight, roiWidth, 0);
  double** dTx = initialize2D<double>(roiHeight, roiWidth, 0);
  double** dTy = initialize2D<double>(roiHeight, roiWidth, 0);
  double** dTxy = initialize2D<double>(roiHeight, roiWidth, 0);

  // Initialize the data structure for FFTW.
  double* dSubset1 = initialize1D<double>(fftSubW * fftSubH, 0); // Subset R.
  double* dSubset2 = initialize1D<double>(fftSubW * fftSubH, 0); // Subset T.
  double* dSubsetC = initialize1D<double>(fftSubW * fftSubH, 0); // Matrix C.

  fftw_complex* freqDom1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSubW * (fftSubH / 2 + 1));
  fftw_complex* freqDom2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSubW * (fftSubH / 2 + 1));
  fftw_complex* freqDomfg = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSubW * (fftSubH / 2 + 1));

  fftw_plan fftwPlan1 = fftw_plan_dft_r2c_2d(fftSubW, fftSubH, dSubset1, freqDom1, FFTW_ESTIMATE);
  fftw_plan fftwPlan2 = fftw_plan_dft_r2c_2d(fftSubW, fftSubH, dSubset2, freqDom2, FFTW_ESTIMATE);
  fftw_plan rfftwPlan = fftw_plan_dft_c2r_2d(fftSubW, fftSubH, freqDomfg, dSubsetC, FFTW_ESTIMATE);

  // Initialize data structure for IC-GN algorithm.
  double** dSubsetR = initialize2D<double>(subsetH, subsetW, 0); // Subset window in R.
  double** dSubsetT = initialize2D<double>(subsetH, subsetW, 0); // Subset window in T.
  double*** dRDescent = initialize3D<double>(subsetH, subsetW, 6, 0); // The steepest descent image DealtR * dW / dp.
  double**** dHessianXY = initialize4D<double>(subsetH, subsetW, 6, 6, 0); // Hessian matrix at each point in subset R.
  double** dError = initialize2D<double>(subsetH, subsetW, 0); // Error matrix in subset R.
  double**** dJacobian = initialize4D<double>(subsetH, subsetW, 2, 6, 0); // Jacobian matrix dW/dp in subset R.

  // Initialize the POI matrix.
  POI processingPOI;
  POI** poi = new POI *[numberY];

  for (int i = 0; i < numberY; i++) {
    poi[i] = new POI[numberX];

    for (int j = 0; j < numberX; j++) {
      int x = int(1 + marginX + subsetX + j * gridSpaceX);
      int y = int(1 + marginY + subsetY + i * gridSpaceY);

      initializePOI(x, y, maxIteration, dConvergeCriterion, &poi[i][j]);
    }
  }

  // Fill R and T.
  for (int i = 0; i < roiHeight; i++) {
    for (int j = 0; j < roiWidth; j++) {
      dR[i][j] = ref[i + cropMarginY][j + cropMarginX];
      dT[i][j] = tar[i + cropMarginY][j + cropMarginX];
    }
  }

  // Compute the gradients of R and T.
  GradientX(ref, dRx, dBuffX, imgWidth, imgHeight, roiWidth, roiHeight, accuracyOrder);
  GradientY(ref, dRy, dBuffY, imgWidth, imgHeight, roiWidth, roiHeight, accuracyOrder);

  GradientX(tar, dTx, dBuffX, imgWidth, imgHeight, roiWidth, roiHeight, accuracyOrder);
  GradientY(tar, dTy, dBuffY, imgWidth, imgHeight, roiWidth, roiHeight, accuracyOrder);
  GradientY(dBuffX, dTxy, dBuffY, imgWidth, imgHeight, roiWidth, roiHeight, accuracyOrder);

  // Compute the bicubic interpolation coefficients of T.
  BiCubicCoefficient(dT, dTx, dTy, dTxy, roiWidth, roiHeight, dTBicubic);

  for (int i = 0; i < numberY; i++) {
    for (int j = 0; j < numberX; j++) {
      processingPOI = poi[i][j];

      FFTCC(dR, dT, dSubset1, dSubset2, dSubsetC, freqDom1, freqDom2, freqDomfg, fftwPlan1, fftwPlan2, rfftwPlan, subsetX, subsetY, processingPOI);
      
//      std::cout << std::fixed;
//      std::cout << std::setprecision(5) << processingPOI.X << " " << processingPOI.Y << " " << processingPOI.dP0[0] << " " << processingPOI.dP0[3] << std::endl;
//      
//      processingPOI.dP0[0] = 0;
//      processingPOI.dP0[0] = 0;
      

      ICGN(dR, dRx, dRy, dT, dTBicubic, dSubsetR, dSubsetT, dJacobian, dRDescent, dHessianXY, dError, roiWidth, roiHeight, subsetX, subsetY, processingPOI);

      // Compute the ZNCC.
      processingPOI.ZNCC = 1 - processingPOI.ZNSSD / 2;
      poi[i][j] = processingPOI;
    }
  }

  destroy(subsetH, subsetW, 6, dHessianXY);
  destroy(subsetH, subsetW, 2, dJacobian);
  destroy(subsetH, subsetW, dRDescent);
  destroy(subsetH, dSubsetR);
  destroy(subsetH, dSubsetT);
  destroy(subsetH, dError);

  fftw_destroy_plan(fftwPlan1);
  fftw_destroy_plan(fftwPlan2);
  fftw_destroy_plan(rfftwPlan);
  fftw_free(freqDom1);
  fftw_free(freqDom2);
  fftw_free(freqDomfg);

  destroy(dSubset1);
  destroy(dSubset2);
  destroy(dSubsetC);

  destroy(roiHeight, roiWidth, 4, dTBicubic);
  destroy(roiHeight, dR);
  destroy(roiHeight, dRx);
  destroy(roiHeight, dRy);
  destroy(roiHeight, dT);
  destroy(roiHeight, dTx);
  destroy(roiHeight, dTy);
  
  RunResult *result = new RunResult();
  result->poi = poi;
  result->numberX = numberX;
  result->numberY = numberY;
  result->cropMarginX = cropMarginX;
  result->cropMarginY = cropMarginY;
  result->imgWidth = imgWidth;
  result->imgHeight = imgHeight;

  return result;
}

RunResult* runWithDefaultParams(double** ref, double** tar, int imgWidth, int imgHeight) {

  int accuracyOrder = 4;
  int subsetX = 16;
  int subsetY = 16;
  int marginX = 10;
  int marginY = 10;
  int gridSpaceX = 30;
  int gridSpaceY = 30;
  int maxIteration = 20;
  double dConvergeCriterion = 0.001;

  return runWithParams(
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
}
