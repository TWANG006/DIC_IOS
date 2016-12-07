//
//  algo.hpp
//  dic
//
//  Created by Tue Le on 9/5/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef algo_hpp
#define algo_hpp

#include "fftw3.h"
#include "math.h"
#include "poi.hpp"

static double m_dBicubicMatrix[16][16] = {
  { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0 },
  { -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0 },
  { 9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1 },
  { -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1 },
  { 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0 },
  { -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1 },
  { 4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1 }
};

static double dWarp[3][3] = {
  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 }
};

static double dHessian[6][6] = {
  { 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0 }
};

static double dInvHessian[6][6] = {
  { 1, 0, 0, 0, 0, 0 },
  { 0, 1, 0, 0, 0, 0 },
  { 0, 0, 1, 0, 0, 0 },
  { 0, 0, 0, 1, 0, 0 },
  { 0, 0, 0, 0, 1, 0 },
  { 0, 0, 0, 0, 0, 1 }
};

static double dNumerator[6] = { 0, 0, 0, 0, 0, 0 };

static double dTao[16];
static double dAlpha[16];

extern POI& FFTCC(
  double** dRef,
  double** dTar,
  double* dSubset1,
  double* dSubset2,
  double* dSubsetC,
  fftw_complex *FreqDom1,
  fftw_complex *FreqDom2,
  fftw_complex *FreqDomfg,
  fftw_plan fftwPlan1,
  fftw_plan fftwPlan2,
  fftw_plan rfftwPlan,
  int iSubsetX,
  int iSubsetY,
  POI& mPOI);

extern POI& ICGN(
  double** dRef,
  double** dRefx,
  double** dRefy,
  double** dTar,
  double**** dTBicubic,
  double** dSubsetR,
  double** dSubsetT,
  double**** dJacobian,
  double*** dRDescent,
  double**** dHessianXY,
  double** dError,
  int iROIWidth,
  int iROIHeight,
  int iSubsetX,
  int iSubsetY,
  POI& mPOI);

extern void GradientX(
  double** dImageT,
  double** dTx,
  double** dTBuff,
  int iImgWidth,
  int iImgHeight,
  int iROIWidth,
  int iROIHeight,
  int iAccuracyOrder);

extern void GradientY(
  double** dImageT,
  double** dTy,
  double** dTBuff,
  int iImgWidth,
  int iImgHeight,
  int iROIWidth,
  int iROIHeight,
  int iAccuracyOrder);

extern void BiCubicCoefficient(
  double** dT,
  double** dTx,
  double** dTy,
  double** dTxy,
  int iROIWidth,
  int iROIHeight,
  double**** dTBicubic);


#endif /* algo_hpp */
