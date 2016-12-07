//
//  algo.cpp
//  dic
//
//  Created by Tue Le on 9/5/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#include "algo.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

POI& FFTCC(
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
  POI& mPOI) {
  
  int k, l, m, n;
  mPOI.iProcessed = 1;

  // Define the size of subset window for FFT-CC algorithm.
  int m_iFFTSubW = iSubsetX * 2;
  int m_iFFTSubH = iSubsetY * 2;

  int m_iCorrPeak, m_iU, m_iV;
  double m_dSubAveR, m_dSubAveT, m_dSubNorR, m_dSubNorT;
  double m_dFFTSubsetSize = m_iFFTSubH * m_iFFTSubW;

  m_dSubAveR = 0; // R_m
  m_dSubAveT = 0; // T_m

  // Feed the gray intensity values into subsets.
  for (l = 0; l < m_iFFTSubH; l++) {
    for (m = 0; m < m_iFFTSubW; m++) {
      dSubset1[l * m_iFFTSubW + m] = dRef[mPOI.Y - iSubsetY + l][mPOI.X - iSubsetX + m];
      m_dSubAveR += dSubset1[l * m_iFFTSubW + m];
      dSubset2[l * m_iFFTSubW + m] = dTar[mPOI.Y - iSubsetY + l][mPOI.X - iSubsetX + m];
      m_dSubAveT += dSubset2[l * m_iFFTSubW + m];
    }
  }
  m_dSubAveR = m_dSubAveR / m_dFFTSubsetSize;
  m_dSubAveT = m_dSubAveT / m_dFFTSubsetSize;

  m_dSubNorR = 0; // sqrt (Sigma(R_i - R_m)^2)
  m_dSubNorT = 0; // sqrt (Sigma(T_i - T_m)^2)
  for (l = 0; l < m_iFFTSubH; l++) {
    for (m = 0; m < m_iFFTSubW; m++) {
      dSubset1[(l * m_iFFTSubW + m)] -= m_dSubAveR;
      dSubset2[(l * m_iFFTSubW + m)] -= m_dSubAveT;
      m_dSubNorR += pow((dSubset1[l * m_iFFTSubW + m]), 2);
      m_dSubNorT += pow((dSubset2[l * m_iFFTSubW + m]), 2);
    }
  }
  // Terminate the processing if one of the two subsets is full of zero intensity.
  if (m_dSubNorR == 0 || m_dSubNorT == 0) {
    mPOI.iDarkSubset = 1; //set flag
    return mPOI;
  }

  // FFT-CC algorithm accelerated by FFTW.
  fftw_execute(fftwPlan1);
  fftw_execute(fftwPlan2);
  for (n = 0; n < m_iFFTSubW * (m_iFFTSubH / 2 + 1); n++) {
    FreqDomfg[n][0] = (FreqDom1[n][0] * FreqDom2[n][0]) + (FreqDom1[n][1] * FreqDom2[n][1]);
    FreqDomfg[n][1] = (FreqDom1[n][0] * FreqDom2[n][1]) - (FreqDom1[n][1] * FreqDom2[n][0]);
  }
  fftw_execute(rfftwPlan);

  mPOI.ZNCC = -2; // maximum C
  m_iCorrPeak = 0; // loacatoin of maximum C
  // Search for maximum C, then normalize C.
  for (k = 0; k < m_iFFTSubW * m_iFFTSubH; k++) {
    if (mPOI.ZNCC < dSubsetC[k]) {
      mPOI.ZNCC = dSubsetC[k];
      m_iCorrPeak = k;
    }
  }
  mPOI.ZNCC /= sqrt(m_dSubNorR * m_dSubNorT) * m_dFFTSubsetSize; //parameter for normalization
  // Calculate the loacation of maximum C.
  m_iU = m_iCorrPeak % m_iFFTSubW;
  m_iV = int(m_iCorrPeak / m_iFFTSubW);

  // Shift the C peak to the right quadrant.
  if (m_iU > iSubsetX) {
    m_iU -= m_iFFTSubW;
  }
  if (m_iV > iSubsetY) {
    m_iV -= m_iFFTSubH;
  }
  mPOI.dP0[0] = m_iU; // integer-pixel u
  mPOI.dP0[1] = 0;
  mPOI.dP0[2] = 0;
  mPOI.dP0[3] = m_iV; // integer-pixel v
  mPOI.dP0[4] = 0;
  mPOI.dP0[5] = 0;

  return mPOI;
}

POI& ICGN(
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
  POI& mPOI) {
  
  int k, l, m, n;
  mPOI.iProcessed = 2;

  // Define the size of subset window for IC-GN algorithm.
  int m_iSubsetW = iSubsetX * 2 + 1;
  int m_iSubsetH = iSubsetY * 2 + 1;

  int m_iTemp, m_iTempX, m_iTempY;
  double m_dSubAveR, m_dSubAveT, m_dSubNorR, m_dSubNorT;
  double m_dU, m_dUx, m_dUy, m_dV, m_dVx, m_dVy;
  double m_dDU, m_dDUx, m_dDUy, m_dDV, m_dDVx, m_dDVy;
  double dWarpX, dWarpY, m_dTemp, m_dTempX, m_dTempY;

  double m_dSubsetSize = m_iSubsetH * m_iSubsetW;

  // Initialize Subset R.
  m_dSubAveR = 0; // R_m
  m_dSubNorR = 0; // sqrt (Sigma(R_i - R_m)^2)

  // Initialize the Hessian matrix for each subset.
  for (k = 0; k < 6; k++) {
    for (n = 0; n < 6; n++) {
      dHessian[k][n] = 0;
    }
  }

  // Feed the gray intensity to subset R.
  for (l = 0; l < m_iSubsetH; l++) {
    for (m = 0; m < m_iSubsetW; m++) {
      dSubsetR[l][m] = dRef[mPOI.Y - iSubsetY + l][mPOI.X - iSubsetX + m];
      m_dSubAveR += dSubsetR[l][m];

      // Evaluate the Jacbian dW/dp at (x, 0).
      dJacobian[l][m][0][0] = 1;
      dJacobian[l][m][0][1] = m - iSubsetX;
      dJacobian[l][m][0][2] = l - iSubsetY;
      dJacobian[l][m][0][3] = 0;
      dJacobian[l][m][0][4] = 0;
      dJacobian[l][m][0][5] = 0;
      dJacobian[l][m][1][0] = 0;
      dJacobian[l][m][1][1] = 0;
      dJacobian[l][m][1][2] = 0;
      dJacobian[l][m][1][3] = 1;
      dJacobian[l][m][1][4] = m - iSubsetX;
      dJacobian[l][m][1][5] = l - iSubsetY;

      // Compute the steepest descent image DealtR * dW / dp.
      for (k = 0; k < 6; k++) {
        dRDescent[l][m][k] = dRefx[mPOI.Y - iSubsetY + l][mPOI.X - iSubsetX + m] * dJacobian[l][m][0][k] + dRefy[mPOI.Y - iSubsetY + l][mPOI.X - iSubsetX + m] * dJacobian[l][m][1][k];
      }

      // Compute the Hessian matrix.
      for (k = 0; k < 6; k++) {
        for (n = 0; n < 6; n++) {
          dHessianXY[l][m][k][n] = dRDescent[l][m][k] * dRDescent[l][m][n]; // Hessian matrix at each point
          dHessian[k][n] += dHessianXY[l][m][k][n]; // sum of Hessian matrix at all the points in subset R
        }
      }
    }
  }

  // Check if Subset R is a all dark subset.
  if (m_dSubAveR == 0) {
    mPOI.iDarkSubset = 2;
    return mPOI;
  }

  m_dSubAveR /= m_dSubsetSize;

  for (l = 0; l < m_iSubsetH; l++) {
    for (m = 0; m < m_iSubsetW; m++) {
      dSubsetR[l][m] = dSubsetR[l][m] - m_dSubAveR; // R_i - R_m
      m_dSubNorR += pow(dSubsetR[l][m], 2); // Sigma(R_i - R_m)^2
    }
  }
  m_dSubNorR = sqrt(m_dSubNorR); // sqrt (Sigma(R_i - R_m)^2)

  if (m_dSubNorR == 0) {
    mPOI.iDarkSubset = 3; //set flag
    return mPOI;
  }

  // Invert the Hessian matrix (Gauss-Jordan algorithm).
  for (l = 0; l < 6; l++) {
    for (m = 0; m < 6; m++) {
      if (l == m) {
        dInvHessian[l][m] = 1;
      } else {
        dInvHessian[l][m] = 0;
      }
    }
  }

  for (l = 0; l < 6; l++) {
    // Find pivot (maximum lth column element) in the rest (6-l) rows.
    m_iTemp = l;
    for (m = l + 1; m < 6; m++) {
      if (dHessian[m][l] > dHessian[m_iTemp][l]) {
        m_iTemp = m;
      }
    }
    if (fabs(dHessian[m_iTemp][l]) == 0) {
      mPOI.iInvertibleMatrix = 1;
      return mPOI;
    }
    // Swap the row which has maximum lth column element.
    if (m_iTemp != l) {
      for (k = 0; k < 6; k++) {
        m_dTemp = dHessian[l][k];
        dHessian[l][k] = dHessian[m_iTemp][k];
        dHessian[m_iTemp][k] = m_dTemp;

        m_dTemp = dInvHessian[l][k];
        dInvHessian[l][k] = dInvHessian[m_iTemp][k];
        dInvHessian[m_iTemp][k] = m_dTemp;
      }
    }
    // Perform row operation to form required identity matrix out of the Hessian matrix.
    for (m = 0; m < 6; m++) {
      m_dTemp = dHessian[m][l];
      if (m != l) {
        for (n = 0; n < 6; n++) {
          dInvHessian[m][n] -= dInvHessian[l][n] * m_dTemp / dHessian[l][l];
          dHessian[m][n] -= dHessian[l][n] * m_dTemp / dHessian[l][l];
        }
      } else {
        for (n = 0; n < 6; n++) {
          dInvHessian[m][n] /= m_dTemp;
          dHessian[m][n] /= m_dTemp;
        }
      }
    }
  }

  // Initialize matrix P and DP.
  for (k = 0; k < 6; k++) {
    mPOI.dP[k] = mPOI.dP0[k]; // Transfer the initial guess to IC-GN algorithm.
  }
  m_dU = mPOI.dP[0];
  m_dUx = mPOI.dP[1];
  m_dUy = mPOI.dP[2];
  m_dV = mPOI.dP[3];
  m_dVx = mPOI.dP[4];
  m_dVy = mPOI.dP[5];

  // Initialize the warp matrix.
  dWarp[0][0] = 1 + m_dUx;
  dWarp[0][1] = m_dUy;
  dWarp[0][2] = m_dU;
  dWarp[1][0] = m_dVx;
  dWarp[1][1] = 1 + m_dVy;
  dWarp[1][2] = m_dV;
  dWarp[2][0] = 0;
  dWarp[2][1] = 0;
  dWarp[2][2] = 1;

  // Initialize DeltaP.
  m_dDU = 0;
  m_dDUx = 0;
  m_dDUy = 0;
  m_dDV = 0;
  m_dDVx = 0;
  m_dDVy = 0;

  // Fill warpped image into Subset T.
  m_dSubAveT = 0;
  m_dSubNorT = 0;
  for (l = 0; l < m_iSubsetH; l++) {
    for (m = 0; m < m_iSubsetW; m++) {
      // Calculate the location of warped subset T.
      dWarpX = mPOI.X + dWarp[0][0] * (m - iSubsetX) + dWarp[0][1] * (l - iSubsetY) + dWarp[0][2];
      dWarpY = mPOI.Y + dWarp[1][0] * (m - iSubsetX) + dWarp[1][1] * (l - iSubsetY) + dWarp[1][2];
      m_iTempX = int(dWarpX);
      m_iTempY = int(dWarpY);

      if ((m_iTempX >= 0) && (m_iTempY >= 0) && (m_iTempX < iROIWidth) && (m_iTempY < iROIHeight)) {
        // It is integer-pixel location, feed the gray intensity of T into the subset T.
        dSubsetT[l][m] = dTar[m_iTempY][m_iTempX];
        m_dSubAveT += dSubsetT[l][m];
      } else {
        mPOI.iOutofROI = 1;  // if the loacation of the warped subset T is out of the ROI, stop iteration and set p as the current value
        return mPOI;
      }
    }
  }

  m_dSubAveT /= m_dSubsetSize;
  
  // Check if Subset T is a all dark subset.
  if (m_dSubAveT == 0) {
    mPOI.iDarkSubset = 4;
    return mPOI;
  }

  for (l = 0; l < m_iSubsetH; l++) {
    for (m = 0; m < m_iSubsetW; m++) {
      dSubsetT[l][m] = dSubsetT[l][m] - m_dSubAveT; // T_i - T_m
      m_dSubNorT += pow(dSubsetT[l][m], 2); // Sigma(T_i - T_m)^2
    }
  }
  m_dSubNorT = sqrt(m_dSubNorT); // sqrt (Sigma(T_i - T_m)^2)

  if (m_dSubNorT == 0) {
    mPOI.iDarkSubset = 5; //set flag
    return mPOI;
  }

  // Compute the error image.
  for (k = 0; k < 6; k++) {
    dNumerator[k] = 0;
  }
  for (l = 0; l < m_iSubsetH; l++) {
    for (m = 0; m < m_iSubsetW; m++) {
      dError[l][m] = (m_dSubNorR / m_dSubNorT) * dSubsetT[l][m] - dSubsetR[l][m];

      // Compute the numerator.
      for (k = 0; k < 6; k++) {
        dNumerator[k] += (dRDescent[l][m][k] * dError[l][m]);
      }
    }
  }

  // Compute DeltaP.
  for (k = 0; k < 6; k++) {
    mPOI.dDP[k] = 0;
    for (n = 0; n < 6; n++) {
      mPOI.dDP[k] += (dInvHessian[k][n] * dNumerator[n]);
    }
  }
  m_dDU = mPOI.dDP[0];
  m_dDUx = mPOI.dDP[1];
  m_dDUy = mPOI.dDP[2];
  m_dDV = mPOI.dDP[3];
  m_dDVx = mPOI.dDP[4];
  m_dDVy = mPOI.dDP[5];

  // Update the warp>
  m_dTemp = (1 + m_dDUx) * (1 + m_dDVy) - m_dDUy * m_dDVx;
  if (m_dTemp == 0) {
    mPOI.iInvertibleMatrix = 2;
    return mPOI;
  }
  // W(P) <- W(P) o W(DP)^-1.
  dWarp[0][0] = ((1 + m_dUx) * (1 + m_dDVy) - m_dUy * m_dDVx) / m_dTemp;
  dWarp[0][1] = (m_dUy * (1 + m_dDUx) - (1 + m_dUx) * m_dDUy) / m_dTemp;
  dWarp[0][2] = m_dU + (m_dUy * (m_dDU * m_dDVx - m_dDV - m_dDV * m_dDUx) - (1 + m_dUx) * (m_dDU * m_dDVy + m_dDU - m_dDUy * m_dDV)) / m_dTemp;
  dWarp[1][0] = (m_dVx * (1 + m_dDVy) - (1 + m_dVy) * m_dDVx) / m_dTemp;
  dWarp[1][1] = ((1 + m_dVy) * (1 + m_dDUx) - m_dVx * m_dDUy) / m_dTemp;
  dWarp[1][2] = m_dV + ((1 + m_dVy) * (m_dDU * m_dDVx - m_dDV - m_dDV * m_dDUx) - m_dVx * (m_dDU * m_dDVy + m_dDU - m_dDUy * m_dDV)) / m_dTemp;
  dWarp[2][0] = 0;
  dWarp[2][1] = 0;
  dWarp[2][2] = 1;


  // Update P.
  mPOI.dP[0] = dWarp[0][2];
  mPOI.dP[1] = dWarp[0][0] - 1;
  mPOI.dP[2] = dWarp[0][1];
  mPOI.dP[3] = dWarp[1][2];
  mPOI.dP[4] = dWarp[1][0];
  mPOI.dP[5] = dWarp[1][1] - 1;

  m_dU = mPOI.dP[0];
  m_dUx = mPOI.dP[1];
  m_dUy = mPOI.dP[2];
  m_dV = mPOI.dP[3];
  m_dVx = mPOI.dP[4];
  m_dVy = mPOI.dP[5];

  // Perform interative optimization, with pre-set maximum iteration step.
  while (mPOI.iIteration < mPOI.iMaxIteration &&
    sqrt(
      pow(mPOI.dDP[0], 2) + pow(mPOI.dDP[1] * iSubsetX, 2) +
        pow(mPOI.dDP[2] * iSubsetY, 2) + pow(mPOI.dDP[3], 2) +
        pow(mPOI.dDP[4] * iSubsetX, 2) + pow(mPOI.dDP[5] * iSubsetY, 2)) >= mPOI.dConvergeCriterion) {
  
    mPOI.iIteration++;
    
    // Fill warpped image into Subset T.
    m_dSubAveT = 0;
    m_dSubNorT = 0;
    for (l = 0; l < m_iSubsetH; l++) {
      for (m = 0; m < m_iSubsetW; m++) {
      
        // Calculate the location of warped subset T.
        dWarpX = mPOI.X + dWarp[0][0] * (m - iSubsetX) + dWarp[0][1] * (l - iSubsetY) + dWarp[0][2];
        dWarpY = mPOI.Y + dWarp[1][0] * (m - iSubsetX) + dWarp[1][1] * (l - iSubsetY) + dWarp[1][2];
        m_iTempX = int(dWarpX);
        m_iTempY = int(dWarpY);

        if ((m_iTempX >= 0) && (m_iTempY >= 0) && (m_iTempX < iROIWidth) && (m_iTempY < iROIHeight)) {
          m_dTempX = dWarpX - m_iTempX;
          m_dTempY = dWarpY - m_iTempY;
          
          // In most case, it is sub-pixel location, estimate the gary intensity using interpolation.
          dSubsetT[l][m] = 0;
          for (k = 0; k < 4; k++) {
            for (n = 0; n < 4; n++) {
              dSubsetT[l][m] += dTBicubic[m_iTempY][m_iTempX][k][n] * pow(m_dTempY, k) * pow(m_dTempX, n);
            }
          }
          m_dSubAveT += dSubsetT[l][m];
        } else {
          // If the loacation of the warped subset T is out of the ROI, stop iteration and set p as the current value.
          mPOI.iOutofROI = 1;
          return mPOI;
        }
      }
    }

    m_dSubAveT /= m_dSubsetSize;
    
    // Check if Subset T is a all dark subset
    if (m_dSubAveT == 0) {
      mPOI.iDarkSubset = 4;
      return mPOI;
    }

    for (l = 0; l < m_iSubsetH; l++) {
      for (m = 0; m < m_iSubsetW; m++) {
        dSubsetT[l][m] = dSubsetT[l][m] - m_dSubAveT; // T_i - T_m
        m_dSubNorT += pow(dSubsetT[l][m], 2); // Sigma(T_i - T_m)^2
      }
    }
    m_dSubNorT = sqrt(m_dSubNorT); // sqrt (Sigma(T_i - T_m)^2)
    if (m_dSubNorT == 0) {
      mPOI.iDarkSubset = 5; //set flag
      return mPOI;
    }

    // Compute the error image.
    for (k = 0; k < 6; k++) {
      dNumerator[k] = 0;
    }
    for (l = 0; l < m_iSubsetH; l++) {
      for (m = 0; m < m_iSubsetW; m++) {
        dError[l][m] = (m_dSubNorR / m_dSubNorT) * dSubsetT[l][m] - dSubsetR[l][m];

        // Compute the numerator.
        for (k = 0; k < 6; k++) {
          dNumerator[k] += (dRDescent[l][m][k] * dError[l][m]);
        }
      }
    }

    // Compute DeltaP.
    for (k = 0; k < 6; k++) {
      mPOI.dDP[k] = 0;
      for (n = 0; n < 6; n++) {
        mPOI.dDP[k] += (dInvHessian[k][n] * dNumerator[n]);
      }
    }
    m_dDU = mPOI.dDP[0];
    m_dDUx = mPOI.dDP[1];
    m_dDUy = mPOI.dDP[2];
    m_dDV = mPOI.dDP[3];
    m_dDVx = mPOI.dDP[4];
    m_dDVy = mPOI.dDP[5];

    // Update the warp.
    m_dTemp = (1 + m_dDUx) * (1 + m_dDVy) - m_dDUy * m_dDVx;
    if (m_dTemp == 0) {
      mPOI.iInvertibleMatrix = 2;
      return mPOI;
    }
    // W(P) <- W(P) o W(DP)^-1.
    dWarp[0][0] = ((1 + m_dUx) * (1 + m_dDVy) - m_dUy * m_dDVx) / m_dTemp;
    dWarp[0][1] = (m_dUy * (1 + m_dDUx) - (1 + m_dUx) * m_dDUy) / m_dTemp;
    dWarp[0][2] = m_dU + (m_dUy * (m_dDU * m_dDVx - m_dDV - m_dDV * m_dDUx) - (1 + m_dUx) * (m_dDU * m_dDVy + m_dDU - m_dDUy * m_dDV)) / m_dTemp;
    dWarp[1][0] = (m_dVx * (1 + m_dDVy) - (1 + m_dVy) * m_dDVx) / m_dTemp;
    dWarp[1][1] = ((1 + m_dVy) * (1 + m_dDUx) - m_dVx * m_dDUy) / m_dTemp;
    dWarp[1][2] = m_dV + ((1 + m_dVy) * (m_dDU * m_dDVx - m_dDV - m_dDV * m_dDUx) - m_dVx * (m_dDU * m_dDVy + m_dDU - m_dDUy * m_dDV)) / m_dTemp;
    dWarp[2][0] = 0;
    dWarp[2][1] = 0;
    dWarp[2][2] = 1;

    // Update P.
    mPOI.dP[0] = dWarp[0][2];
    mPOI.dP[1] = dWarp[0][0] - 1;
    mPOI.dP[2] = dWarp[0][1];
    mPOI.dP[3] = dWarp[1][2];
    mPOI.dP[4] = dWarp[1][0];
    mPOI.dP[5] = dWarp[1][1] - 1;

    m_dU = mPOI.dP[0];
    m_dUx = mPOI.dP[1];
    m_dUy = mPOI.dP[2];
    m_dV = mPOI.dP[3];
    m_dVx = mPOI.dP[4];
    m_dVy = mPOI.dP[5];
  }

  // Compute the ZNSSD.
  mPOI.ZNSSD = 0;
  for (l = 0; l < m_iSubsetH; l++) {
    for (m = 0; m < m_iSubsetW; m++) {
      mPOI.ZNSSD += pow((dError[l][m] / m_dSubNorR), 2);
    }
  }

  return mPOI;
}

void GradientX(
  double** dImage,
  double** dTx,
  double** dBuff,
  int iImgWidth,
  int iImgHeight,
  int iROIWidth,
  int iROIHeight,
  int iAccuracyOrder) {
  
  int i, j;
  switch (iAccuracyOrder) {
  case 2:
    for (i = floor(iAccuracyOrder / 2); i < iImgHeight - floor(iAccuracyOrder / 2); i++) {
      for (j = floor(iAccuracyOrder / 2); j < iImgWidth - floor(iAccuracyOrder / 2); j++) {
        dBuff[i][j] = 0.5*(dImage[i][j + 1] - dImage[i][j - 1]);
      }
    }
    break;
  case 4:
    for (i = floor(iAccuracyOrder / 2); i < iImgHeight - floor(iAccuracyOrder / 2); i++) {
      for (j = floor(iAccuracyOrder / 2); j < iImgWidth - floor(iAccuracyOrder / 2); j++) {
        dBuff[i][j] = 2.0 / 3 * dImage[i][j + 1] - 1.0 / 12 * dImage[i][j + 2] - 2.0 / 3 * dImage[i][j - 1] + 1.0 / 12 * dImage[i][j - 2];
      }
    }
    break;
  case 8:
    for (i = floor(iAccuracyOrder / 2); i < iImgHeight - floor(iAccuracyOrder / 2); i++) {
      for (j = floor(iAccuracyOrder / 2); j < iImgWidth - floor(iAccuracyOrder / 2); j++) {
        dBuff[i][j] = 4.0 / 105 * dImage[i][j + 3] - 1.0 / 280 * dImage[i][j + 4] + 4.0 / 5 * dImage[i][j + 1] - 1.0 / 5 * dImage[i][j + 2] - 4.0 / 5 * dImage[i][j - 1] + 1.0 / 5 * dImage[i][j - 2] - 4.0 / 105 * dImage[i][j - 3] + 1.0 / 280 * dImage[i][j - 4];
      }
    }
    break;
  default:
    for (i = floor(iAccuracyOrder / 2); i < iImgHeight - floor(iAccuracyOrder / 2); i++) {
      for (j = floor(iAccuracyOrder / 2); j < iImgWidth - floor(iAccuracyOrder / 2); j++) {
        dBuff[i][j] = 0.5*(dImage[i][j + 1] - dImage[i][j - 1]);
      }
    }
    break;
  }

  for (i = 0; i < iROIHeight; i++) {
    for (j = 0; j < iROIWidth; j++) {
      dTx[i][j] = dBuff[i + iAccuracyOrder][j + iAccuracyOrder];
    }
  }
}

void GradientY(
  double** dImage,
  double** dTy,
  double** dBuff,
  int iImgWidth,
  int iImgHeight,
  int iROIWidth,
  int iROIHeight,
  int iAccuracyOrder) {
  
  int i, j;
  switch (iAccuracyOrder) {
  case 2:
    for (i = floor(iAccuracyOrder / 2); i < iImgHeight - floor(iAccuracyOrder / 2); i++) {
      for (j = floor(iAccuracyOrder / 2); j < iImgWidth - floor(iAccuracyOrder / 2); j++) {
        dBuff[i][j] = 0.5 * (dImage[i + 1][j] - dImage[i - 1][j]);
      }
    }
    break;
  case 4:
    for (i = floor(iAccuracyOrder / 2); i < iImgHeight - floor(iAccuracyOrder / 2); i++) {
      for (j = floor(iAccuracyOrder / 2); j < iImgWidth - floor(iAccuracyOrder / 2); j++) {
        dBuff[i][j] = 2.0 / 3 * dImage[i + 1][j] - 1.0 / 12 * dImage[i + 2][j] - 2.0 / 3 * dImage[i - 1][j] + 1.0 / 12 * dImage[i - 2][j];
      }
    }
    break;
  case 8:
    for (i = floor(iAccuracyOrder / 2); i < iImgHeight - floor(iAccuracyOrder / 2); i++) {
      for (j = floor(iAccuracyOrder / 2); j < iImgWidth - floor(iAccuracyOrder / 2); j++) {
        dBuff[i][j] = 4.0 / 105 * dImage[i + 3][j] - 1.0 / 280 * dImage[i + 4][j] + 4.0 / 5 * dImage[i + 1][j] - 1.0 / 5 * dImage[i + 2][j] - 4.0 / 5 * dImage[i - 1][j] + 1.0 / 5 * dImage[i - 2][j] - 4.0 / 105 * dImage[i - 3][j] + 1.0 / 280 * dImage[i - 4][j];
      }
    }
    break;
  default:
    for (i = floor(iAccuracyOrder / 2); i < iImgHeight - floor(iAccuracyOrder / 2); i++) {
      for (j = floor(iAccuracyOrder / 2); j < iImgWidth - floor(iAccuracyOrder / 2); j++) {
        dBuff[i][j] = 0.5 * (dImage[i + 1][j] - dImage[i - 1][j]);
      }
    }
    break;
  }

  for (i = 0; i < iROIHeight; i++) {
    for (j = 0; j < iROIWidth; j++) {
      dTy[i][j] = dBuff[i + iAccuracyOrder][j + iAccuracyOrder];
    }
  }
}

void BiCubicCoefficient(
  double** dT,
  double** dTx,
  double** dTy,
  double** dTxy,
  int iROIWidth,
  int iROIHeight,
  double**** dTBicubic) {
  
  int i, j, l, k;

  // Compute the biubic interpolation coefficents of T.
  for (i = 0; i < iROIHeight - 1; i++) {
    for (j = 0; j < iROIWidth - 1; j++) {
      dTao[0] = dT[i][j];
      dTao[1] = dT[i][j + 1];
      dTao[2] = dT[i + 1][j];
      dTao[3] = dT[i + 1][j + 1];
      dTao[4] = dTx[i][j];
      dTao[5] = dTx[i][j + 1];
      dTao[6] = dTx[i + 1][j];
      dTao[7] = dTx[i + 1][j + 1];
      dTao[8] = dTy[i][j];
      dTao[9] = dTy[i][j + 1];
      dTao[10] = dTy[i + 1][j];
      dTao[11] = dTy[i + 1][j + 1];
      dTao[12] = dTxy[i][j];
      dTao[13] = dTxy[i][j + 1];
      dTao[14] = dTxy[i + 1][j];
      dTao[15] = dTxy[i + 1][j + 1];

      for (k = 0; k < 16; k++) {
        dAlpha[k] = 0;
        for (l = 0; l < 16; l++) {
          dAlpha[k] += (m_dBicubicMatrix[k][l] * dTao[l]);
        }
      }

      dTBicubic[i][j][0][0] = dAlpha[0];
      dTBicubic[i][j][0][1] = dAlpha[1];
      dTBicubic[i][j][0][2] = dAlpha[2];
      dTBicubic[i][j][0][3] = dAlpha[3];
      dTBicubic[i][j][1][0] = dAlpha[4];
      dTBicubic[i][j][1][1] = dAlpha[5];
      dTBicubic[i][j][1][2] = dAlpha[6];
      dTBicubic[i][j][1][3] = dAlpha[7];
      dTBicubic[i][j][2][0] = dAlpha[8];
      dTBicubic[i][j][2][1] = dAlpha[9];
      dTBicubic[i][j][2][2] = dAlpha[10];
      dTBicubic[i][j][2][3] = dAlpha[11];
      dTBicubic[i][j][3][0] = dAlpha[12];
      dTBicubic[i][j][3][1] = dAlpha[13];
      dTBicubic[i][j][3][2] = dAlpha[14];
      dTBicubic[i][j][3][3] = dAlpha[15];
    }
  }
}
