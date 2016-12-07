//
//  fftcc.hpp
//  dic
//
//  Created by Tue Le on 9/25/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef fftcc_hpp
#define fftcc_hpp

#include "fftw3.h"

#include "fftcc_types.h"

class FFTCC {
public:
  
  FFTCC(
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
  
  ~FFTCC();
  
  void resetRefImg(const unsigned char* refImg);

  void setTarImg(const unsigned char* tarImg);
  
  void initializeFFTCC(
    // Inputs
    const unsigned char* refImg,
    // Outputs
    int_t*& iPOIXY,
    real_t*& fU,
    real_t*& fV,
    real_t*& fZNCC);
  
	void algorithmFFTCC(
    // Inputs
    const unsigned char* tarImg,
    const int_t* iPOIXY,
    // Outputs
    real_t* fU,
    real_t* fV,
    real_t* fZNCC);
  
	void computeFFTCC(
    // Inputs
    const int_t *iPOIXY,
    const int id,
    // Outputs
    real_t &fU,
    real_t &fV,
    real_t &fZNCC);

	void finalizeFFTCC(
    int_t *& iPOIXY,
    real_t *& fU,
    real_t *& fV,
    real_t *& fZNCC);

  
	int_t getNumPOIsX() const { return m_iNumPOIX; }
	int_t getNumPOIsY() const { return m_iNumPOIY; }
	int_t getNumPOIs() const { return (m_iNumPOIX * m_iNumPOIY); }
	int_t getROISize() const { return (m_iROIWidth * m_iROIHeight); }
	int_t getImgSize() const { return (m_iImgHeight == -1 || m_iImgWidth == -1) ? getROISize() : m_iImgHeight * m_iImgWidth; }

private:
  bool recomputeNumPOI();
  
  bool m_isDestroyed;
  
	const unsigned char* m_refImg;
	const unsigned char* m_tarImg;

  // Whole Image Size
	int_t m_iImgWidth;
  int_t m_iImgHeight;
  
  // ROIsize
	int_t m_iROIWidth;
  int_t m_iROIHeight;
  
  // ROI top-left point
	int_t m_iStartX;
  int_t m_iStartY;
  
  // subsetSize = (2 * m_iSubsetX + 1) * (2 * m_iSubsetY + 1)
	int_t m_iSubsetX;
  int_t m_iSubsetY;
  
  // Number of pixels between each two POIs
	int_t m_iGridSpaceX;
  int_t m_iGridSpaceY;
  
  // Extra safe margin set for the ROI
	int_t m_iMarginX;
  int_t m_iMarginY;
  
  // Number of POIs = m_iNumPOIX * m_iNumPOIY
	int_t m_iNumPOIX;
  int_t m_iNumPOIY;
  int_t m_iPOINum;
  
  // FFT
	int_t m_iFFTSubsetW;
	int_t m_iFFTSubsetH;
  
  real_t *m_fSubset1;
  fftw3Complex *m_freqDom1;
  real_t *m_fSubset2;
  fftw3Complex *m_freqDom2;
  real_t *m_fSubsetC;
  fftw3Complex *m_freqDomC;

  fftw3Plan *m_fftwPlan1;
  fftw3Plan *m_fftwPlan2;
  fftw3Plan *m_rfftwPlan;
};

#endif /* fftcc_hpp */
