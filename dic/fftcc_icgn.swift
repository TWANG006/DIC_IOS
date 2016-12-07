//
//  fftcc_icgn.swift
//  padic
//
//  Created by Tue Le on 10/3/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

import Foundation

func fftccIcgn(
  imgWidth: Int,
  imgHeight: Int,
  cropX: Int,
  cropY: Int,
  startX: Int,
  startY: Int,
  subsetX: Int,
  subsetY: Int,
  marginX: Int,
  marginY: Int,
  gridSpaceX: Int,
  gridSpaceY: Int,
  numIter: Int,
  deltaP: Float,
  refImg: inout [CUnsignedChar],
  tarImg: inout [CUnsignedChar]) -> IcgnResult {
  
  let start = clock()
  
  let roiWidth = imgWidth - cropX * 2
  let roiHeight = imgHeight - cropY * 2

  // FFTCC.
  let fftcc = FFTCCFacade(
    iImgWidth: imgWidth,
    iImgHeight: imgHeight,
    iROIWidth: roiWidth,
    iROIHeight: roiHeight,
    iStartX: startX,
    iStartY: startY,
    iSubsetX: subsetX,
    iSubsetY: subsetY,
    iGridSpaceX: gridSpaceX,
    iGridSpaceY: gridSpaceY,
    iMarginX: marginX,
    iMarginY: marginY)

  fftcc.initialize(refImg: &refImg)
  fftcc.compute(tarImg: &tarImg)

  // ICGN.
  let numberX = fftcc.getNumPoiX()
  let numberY = fftcc.getNumPoiY()
  var poiXY: [Int32] = []
  var fU: [Float] = []
  var fV: [Float] = []

  for poiY in 0 ..< numberY {
    for poiX in 0 ..< numberX {
      let (x, y) = fftcc.getPoiPos(poiX: poiX, poiY: poiY)
      poiXY.append(Int32(y))
      poiXY.append(Int32(x))
      let u = fftcc.getFU(poiX: poiX, poiY: poiY)
      let v = fftcc.getFV(poiX: poiX, poiY: poiY)
//      u = 0
//      v = 0
//      print(String(format: "%d %d %.5lf %.5f", Int(x), Int(y), Float(u), Float(v)))
      fV.append(Float(v))
      fU.append(Float(u))
    }
  }

  let icgn = ICGN(
    iImgWidth: imgWidth,
    iImgHeight: imgHeight,
    iStartX: startX,
    iStartY: startY,
    iROIWidth: roiWidth,
    iROIHeight: roiHeight,
    iSubsetX: subsetX,
    iSubsetY: subsetY,
    iNumberX: numberX,
    iNumberY: numberY,
    iNumIterations: numIter,
    fDeltaP: Float(deltaP))

  let result = icgn.compute(
    refImg: &refImg,
    tarImg: &tarImg,
    poiXY: &poiXY,
    fU: &fU,
    fV: &fV)
  
  print(Double(clock() - start) / Double(CLOCKS_PER_SEC))
  
  return result
}
