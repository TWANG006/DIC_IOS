//
//  FFTCCFacade.swift
//  dic
//
//  Created by Tue Le on 9/26/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

import Foundation

class FFTCCFacade {

  var fftcc: UnsafeMutableRawPointer?
  
  let imgWidth: Int
  let imgHeight: Int
  
  
  let poiXY: UnsafeMutablePointer<UnsafeMutablePointer<Int32>?> =
    UnsafeMutablePointer.allocate(capacity: 1)
  
  let fU: UnsafeMutablePointer<UnsafeMutablePointer<Double>?> =
    UnsafeMutablePointer.allocate(capacity: 1)
  
  let fV: UnsafeMutablePointer<UnsafeMutablePointer<Double>?> =
    UnsafeMutablePointer.allocate(capacity: 1)
  
  let fZNCC: UnsafeMutablePointer<UnsafeMutablePointer<Double>?> =
    UnsafeMutablePointer.allocate(capacity: 1)

  init(
    iImgWidth: Int,
    iImgHeight: Int,
    iROIWidth: Int,
    iROIHeight: Int,
    iStartX: Int,
    iStartY: Int,
    iSubsetX: Int,
    iSubsetY: Int,
    iGridSpaceX: Int,
    iGridSpaceY: Int,
    iMarginX: Int,
    iMarginY: Int) {
    
    imgWidth = iImgWidth
    imgHeight = iImgHeight
    
    fftcc = newFFTCC(
      Int32(iImgWidth),
      Int32(iImgHeight),
      Int32(iROIWidth),
      Int32(iROIHeight),
      Int32(iStartX),
      Int32(iStartY),
      Int32(iSubsetX),
      Int32(iSubsetY),
      Int32(iGridSpaceX),
      Int32(iGridSpaceY),
      Int32(iMarginX),
      Int32(iMarginY)
    )
  }
  
  func initialize(refImg: inout [CUnsignedChar]) {
    initializeFFTCC(fftcc, &refImg, poiXY, fU, fV, fZNCC)
  }
  
  func compute(tarImg: inout [CUnsignedChar]) {
    algorithmFFTCC(fftcc, &tarImg, poiXY.pointee, fU.pointee, fV.pointee, fZNCC.pointee)
  }
  
  func getNumPoiX() -> Int {
    return Int(getNumPoiXFFTCC(fftcc))
  }
  
  func getNumPoiY() -> Int {
    return Int(getNumPoiYFFTCC(fftcc))
  }
  
  func getFU(poiX: Int, poiY: Int) -> Double {
    let poiId = getPoiId(poiX: poiX, poiY: poiY)
    return (fU.pointee?.advanced(by: poiId).pointee)!
  }
  
  func getFV(poiX: Int, poiY: Int) -> Double {
    let poiId = getPoiId(poiX: poiX, poiY: poiY)
    return (fV.pointee?.advanced(by: poiId).pointee)!
  }
  
  func getPoiPos(poiX: Int, poiY: Int) -> (Int, Int) {
    let poiId = getPoiId(poiX: poiX, poiY: poiY)
    let y = Int((poiXY.pointee?.advanced(by: poiId * 2).pointee)!)
    let x = Int((poiXY.pointee?.advanced(by: poiId * 2 + 1).pointee)!)
    
    return (x, y)
  }
  
  func getPoiId(poiX: Int, poiY: Int) -> Int {
    let numPoiX = getNumPoiX()
    return poiX + numPoiX * poiY
  }
  
  func getImgWidth() -> Int {
    return imgWidth
  }
  
  func getImgHeight() -> Int {
    return imgHeight
  }
  
  static func run(refImg: inout [CUnsignedChar], tarImg: inout [CUnsignedChar], imgWidth: Int, imgHeight: Int) -> FFTCCFacade {
    let cropX = 50
    let cropY = 50
    let roiWidth = imgWidth - cropX * 2
    let roiHeight = imgHeight - cropY * 2
    let startX = cropX
    let startY = cropY
    let subsetX = 15
    let subsetY = 15
    let gridSpaceX = 40
    let gridSpaceY = 40
    let marginX = 10
    let marginY = 10
    
    let f = FFTCCFacade(
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
    
    f.initialize(refImg: &refImg)
    f.compute(tarImg: &tarImg)
    
    return f
  }
}
