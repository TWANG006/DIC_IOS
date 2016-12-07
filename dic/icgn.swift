//
//  ICGN.swift
//  padic
//
//  Created by Tue Le on 9/27/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

import Foundation
import Metal

class ICGN {

  // Parameters.
  private var m_iImgWidth: Int
  private var m_iImgHeight: Int
  private var m_iStartX: Int
  private var m_iStartY: Int
  private var m_iROIWidth: Int
  private var m_iROIHeight: Int
  private var m_iSubsetX: Int
  private var m_iSubsetY: Int
  private var m_iSubsetH: Int
  private var m_iSubsetW: Int
  private var m_iSubsetSize: Int
  private var m_iNumberX: Int
  private var m_iNumberY: Int
  private var m_iPOINumber: Int
  private var m_iNumIterations: Int
  private var m_fDeltaP: Float

  // Metal.
  private var device: MTLDevice
  private var commandQueue: MTLCommandQueue
  private var defaultLibrary: MTLLibrary
  
  // Buffers.
  private var m_d_fRefImg: MTLBuffer?
  private var m_d_fTarImg: MTLBuffer?
  private var m_d_iPOIXY: MTLBuffer?
  private var m_d_fU: MTLBuffer?
  private var m_d_fV: MTLBuffer?
  private var m_d_fRx: MTLBuffer
  private var m_d_fRy: MTLBuffer
  private var m_d_fTx: MTLBuffer
  private var m_d_fTy: MTLBuffer
  private var m_d_fTxy: MTLBuffer
  private var m_d_f4InterpolationLUT: MTLBuffer
  private var m_d_iIterationNums: MTLBuffer
  private var m_d_fSubsetR: MTLBuffer
  private var m_d_fSubsetT: MTLBuffer
  private var m_d_fSubsetAveR: MTLBuffer
  private var m_d_fSubsetAveT: MTLBuffer
  private var m_d_invHessian: MTLBuffer
  private var m_d_RDescent: MTLBuffer
  private var m_d_dP: MTLBuffer
  
  init(
    iImgWidth: Int,
    iImgHeight: Int,
    iStartX: Int,
    iStartY: Int,
    iROIWidth: Int,
    iROIHeight: Int,
    iSubsetX: Int,
    iSubsetY: Int,
    iNumberX: Int,
    iNumberY: Int,
    iNumIterations: Int,
    fDeltaP: Float) {
    
    // Params.
    m_iImgWidth = iImgWidth
    m_iImgHeight = iImgHeight
    m_iStartX = iStartX
    m_iStartY = iStartY
    m_iROIWidth = iROIWidth
    m_iROIHeight = iROIHeight
    m_iSubsetX = iSubsetX
    m_iSubsetY = iSubsetY
    m_iSubsetH = m_iSubsetY * 2 + 1
    m_iSubsetW = m_iSubsetX * 2 + 1
  	m_iSubsetSize = m_iSubsetH * m_iSubsetW
    m_iNumberX = iNumberX
    m_iNumberY = iNumberY
    m_iPOINumber = m_iNumberX * m_iNumberY;
    m_iNumIterations = iNumIterations
    m_fDeltaP = fDeltaP
 
    // Metal.
    device = MTLCreateSystemDefaultDevice()!
    commandQueue = device.makeCommandQueue()
    defaultLibrary = device.newDefaultLibrary()!
  
    // Buffers.
    let realSize = 4
    let real2Size = realSize * 2
    let real4Size = realSize * 4
    let intSize = 4
  
    m_d_fRx = device.makeBuffer(length: realSize * m_iROIWidth * m_iROIHeight, options: [])
    m_d_fRy = device.makeBuffer(length: realSize * m_iROIWidth * m_iROIHeight, options: [])
    
    m_d_fTx = device.makeBuffer(length: realSize * m_iROIWidth * m_iROIHeight, options: [])
    m_d_fTy = device.makeBuffer(length: realSize * m_iROIWidth * m_iROIHeight, options: [])
    m_d_fTxy = device.makeBuffer(length: realSize * m_iROIWidth * m_iROIHeight, options: [])
    
    m_d_f4InterpolationLUT = device.makeBuffer(length: real4Size * 4 * m_iROIWidth * m_iROIHeight, options: [])
    
    m_d_fSubsetR = device.makeBuffer(length: realSize * m_iPOINumber * m_iSubsetSize, options: [])
    m_d_fSubsetT = device.makeBuffer(length: realSize * m_iPOINumber * m_iSubsetSize, options: [])
    
    m_d_fSubsetAveR = device.makeBuffer(length: realSize * m_iPOINumber * (m_iSubsetSize + 1), options: [])
    m_d_fSubsetAveT = device.makeBuffer(length: realSize * m_iPOINumber * (m_iSubsetSize + 1), options: [])
    
    m_d_invHessian = device.makeBuffer(length: realSize * m_iPOINumber * 6 * 6, options: [])
    m_d_RDescent = device.makeBuffer(length: real2Size * 3 * m_iPOINumber * m_iSubsetSize, options: [])
    
    m_d_iIterationNums = device.makeBuffer(length: intSize * m_iPOINumber, options: []);
    m_d_dP = device.makeBuffer(length: realSize * m_iPOINumber * 6, options: []);
  }
  
  func compute(
    refImg: inout [CUnsignedChar],
    tarImg: inout [CUnsignedChar],
    poiXY: inout [Int32],
    fU: inout [Float],
    fV: inout [Float]) -> IcgnResult {
    
    // Allocate buffer for refImg, tarImg, poiXY, fU, fV.
    m_d_fRefImg = device.makeBuffer(bytes: &refImg, length: m_iImgWidth * m_iImgHeight, options: [])
    m_d_fTarImg = device.makeBuffer(bytes: &tarImg, length: m_iImgWidth * m_iImgHeight, options: [])
    
    m_d_iPOIXY = device.makeBuffer(bytes: poiXY, length: 4 * m_iPOINumber * 2, options: [])
    m_d_fU = device.makeBuffer(bytes: fU, length: 4 * m_iPOINumber, options: [])
    m_d_fV = device.makeBuffer(bytes: fV, length: 4 * m_iPOINumber, options: [])
    
		gradientXY2Images(
      device: device,
      library: defaultLibrary,
      commandQueue: commandQueue,
      imgF: m_d_fRefImg!,
      imgG: m_d_fTarImg!,
      startX: m_iStartX,
      startY: m_iStartY,
      roiWidth: m_iROIWidth,
      roiHeight: m_iROIHeight,
      imgWidth: m_iImgWidth,
      imgHeight: m_iImgHeight,
      fx: m_d_fRx,
      fy: m_d_fRy,
      gx: m_d_fTx,
      gy: m_d_fTy,
      gxy: m_d_fTxy)
    
		bicubicCoefficients(
      device: device,
      library: defaultLibrary,
      commandQueue: commandQueue,
      dIn_fImgT: m_d_fTarImg!,
      dIn_fTx: m_d_fTx,
      dIn_fTy: m_d_fTy,
      dIn_fTxy: m_d_fTxy,
      iStartX: m_iStartX,
      iStartY: m_iStartY,
      iROIWidth: m_iROIWidth,
      iROIHeight: m_iROIHeight,
      iImgWidth: m_iImgWidth,
      iImgHeight: m_iImgHeight,
      dOut_fBicubicInterpolants: m_d_f4InterpolationLUT)
    
    refAllSubetsNorm(
      device: device,
      library: defaultLibrary,
      commandQueue: commandQueue,
      d_refImg: m_d_fRefImg!,
      d_iPOIXY: m_d_iPOIXY!,
      iSubsetW: m_iSubsetW,
      iSubsetH: m_iSubsetH,
      iSubsetX: m_iSubsetX,
      iSubsetY: m_iSubsetY,
      iImgWidth: m_iImgWidth,
      iImgHeight: m_iImgHeight,
      whole_dSubSet: m_d_fSubsetR,
      whole_dSubsetAve: m_d_fSubsetAveR,
      poiNumber: m_iPOINumber)
    
    inverseHessian(
      device: device,
      library: defaultLibrary,
      commandQueue: commandQueue,
      d_Rx: m_d_fRx,
      d_Ry: m_d_fRy,
      d_iPOIXY: m_d_iPOIXY!,
      iSubsetX: m_iSubsetX,
      iSubsetY: m_iSubsetY,
      iSubsetW: m_iSubsetW,
      iSubsetH: m_iSubsetH,
      iStartX: m_iStartX,
      iStartY: m_iStartY,
      iROIWidth: m_iROIWidth,
      iROIHeight: m_iROIHeight,
      whole_d_RDescent: m_d_RDescent,
      whole_d_InvHessian: m_d_invHessian,
      poiNumber: m_iPOINumber)
  
    icgnCompute(
      device: device,
      library: defaultLibrary,
      commandQueue: commandQueue,
      d_fU: m_d_fU!,
      d_fV: m_d_fV!,
      d_iPOIXY: m_d_iPOIXY!,
      iImgWidth: m_iImgWidth,
      iImgHeight: m_iImgHeight,
      iStartX: m_iStartX,
      iStartY: m_iStartY,
      iROIWidth: m_iROIWidth,
      iROIHeight: m_iROIHeight,
      iSubsetX: m_iSubsetX,
      iSubsetY: m_iSubsetY,
      iSubsetW: m_iSubsetW,
      iSubsetH: m_iSubsetH,
      iMaxIteration: m_iNumIterations,
      fDeltaP: m_fDeltaP,
      d_tarImg: m_d_fTarImg!,
      whole_d_dInvHessian: m_d_invHessian,
      m_dTBicubic: m_d_f4InterpolationLUT,
      whole_d_2dRDescent: m_d_RDescent,
      whole_d_dSubsetAveR: m_d_fSubsetAveR,
      whole_d_dSubsetT: m_d_fSubsetT,
      whole_d_dSubsetAveT: m_d_fSubsetAveT,
      whole_d_iIteration: m_d_iIterationNums,
      whole_d_dP: m_d_dP,
      poiNumber: m_iPOINumber)
    
    let result = IcgnResult()
    result.imgWidth = m_iImgWidth
    result.imgHeight = m_iImgHeight
    result.numberX = m_iNumberX
    result.numberY = m_iNumberY
    result.poiXY = poiXY
    copyFloat(buffer: m_d_fU!, out: &(result.fU))
    copyFloat(buffer: m_d_fV!, out: &(result.fV))
    
    return result
  }
  
  private func copyFloat(buffer: MTLBuffer, out: inout [Float]) {
    let n = buffer.length / 4
    
    let data = NSData(
      bytesNoCopy: buffer.contents(),
      length: buffer.length, freeWhenDone: false)
    
    out = [Float](repeating: 0, count: n)
    
    data.getBytes(&out, length: buffer.length)
  }
}
