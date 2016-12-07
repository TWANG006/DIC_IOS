//
//  icgn_utils.swift
//  padic
//
//  Created by Tue Le on 9/29/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

import Foundation
import Metal

let BLOCK_SIZE_X = 16
let BLOCK_SIZE_Y = 16
let BLOCK_SIZE_64 = 64

func gradientXY2Images(
  // Device.
  device: MTLDevice,
  library: MTLLibrary,
  commandQueue: MTLCommandQueue,
  // Input.
  imgF: MTLBuffer,
  imgG: MTLBuffer,
  startX: Int,
  startY: Int,
  roiWidth: Int,
  roiHeight: Int,
  imgWidth: Int,
  imgHeight: Int,
  // Output.
  fx: MTLBuffer,
  fy: MTLBuffer,
  gx: MTLBuffer,
  gy: MTLBuffer,
  gxy: MTLBuffer
) {
  let commandBuffer = commandQueue.makeCommandBuffer()
  let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
  
  let f = library.makeFunction(name: "gradientXY2ImagesMetal")
  let computePipelineFilter = try? device.makeComputePipelineState(function: f!)
  computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
  
  computeCommandEncoder.setBuffer(imgF, offset: 0, at: 0)
  computeCommandEncoder.setBuffer(imgG, offset: 0, at: 1)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: startX), offset: 0, at: 2)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: startY), offset: 0, at: 3)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: roiWidth), offset: 0, at: 4)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: roiHeight), offset: 0, at: 5)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: imgWidth), offset: 0, at: 6)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: imgHeight), offset: 0, at: 7)
  computeCommandEncoder.setBuffer(fx, offset: 0, at: 8)
  computeCommandEncoder.setBuffer(fy, offset: 0, at: 9)
  computeCommandEncoder.setBuffer(gx, offset: 0, at: 10)
  computeCommandEncoder.setBuffer(gy, offset: 0, at: 11)
  computeCommandEncoder.setBuffer(gxy, offset: 0, at: 12)
  
  let threadsPerGroup = MTLSize(width: BLOCK_SIZE_X, height: BLOCK_SIZE_Y, depth: 1)
  let numThreadgroups = MTLSize(
    width: (roiWidth + 2 + BLOCK_SIZE_X - 3) / (BLOCK_SIZE_X - 2),
    height: (roiHeight + 2 + BLOCK_SIZE_Y - 3) / (BLOCK_SIZE_Y - 2),
    depth: 1)
  
  computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  computeCommandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
}

func bicubicCoefficients(
  // Device.
  device: MTLDevice,
  library: MTLLibrary,
  commandQueue: MTLCommandQueue,
  // Inputs.
  dIn_fImgT: MTLBuffer,
  dIn_fTx: MTLBuffer,
  dIn_fTy: MTLBuffer,
  dIn_fTxy: MTLBuffer,
  iStartX: Int,
  iStartY: Int,
  iROIWidth: Int,
  iROIHeight: Int,
  iImgWidth: Int,
  iImgHeight: Int,
  // Outputs.
  dOut_fBicubicInterpolants: MTLBuffer) {
  
  let commandBuffer = commandQueue.makeCommandBuffer()
  let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
  
  let f = library.makeFunction(name: "bicubicCoefficientsMetal")
  let computePipelineFilter = try? device.makeComputePipelineState(function: f!)
  computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
  
  computeCommandEncoder.setBuffer(dIn_fImgT, offset: 0, at: 0)
  computeCommandEncoder.setBuffer(dIn_fTx, offset: 0, at: 1)
  computeCommandEncoder.setBuffer(dIn_fTy, offset: 0, at: 2)
  computeCommandEncoder.setBuffer(dIn_fTxy, offset: 0, at: 3)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iStartX), offset: 0, at: 4)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iStartY), offset: 0, at: 5)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iROIWidth), offset: 0, at: 6)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iROIHeight), offset: 0, at: 7)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iImgWidth), offset: 0, at: 8)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iImgHeight), offset: 0, at: 9)
  computeCommandEncoder.setBuffer(dOut_fBicubicInterpolants, offset: 0, at: 10)
  
  let threadsPerGroup = MTLSize(width: BLOCK_SIZE_X, height: BLOCK_SIZE_Y, depth: 1)
  let numThreadgroups = MTLSize(
    width: (iROIWidth - 1) / BLOCK_SIZE_X + 1,
    height: (iROIHeight - 1) / BLOCK_SIZE_Y + 1,
    depth: 1)
  
  computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  computeCommandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
}

func refAllSubetsNorm(
  // Device.
  device: MTLDevice,
  library: MTLLibrary,
  commandQueue: MTLCommandQueue,
  // Inputs.
  d_refImg: MTLBuffer,
  d_iPOIXY: MTLBuffer,
  iSubsetW: Int,
  iSubsetH: Int,
  iSubsetX: Int,
  iSubsetY: Int,
  iImgWidth: Int,
  iImgHeight: Int,
  // Outputs.
  whole_dSubSet: MTLBuffer,
  whole_dSubsetAve: MTLBuffer,
  // For thread dim.
  poiNumber: Int) {
  
  let commandBuffer = commandQueue.makeCommandBuffer()
  let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
  
  let f = library.makeFunction(name: "refAllSubetsNormMetal")
  let computePipelineFilter = try? device.makeComputePipelineState(function: f!)
  computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
  
  computeCommandEncoder.setBuffer(d_refImg, offset: 0, at: 0)
  computeCommandEncoder.setBuffer(d_iPOIXY, offset: 0, at: 1)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetW), offset: 0, at: 2)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetH), offset: 0, at: 3)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetX), offset: 0, at: 4)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetY), offset: 0, at: 5)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iImgWidth), offset: 0, at: 6)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iImgHeight), offset: 0, at: 7)
  computeCommandEncoder.setBuffer(whole_dSubSet, offset: 0, at: 8)
  computeCommandEncoder.setBuffer(whole_dSubsetAve, offset: 0, at: 9)
  
  let threadsPerGroup = MTLSize(width: BLOCK_SIZE_64, height: 1, depth: 1)
  
  let numThreadgroups = MTLSize(width: poiNumber, height: 1, depth: 1)
  
  computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  computeCommandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
}

func inverseHessian(
  // Device.
  device: MTLDevice,
  library: MTLLibrary,
  commandQueue: MTLCommandQueue,
  // Inputs.
  d_Rx: MTLBuffer,
  d_Ry: MTLBuffer,
  d_iPOIXY: MTLBuffer,
  iSubsetX: Int,
  iSubsetY: Int,
  iSubsetW: Int,
  iSubsetH: Int,
  iStartX: Int,
  iStartY: Int,
  iROIWidth: Int,
  iROIHeight: Int,
  // Outputs.
  whole_d_RDescent: MTLBuffer,
  whole_d_InvHessian: MTLBuffer,
  // For thread dim.
  poiNumber: Int) {
  
  let commandBuffer = commandQueue.makeCommandBuffer()
  let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
  
  let f = library.makeFunction(name: "inverseHessianMetal")
  let computePipelineFilter = try? device.makeComputePipelineState(function: f!)
  computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
  
  computeCommandEncoder.setBuffer(d_Rx, offset: 0, at: 0)
  computeCommandEncoder.setBuffer(d_Ry, offset: 0, at: 1)
  computeCommandEncoder.setBuffer(d_iPOIXY, offset: 0, at: 2)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetX), offset: 0, at: 3)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetY), offset: 0, at: 4)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetW), offset: 0, at: 5)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetH), offset: 0, at: 6)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iStartX), offset: 0, at: 7)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iStartY), offset: 0, at: 8)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iROIWidth), offset: 0, at: 9)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iROIHeight), offset: 0, at: 10)
  computeCommandEncoder.setBuffer(whole_d_RDescent, offset: 0, at: 11)
  computeCommandEncoder.setBuffer(whole_d_InvHessian, offset: 0, at: 12)

  let threadsPerGroup = MTLSize(width: BLOCK_SIZE_64, height: 1, depth: 1)
  
  let numThreadgroups = MTLSize(width: poiNumber, height: 1, depth: 1)
  
  computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  computeCommandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
}

func toFloatArr(buffer: MTLBuffer, array: inout [Float]) {
  let nbytes = buffer.length
  let n = nbytes / 4
  let data = NSData(bytesNoCopy: buffer.contents(), length: nbytes, freeWhenDone: false)
  array = [Float](repeating: 0, count: n)
  data.getBytes(&array, length: nbytes)
}

func icgnCompute(
  // Device.
  device: MTLDevice,
  library: MTLLibrary,
  commandQueue: MTLCommandQueue,
  // Inputs and outputs.
  d_fU: MTLBuffer,
  d_fV: MTLBuffer,
  // Inputs.
  d_iPOIXY: MTLBuffer,
  iImgWidth: Int,
  iImgHeight: Int,
  iStartX: Int,
  iStartY: Int,
  iROIWidth: Int,
  iROIHeight: Int,
  iSubsetX: Int,
  iSubsetY: Int,
  iSubsetW: Int,
  iSubsetH: Int,
  iMaxIteration: Int,
  fDeltaP: Float,
  d_tarImg: MTLBuffer,
  whole_d_dInvHessian: MTLBuffer,
  m_dTBicubic: MTLBuffer,
  whole_d_2dRDescent: MTLBuffer,
  whole_d_dSubsetAveR: MTLBuffer,
  // Tempts.
  whole_d_dSubsetT: MTLBuffer,
  whole_d_dSubsetAveT: MTLBuffer,
  // Outputs.
  whole_d_iIteration: MTLBuffer,
  whole_d_dP: MTLBuffer,
  // For thread dim.
  poiNumber: Int) {
  
  let commandBuffer = commandQueue.makeCommandBuffer()
  let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
  
  let f = library.makeFunction(name: "icgnComputeMetal")
  let computePipelineFilter = try? device.makeComputePipelineState(function: f!)
  computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
  
  computeCommandEncoder.setBuffer(d_fU, offset: 0, at: 0)
  computeCommandEncoder.setBuffer(d_fV, offset: 0, at: 1)
  computeCommandEncoder.setBuffer(d_iPOIXY, offset: 0, at: 2)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iImgWidth), offset: 0, at: 3)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iImgHeight), offset: 0, at: 4)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iStartX), offset: 0, at: 5)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iStartY), offset: 0, at: 6)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iROIWidth), offset: 0, at: 7)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iROIHeight), offset: 0, at: 8)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetX), offset: 0, at: 9)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetY), offset: 0, at: 10)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetW), offset: 0, at: 11)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iSubsetH), offset: 0, at: 12)
  computeCommandEncoder.setBuffer(makeIntBuffer(device: device, x: iMaxIteration), offset: 0, at: 13)
  computeCommandEncoder.setBuffer(makeFloatBuffer(device: device, x: fDeltaP), offset: 0, at: 14)
  computeCommandEncoder.setBuffer(d_tarImg, offset: 0, at: 15)
  computeCommandEncoder.setBuffer(whole_d_dInvHessian, offset: 0, at: 16)
  computeCommandEncoder.setBuffer(m_dTBicubic, offset: 0, at: 17)
  computeCommandEncoder.setBuffer(whole_d_2dRDescent, offset: 0, at: 18)
  computeCommandEncoder.setBuffer(whole_d_dSubsetAveR, offset: 0, at: 19)
  computeCommandEncoder.setBuffer(whole_d_dSubsetT, offset: 0, at: 20)
  computeCommandEncoder.setBuffer(whole_d_dSubsetAveT, offset: 0, at: 21)
  computeCommandEncoder.setBuffer(whole_d_iIteration, offset: 0, at: 22)
  computeCommandEncoder.setBuffer(whole_d_dP, offset: 0, at: 23)
  
  let threadsPerGroup = MTLSize(width: BLOCK_SIZE_64, height: 1, depth: 1)
  
  let numThreadgroups = MTLSize(width: poiNumber, height: 1, depth: 1)
  
  computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  computeCommandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
}

func makeIntBuffer(device: MTLDevice, x: Int) -> MTLBuffer {
  return device.makeBuffer(bytes: [Int32(x)], length: 4, options: [])
}

func makeFloatBuffer(device: MTLDevice, x: Float) -> MTLBuffer {
  return device.makeBuffer(bytes: [Float32(x)], length: 4, options: [])
}
