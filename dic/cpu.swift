//
//  AlgoFacade.swift
//  dic
//
//  Created by Tue Le on 9/13/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

import Foundation

class PoiFacade {
  private var poi: UnsafeMutablePointer<POI>
  
  init(poi: UnsafeMutablePointer<POI>) {
    self.poi = poi
  }
  
  var x: Int {
    return Int(poiGetX(poi))
  }
  
  var y: Int {
    return Int(poiGetY(poi))
  }
  
  var u: Double {
    return poiGetU(poi)
  }
  
  var v: Double {
    return poiGetV(poi)
  }
}

class ResultFacade {
  private var result: UnsafeMutablePointer<RunResult>
  
  init(result: UnsafeMutablePointer<RunResult>) {
    self.result = result
  }
  
  var numberX: Int {
    return Int(runResultGetNumberX(result))
  }
  
  var numberY: Int {
    return Int(runResultGetNumberY(result))
  }
  
  var cropMarginX: Int {
    return Int(runResultGetCropMarginX(result))
  }
  
  var cropMarginY: Int {
    return Int(runResultGetCropMarginY(result))
  }
  
  var imgWidth: Int {
    return Int(runResultGetImgWidth(result))
  }
  
  var imgHeight: Int {
    return Int(runResultGetImgHeight(result))
  }
  
  func poi(x: Int, y: Int) -> PoiFacade {
    return PoiFacade(poi: runResultGetPOI(result, Int32(x), Int32(y)))
  }
}

class Cpu {
  static func run(
    ref: inout [Double],
    tar: inout [Double],
    width: Int,
    height: Int,
    accuracyOrder: Int,
    subsetX: Int,
    subsetY: Int,
    marginX: Int,
    marginY: Int,
    gridSpaceX: Int,
    gridSpaceY: Int,
    maxIter: Int,
    deltaP: Double) -> IcgnResult {
    
    let start = clock()
    
    let facade = ResultFacade(
      result: bridge(
        &ref,
        &tar,
        Int32(width),
        Int32(height),
        Int32(accuracyOrder),
        Int32(subsetX),
        Int32(subsetY),
        Int32(marginX),
        Int32(marginY),
        Int32(gridSpaceX),
        Int32(gridSpaceY),
        Int32(maxIter),
        deltaP
      )
    )
      
    let result = IcgnResult()
    result.imgWidth = width
    result.imgHeight = height
    result.numberX = facade.numberX
    result.numberY = facade.numberY
    
    for y in 0 ..< result.numberY {
      for x in 0 ..< result.numberX {
        let poi = facade.poi(x: x, y: y)
        result.poiXY.append(Int32(poi.y))
        result.poiXY.append(Int32(poi.x))
        result.fU.append(Float(poi.u))
        result.fV.append(Float(poi.v))
      }
    }
    
    print(result.numberX * result.numberY)
    
    print(Double(clock() - start) / Double(CLOCKS_PER_SEC))
    
    return result
  }
}
