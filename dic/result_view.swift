//
//  ResultView.swift
//  dic
//
//  Created by Tue Le on 9/13/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

import UIKit

class ResultView: UIView {

  override func draw(_ rect: CGRect) {
    if let result = resultOpt {
      let width = Double(bounds.size.width)
      let height = Double(bounds.size.height)
      
      let imgWidth = Double(result.imgWidth)
      let imgHeight = Double(result.imgHeight)
      
      let fitWidth = min(width, height * imgWidth / imgHeight)
      let fitHeight = min(height, width * imgHeight / imgWidth)
      
      let scaleX = fitWidth / imgWidth
      let scaleY = fitHeight / imgHeight
      
      let numberX = result.numberX
      let numberY = result.numberY
      
      for x in 0..<numberX {
        for y in 0..<numberY {
          let poiY = result.poiXY[y * numberX * 2 + x * 2 + 0]
          let poiX = result.poiXY[y * numberX * 2 + x * 2 + 1]
          let poiU = result.fU[y * numberX + x]
          let poiV = result.fV[y * numberX + x]
          
          let fromX = Double(poiX)
          let fromY = Double(poiY)
          let toX = fromX + Double(poiU) * 10
          let toY = fromY + Double(poiV) * 10
          
          let from = CGPoint(x: fromX * scaleX, y: fromY * scaleY)
          let to = CGPoint(x: toX * scaleX, y: toY * scaleY)
          
          drawLine(from: from, to: to)
        }
      }
    }
  }
  
  func setResult(result: IcgnResult) {
    resultOpt = result
    setNeedsDisplay()
  }
  
  private func drawLine(from: CGPoint, to: CGPoint) {
    let path = UIBezierPath()
    path.move(to: from)
    path.addLine(to: to)
    UIColor.darkGray.set()
    path.stroke()
    path.fill()
  }
  
  var resultOpt: IcgnResult?
}
