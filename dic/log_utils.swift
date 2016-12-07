//
//  log_utils.swift
//  padic
//
//  Created by Tue Le on 9/29/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

import Foundation
import Metal

func logFloatBuffer(buffer: MTLBuffer, fileName: String) {
  let length = buffer.length / 4
  let data = NSData(
    bytesNoCopy: buffer.contents(),
    length: length * 4,
    freeWhenDone: false)
  
  var array = [Float](repeating: 0, count: length)

  data.getBytes(&array, length: length * 4)

  var text = ""
  for i in 0 ..< length {
    text += String(NSString.init(format: "%.5f", array[i])) + "\n"
  }
  
  do {
    try text.write(
      to: Foundation.URL(string: "file:///Users/tue/workspace/fyp/padic/log/" + fileName)!,
      atomically: true,
      encoding: String.Encoding.utf8)
  } catch let error as NSError {
    print(error)
  }
}

func logUnsignedCharBuffer(buffer: MTLBuffer, fileName: String) {
  let length = buffer.length / 1
  let data = NSData(
    bytesNoCopy: buffer.contents(),
    length: length * 1,
    freeWhenDone: false)
  
  var array = [CUnsignedChar](repeating: 0, count: length)

  data.getBytes(&array, length: length * 1)

  var text = ""
  for i in 0 ..< length {
    text += String(NSString.init(format: "%d", Int(array[i]))) + "\n"
  }
  
  do {
    try text.write(
      to: Foundation.URL(string: "file:///Users/tue/workspace/fyp/padic/log/" + fileName)!,
      atomically: true,
      encoding: String.Encoding.utf8)
  } catch let error as NSError {
    print(error)
  }
}

func logIntBuffer(buffer: MTLBuffer, fileName: String) {
  let nbytes = buffer.length
  let n = nbytes / 4
  
  let data = NSData(
    bytesNoCopy: buffer.contents(),
    length: nbytes,
    freeWhenDone: false)
  
  var array = [Int32](repeating: 0, count: n)

  data.getBytes(&array, length: nbytes)

  var text = ""
  for i in 0 ..< n {
    text += String(NSString.init(format: "%d", Int(array[i]))) + "\n"
  }
  
  do {
    try text.write(
      to: Foundation.URL(string: "file:///Users/tue/workspace/fyp/padic/log/" + fileName)!,
      atomically: true,
      encoding: String.Encoding.utf8)
  } catch let error as NSError {
    print(error)
  }
}
