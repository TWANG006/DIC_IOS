//
//  ViewController.swift
//  dic
//
//  Created by Tue Le on 9/4/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  override func viewDidLoad() {
    super.viewDidLoad()
    // Do any additional setup after loading the view, typically from a nib.
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }
  
  @IBOutlet weak var runningLabel: UILabel!
  @IBOutlet weak var refImageView: UIImageView!
  @IBOutlet weak var tarImageView: UIImageView!
  @IBOutlet weak var runButton: UIButton!
  
  @IBOutlet weak var cpuResultView: ResultView!
  @IBOutlet weak var gpuResultView: ResultView!
  
  @IBAction func selectImageRef(_ sender: UIButton) {
    selectingRefOpt = true
    selectImage()
  }
  
  @IBAction func selectImageTar(_ sender: UIButton) {
    selectingRefOpt = false
    selectImage()
  }
  
  @IBAction func run(_ sender: UIButton) {
    if let ref = refOpt {
      if let tar = tarOpt {
        DispatchQueue.global().async {
          let cpuResult = self.runCpu(refImg: ref, tarImg: tar)
          DispatchQueue.main.async {
            self.displayCpuResult(result: cpuResult)
          }
          let gpuResult = self.runGpu(refImg: ref, tarImg: tar)
          DispatchQueue.main.async {
            self.displayGpuresult(result: gpuResult)
          }
        }
      }
    }
  }
  
  private func runCpu(refImg: UIImage, tarImg: UIImage) -> IcgnResult {
  
    print("CPU")
    
    let imgWidth: Int = Int(refImg.size.width)
    let imgHeight: Int = Int(refImg.size.height)
    
    var ref = refImg.greyscale()
    var tar = tarImg.greyscale()
    
    return Cpu.run(
      ref: &ref,
      tar: &tar,
      width: imgWidth,
      height: imgHeight,
      accuracyOrder: DefaultParams.cropX,
      subsetX: DefaultParams.subsetX,
      subsetY: DefaultParams.subsetY,
      marginX: DefaultParams.marginX,
      marginY: DefaultParams.marginY,
      gridSpaceX: DefaultParams.gridSpaceX,
      gridSpaceY: DefaultParams.gridSpaceY,
      maxIter: DefaultParams.numIter,
      deltaP: DefaultParams.deltaP)
  }
  
  private func runGpu(refImg: UIImage, tarImg: UIImage) -> IcgnResult {
  
    print("GPU")
    
    let imgWidth: Int = Int(refImg.size.width)
    let imgHeight: Int = Int(refImg.size.height)
    
    var ref = refImg.greyscaleUChar()
    var tar = tarImg.greyscaleUChar()
    
    return fftccIcgn(
      imgWidth: imgWidth,
      imgHeight: imgHeight,
      cropX: DefaultParams.cropX,
      cropY: DefaultParams.cropY,
      startX: DefaultParams.startX,
      startY: DefaultParams.startY,
      subsetX: DefaultParams.subsetX,
      subsetY: DefaultParams.subsetY,
      marginX: DefaultParams.marginX,
      marginY: DefaultParams.marginY,
      gridSpaceX: DefaultParams.gridSpaceX,
      gridSpaceY: DefaultParams.gridSpaceY,
      numIter: DefaultParams.numIter,
      deltaP: Float(DefaultParams.deltaP),
      refImg: &ref,
      tarImg: &tar)
  }
  
  private func displayCpuResult(result: IcgnResult) {
    cpuResultView.setResult(result: result)
  }
  
  private func displayGpuresult(result: IcgnResult) {
    gpuResultView.setResult(result: result)
  }
  
  private func selectImage() {
    let imagePicker = UIImagePickerController()
    imagePicker.delegate = self
    imagePicker.sourceType = .photoLibrary
    self.present(imagePicker, animated: true, completion: nil)
  }
  
  public func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
    
    let imageOpt: UIImage? = info["UIImagePickerControllerOriginalImage"] as? UIImage
    
    if let image = imageOpt {
      if let selectingRef = selectingRefOpt {
        if selectingRef {
          refOpt = image
          refImageView.image = image
        } else {
          tarOpt = image
          tarImageView.image = image
        }
      }
    }
    
    self.dismiss(animated: false, completion: nil)
  }
  
  private var refOpt: UIImage?
  private var tarOpt: UIImage?
  private var selectingRefOpt: Bool?
}

extension UIImage {
  func greyscale() -> [Double] {
    
    let pixelData = self.cgImage!.dataProvider!.data
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
  
    let width = Int(self.size.width)
    let height = Int(self.size.height)
  
    var result: [Double] = []
    print(width, height)
  
    for y in 0...height {
      for x in 0...width {
        let i = (width * y + x) * 4
        
        let r = Double(data[i])
        let g = Double(data[i + 1])
        let b = Double(data[i + 2])
        let grey = (r + g + b) / 3
        
        result.append(grey)
      }
    }
  
    return result
  }
  
  func greyscaleUChar() -> [CUnsignedChar] {
    let pixelData = self.cgImage!.dataProvider!.data
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
  
    let width = Int(self.size.width)
    let height = Int(self.size.height)
  
    var result: [CUnsignedChar] = []
  
    for y in 0...height {
      for x in 0...width {
        let pixelInfo = (width * y + x) * 4
        let r = Double(data[pixelInfo])
        let g = Double(data[pixelInfo + 1])
        let b = Double(data[pixelInfo + 2])
        let grey = (r + g + b) / 3
        
        result.append(CUnsignedChar(grey))
      }
    }
  
    return result
  }
}
