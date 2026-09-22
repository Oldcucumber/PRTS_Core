import CoreVideo
import Foundation
import PRTSContracts

public enum CameraAdapterError: Error { case requiresUprightBGRA }

public enum CameraAdapter {
    /// Frontend supplies upright BGRA. The returned RGB bytes own their storage.
    /// This explicit copy is the initial adapter, not a zero-copy claim.
    public static func frame(_ pixelBuffer: CVPixelBuffer, timestamp: SessionTime, sequence: UInt64,
                             spatial: SpatialSample? = nil) throws -> RGBFrame {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw CameraAdapterError.requiresUprightBGRA
        }
        CVPixelBufferLockBaseAddress(pixelBuffer,.readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer,.readOnly) }
        let width=CVPixelBufferGetWidth(pixelBuffer), height=CVPixelBufferGetHeight(pixelBuffer)
        let stride=CVPixelBufferGetBytesPerRow(pixelBuffer)
        let base=CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to:UInt8.self)
        var bytes=Data(count:width*height*3)
        bytes.withUnsafeMutableBytes { buffer in
            let rgb=buffer.bindMemory(to:UInt8.self)
            for y in 0..<height {
                for x in 0..<width {
                    let p=y*stride+x*4, q=(y*width+x)*3
                    rgb[q]=base[p+2]; rgb[q+1]=base[p+1]; rgb[q+2]=base[p]
                }
            }
        }
        return RGBFrame(timestamp:timestamp,sequence:sequence,width:width,height:height,
                        bytes:bytes,spatial:spatial)
    }
}
