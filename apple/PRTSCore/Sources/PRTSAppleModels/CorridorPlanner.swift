import Foundation
import prts_vlm

public struct PixelPoint: Codable, Equatable {
    public let x: Int32
    public let y: Int32
    public init(x: Int32, y: Int32) { self.x=x; self.y=y }
}

public struct CorridorPath {
    public let pixels: [PixelPoint]
    public let coverage: Double
    public let normalized: [[Double]]
    public let headingDegrees: Double?
    public let nearHeadingDegrees: Double?
    public let farHeadingDegrees: Double?
    public let checkedPixels: Int
    public let outsideMaskPixels: Int
    public var valid: Bool { checkedPixels > 0 && outsideMaskPixels == 0 }
}

/// Connected image-space candidate geometry. No GPS-to-image projection or
/// metric clearance is implied. The same C++ implementation is run in Windows
/// parity checks; the Swift/Apple build still needs the team's Xcode validation.
public enum CorridorPlanner {
    public static func search(freeMask: [UInt8], width: Int, height: Int,
                              anchorX: Int? = nil, preference: Int = 0,
                              prior: [PixelPoint] = []) throws -> CorridorPath {
        guard width >= 2, height >= 2, width <= 4096, height <= 4096,
              freeMask.count == width * height, (-1...1).contains(preference) else {
            throw NativeModelError.runtime("Invalid corridor mask or preference")
        }
        let priorXY=prior.flatMap { [$0.x,$0.y] }
        var output=[Int32](repeating:0,count:2*width*height)
        var coverage=0.0
        let count=freeMask.withUnsafeBufferPointer { mask in
            priorXY.withUnsafeBufferPointer { previous in
                output.withUnsafeMutableBufferPointer { result in
                    prts_corridor_search(mask.baseAddress,Int32(width),Int32(height),
                        Int32(anchorX ?? -1),Int32(preference),previous.baseAddress,Int32(prior.count),
                        result.baseAddress,Int32(width*height),&coverage)
                }
            }
        }
        guard count >= 0 else { throw NativeModelError.runtime("Corridor search failed: \(count)") }
        let xy=Array(output.prefix(Int(count)*2))
        var checked: Int32=0
        let outside=freeMask.withUnsafeBufferPointer { mask in
            xy.withUnsafeBufferPointer { points in
                prts_path_validate(mask.baseAddress,Int32(width),Int32(height),points.baseAddress,count,&checked)
            }
        }
        guard outside >= 0 else { throw NativeModelError.runtime("Corridor validation failed") }
        var angles=[Double](repeating:0,count:3)
        if count > 0 {
            let code=xy.withUnsafeBufferPointer { points in
                angles.withUnsafeMutableBufferPointer { values in
                    prts_path_heading(points.baseAddress,count,Int32(width),Int32(height),values.baseAddress)
                }
            }
            guard code == 0 else { throw NativeModelError.runtime("Corridor heading failed") }
        }
        let pixels=(0..<Int(count)).map { PixelPoint(x:xy[2*$0],y:xy[2*$0+1]) }
        return CorridorPath(pixels:pixels,coverage:coverage,
            normalized:pixels.map { [Double($0.x)/Double(width),Double($0.y)/Double(height)] },
            headingDegrees:count > 0 ? angles[0] : nil,
            nearHeadingDegrees:count > 0 ? angles[1] : nil,
            farHeadingDegrees:count > 0 ? angles[2] : nil,
            checkedPixels:Int(checked),outsideMaskPixels:Int(outside))
    }
}
