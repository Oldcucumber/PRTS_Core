import Foundation
import PRTSContracts
import prts_vlm

public struct ObjectDetection: Codable {
    public let classID: Int
    public let label: String
    public let score: Float
    public let box: [Float] // x1,y1,x2,y2, normalized original upright image
    public let blocking: Bool
}

public struct SemanticGrid {
    public let width: Int
    public let height: Int
    public let classIDs: [UInt8]
    public let classNames: [Int:String]
    public let walkProbability: [Float]
    public let confidence: [Float]
    public let walkable: [UInt8]
}

public struct PerceptionObservation {
    public let frameSequence: UInt64
    public let timestamp: SessionTime
    public let semantic: SemanticGrid
    public let detections: [ObjectDetection]
    public let processingSeconds: Double
}

private struct PerceptionManifest: Decodable {
    let input_shape: [Int]
    let labels: [String:String]
    let mean: [Float]?
    let std: [Float]?
    let output: String?
}

/// Actual ONNX inference and image-space decoding. Dense class IDs are the
/// authoritative overlay; the frontend may render them directly as a texture.
/// C ABI and numeric operations have Windows checks. Apple build, CoreML and
/// end-to-end preprocessing quality are tracked separately in the report.
public final class NativePerception {
    private let semanticModel: ONNXModel
    private let detectorModel: ONNXModel
    private let semanticMeta: PerceptionManifest
    private let detectorMeta: PerceptionManifest
    private let labels: [Int:String]
    private let walkIDs: [Int32]
    public init(semantic: URL, semanticManifest: URL, detector: URL, detectorManifest: URL,
                useCoreML: Bool = false) throws {
        let decoder=JSONDecoder()
        semanticMeta=try decoder.decode(PerceptionManifest.self,from:Data(contentsOf:semanticManifest))
        detectorMeta=try decoder.decode(PerceptionManifest.self,from:Data(contentsOf:detectorManifest))
        guard semanticMeta.input_shape.count == 4, detectorMeta.input_shape.count == 4,
              semanticMeta.output?.contains("probabilities") == true else {
            throw NativeModelError.runtime("Expected the exported Mask2Former probability manifest")
        }
        labels=Dictionary(uniqueKeysWithValues:semanticMeta.labels.compactMap { key,value in
            Int(key).map { ($0,value) }
        })
        let walkNames:Set<String>=["sidewalk","flat-sidewalk","pedestrian area","paved trail","other walkable surface"]
        walkIDs=labels.filter { walkNames.contains($0.value.lowercased()) }.map { Int32($0.key) }.sorted()
        semanticModel=try ONNXModel(model:semantic,useCoreML:useCoreML)
        detectorModel=try ONNXModel(model:detector,useCoreML:useCoreML)
    }

    private func prepare(_ frame: RGBFrame, _ meta: PerceptionManifest) throws -> (FloatTensor,[Int32]) {
        let h=meta.input_shape[2],w=meta.input_shape[3]
        let mean=meta.mean ?? [0,0,0],std=meta.std ?? [1,1,1]
        guard h > 0, w > 0, h <= 4096, w <= 4096, mean.count == 3, std.count == 3 else {
            throw NativeModelError.runtime("Invalid image normalization manifest")
        }
        var output=[Float](repeating:0,count:3*w*h),crop=[Int32](repeating:0,count:4)
        let code=frame.bytes.withUnsafeBytes { input in
            mean.withUnsafeBufferPointer { m in std.withUnsafeBufferPointer { s in
                output.withUnsafeMutableBufferPointer { values in crop.withUnsafeMutableBufferPointer { rect in
                    prts_letterbox_rgb(input.bindMemory(to:UInt8.self).baseAddress,Int32(frame.width),Int32(frame.height),
                        Int32(w),Int32(h),m.baseAddress,s.baseAddress,values.baseAddress,rect.baseAddress)
                }}
            }}
        }
        guard code == 0 else { throw NativeModelError.runtime("Image preprocessing failed") }
        return (FloatTensor(values:output,shape:[1,3,Int64(h),Int64(w)]),crop)
    }

    public func observe(_ frame: RGBFrame) throws -> PerceptionObservation {
        let start=ProcessInfo.processInfo.systemUptime
        let (input,crop)=try prepare(frame,semanticMeta),scores=try semanticModel.run(input)
        guard scores.shape.count == 4, scores.shape[0] == 1 else { throw NativeModelError.runtime("Invalid semantic tensor shape") }
        let nc=Int(scores.shape[1]),sh=Int(scores.shape[2]),sw=Int(scores.shape[3])
        guard scores.values.count == nc*sh*sw else { throw NativeModelError.runtime("Invalid semantic tensor storage") }
        let scale=min(1.0,320.0/Double(max(frame.width,frame.height)))
        let w=Int((Double(frame.width)*scale).rounded(.toNearestOrEven)),h=Int((Double(frame.height)*scale).rounded(.toNearestOrEven))
        var classes=[UInt8](repeating:0,count:w*h),walkable=classes
        var walk=[Float](repeating:0,count:w*h),confidence=walk
        let code=scores.values.withUnsafeBufferPointer { values in crop.withUnsafeBufferPointer { rect in
            walkIDs.withUnsafeBufferPointer { walkIDs in classes.withUnsafeMutableBufferPointer { classes in
                walk.withUnsafeMutableBufferPointer { walk in confidence.withUnsafeMutableBufferPointer { confidence in
                    walkable.withUnsafeMutableBufferPointer { mask in
                        prts_decode_semantic(values.baseAddress,Int32(nc),Int32(sw),Int32(sh),rect.baseAddress,
                            Int32(semanticMeta.input_shape[3]),Int32(semanticMeta.input_shape[2]),Int32(w),Int32(h),
                            walkIDs.baseAddress,Int32(walkIDs.count),classes.baseAddress,walk.baseAddress,confidence.baseAddress,mask.baseAddress)
                    }
                }}
            }}
        }}
        guard code == 0 else { throw NativeModelError.runtime("Semantic decoding failed") }
        let grid=SemanticGrid(width:w,height:h,classIDs:classes,classNames:labels,
            walkProbability:walk,confidence:confidence,walkable:walkable)
        return PerceptionObservation(frameSequence:frame.sequence,timestamp:frame.timestamp,semantic:grid,
            detections:try detect(frame),processingSeconds:ProcessInfo.processInfo.systemUptime-start)
    }

    /// Independent lightweight detector, usable by waiting tasks between dense frames.
    public func detect(_ frame: RGBFrame) throws -> [ObjectDetection] {
        let (input,crop)=try prepare(frame,detectorMeta),raw=try detectorModel.run(input)
        guard raw.shape.count == 3, raw.shape[0] == 1, raw.shape[1] > 4 else {
            throw NativeModelError.runtime("Expected YOLO [1,4+classes,anchors]")
        }
        let n=Int(raw.shape[2]),nc=Int(raw.shape[1])-4
        guard raw.values.count == (nc+4)*n else { throw NativeModelError.runtime("Invalid detector tensor storage") }
        struct Box { let index:Int; let cid:Int; let score:Float; let xyxy:[Float] }
        var candidates=[Box]()
        for i in 0..<n {
            var best:Float = -.infinity,cid=0
            for c in 0..<nc { let score=raw.values[(c+4)*n+i]; if score > best { best=score;cid=c } }
            if best < 0.25 { continue }
            let x=raw.values[i],y=raw.values[n+i],w=raw.values[2*n+i],h=raw.values[3*n+i]
            candidates.append(Box(index:i,cid:cid,score:best,xyxy:[x-w/2,y-h/2,x+w/2,y+h/2]))
        }
        candidates.sort { $0.score == $1.score ? $0.index < $1.index : $0.score > $1.score }
        func overlap(_ a:[Float],_ b:[Float]) -> Float {
            let intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            let union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            return union > 0 ? intersection/union : 0
        }
        var kept=[Box]()
        for item in candidates {
            if !kept.contains(where: { $0.cid == item.cid && overlap($0.xyxy,item.xyxy) > 0.55 }) { kept.append(item) }
        }
        return kept.map { item in
            let b=item.xyxy,x=Float(crop[0]),y=Float(crop[1]),w=Float(crop[2]),h=Float(crop[3])
            let normalized=[(b[0]-x)/w,(b[1]-y)/h,(b[2]-x)/w,(b[3]-y)/h].map { min(1,max(0,$0)) }
            return ObjectDetection(classID:item.cid,label:detectorMeta.labels[String(item.cid)] ?? "unknown",
                score:item.score,box:normalized,blocking:item.cid != 9 && item.cid != 11)
        }
    }
}
