import CoreGraphics
import Foundation
import PRTSContracts
import Vision

extension RGBFrame {
    public func cgImage() throws -> CGImage {
        guard let provider=CGDataProvider(data:bytes as CFData),
              let image=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:24,
                bytesPerRow:width*3,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo(rawValue:0),
                provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else {
            throw NativeModelError.runtime("Cannot wrap the upright RGB buffer")
        }
        return image
    }
    public func cropped(to box: [Float]) -> (RGBFrame,[Float])? {
        guard box.count == 4 else { return nil }
        let x1=max(0,min(width-1,Int(floor(Double(box[0])*Double(width)))))
        let y1=max(0,min(height-1,Int(floor(Double(box[1])*Double(height)))))
        let x2=max(0,min(width,Int(ceil(Double(box[2])*Double(width)))))
        let y2=max(0,min(height,Int(ceil(Double(box[3])*Double(height)))))
        guard x2 > x1,y2 > y1 else { return nil }
        let w=x2-x1,h=y2-y1
        var result=Data(count:w*h*3)
        result.withUnsafeMutableBytes { target in bytes.withUnsafeBytes { source in
            for y in 0..<h {
                target.baseAddress!.advanced(by:y*w*3).copyMemory(
                    from:source.baseAddress!.advanced(by:((y+y1)*width+x1)*3),byteCount:w*3)
            }
        }}
        return (RGBFrame(timestamp:timestamp,sequence:sequence,width:w,height:h,bytes:result),
                [Float(x1)/Float(width),Float(y1)/Float(height),Float(x2)/Float(width),Float(y2)/Float(height)])
    }
}

/// Apple offline OCR backend. Its model/revision differs from RapidOCR; all bus
/// and notice replays must be re-run on the target OS before claiming parity.
public final class VisionTextReader {
    public init() {}
    public func read(_ frame: RGBFrame) throws -> [TextEvidence] {
        let request=VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection=false
        request.recognitionLanguages=["zh-Hans","zh-Hant","en-US"]
        let supported=try request.supportedRecognitionLanguages()
        request.recognitionLanguages=request.recognitionLanguages.filter { supported.contains($0) }
        try VNImageRequestHandler(cgImage:frame.cgImage(),orientation:.up,options:[:]).perform([request])
        return (request.results ?? []).compactMap { result in
            guard let text=result.topCandidates(1).first else { return nil }
            let corners=[result.topLeft,result.topRight,result.bottomRight,result.bottomLeft]
            let box=corners.map { [Float($0.x)*Float(frame.width),Float(1-$0.y)*Float(frame.height)] }
            return TextEvidence(text:text.string,score:text.confidence,box:box)
        }
    }
}

public struct VehicleEvidenceResult {
    public let texts: [TextEvidence]
    public let vehicles: [VehicleObservation]
}

/// Exact route reading never receives the user's requested number.
public enum VehicleReader {
    public static func read(frame: RGBFrame, detections: [ObjectDetection], kind: String,
                            ocr: VisionTextReader, model: NativeVisionLanguageModel) throws -> VehicleEvidenceResult {
        var entries=[TextEvidence](),vehicles=[VehicleObservation]()
        for (index,detection) in detections.enumerated() where detection.label == kind {
            guard let (crop,box)=frame.cropped(to:detection.box) else { continue }
            let id="\(frame.sequence):\(index)"
            vehicles.append(VehicleObservation(label:kind,observationID:id))
            var text=try ocr.read(crop)
            for i in text.indices {
                let y=text[i].box.map { $0[1] }.reduce(0,+)/Float(max(1,text[i].box.count)*crop.height)
                let body=y > 0.62 || TextRules.matches("(?i)SEATING|STANDEES|座位|企位|核载|核載|载客|車號|车号",text[i].text)
                text[i].text_role=body ? "vehicle_body" : "display_candidate"
                text[i].vehicle_label=kind;text[i].vehicle_id=id
                text[i].box=text[i].box.map { [box[0]*Float(frame.width)+$0[0],box[1]*Float(frame.height)+$0[1]] }
            }
            let subject=kind == "train" ? "这列车" : "这辆公交车"
            let answer=try model.generate(prompt:subject+"线路显示屏上的线路号码是什么？只回答线路标识或无法确定。车牌、车身编号、座位数都不是线路。",frame:crop,maxTokens:40)
            if answer.finishReason == "cancelled" { break }
            let ids=Set(TextRules.identifiers(answer.text))
            let clear=ids.count == 1 && answer.finishReason == "stop"
                && !TextRules.matches("无法|不确定|可能|看不清|没有|未能",answer.text)
            let observed=clear ? ids.first! : ""
            for i in text.indices {
                text[i].route_verification_attempted=true
                text[i].route_verified = !observed.isEmpty && text[i].score >= 0.45 && TextRules.identifiers(text[i].text).contains(observed)
                if text[i].route_verified { text[i].text_role="route_display" }
            }
            if !observed.isEmpty && !text.contains(where: { $0.route_verified }) {
                let pixels=[box[0]*Float(frame.width),box[1]*Float(frame.height),box[2]*Float(frame.width),box[3]*Float(frame.height)]
                var candidate=TextEvidence(text:observed,score:0,box:[[pixels[0],pixels[1]],[pixels[2],pixels[1]],[pixels[2],pixels[3]],[pixels[0],pixels[3]]],
                    vehicleLabel:kind,vehicleID:id,textRole:"unconfirmed_route")
                candidate.route_candidate=true;candidate.route_verification_attempted=true;text.append(candidate)
            }
            entries.append(contentsOf:text)
        }
        return VehicleEvidenceResult(texts:entries,vehicles:vehicles)
    }
}
