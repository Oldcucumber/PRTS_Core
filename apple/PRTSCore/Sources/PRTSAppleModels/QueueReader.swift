import Foundation
import PRTSContracts
import prts_vlm

/// Queue OCR context refinement and visible digit gaps; no requested identifier.
public enum QueueReader {
    private static func bounds(_ e:TextEvidence) -> [Float] {
        [e.box.map { $0[0] }.min()!,e.box.map { $0[1] }.min()!,
         e.box.map { $0[0] }.max()!,e.box.map { $0[1] }.max()!]
    }
    private static func overlap(_ a:[Float],_ b:[Float]) -> Float {
        let intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
    }
    private static func split(_ entry:TextEvidence,frame:RGBFrame) -> [TextEvidence] {
        guard TextRules.matches("^[0-9]{3,}$",entry.text),entry.box.count == 4 else { return [entry] }
        let count=entry.text.utf8.count,q=entry.box.flatMap { $0 }
        var ranges=[Int32](repeating:0,count:count*2),boxes=[Float](repeating:0,count:count*8)
        let n=frame.bytes.withUnsafeBytes { source in q.withUnsafeBufferPointer { quad in
            ranges.withUnsafeMutableBufferPointer { r in boxes.withUnsafeMutableBufferPointer { b in
                prts_queue_split_rgb(source.bindMemory(to:UInt8.self).baseAddress,Int32(frame.width),Int32(frame.height),
                    quad.baseAddress,Int32(count),r.baseAddress,b.baseAddress)
            }}
        }}
        guard n>1 else { return [entry] }
        let characters=Array(entry.text)
        return (0..<Int(n)).map { i in
            var part=entry;part.text=String(characters[Int(ranges[2*i])..<Int(ranges[2*i+1])])
            part.box=(0..<4).map { j in [boxes[i*8+j*2],boxes[i*8+j*2+1]] }
            part.number_source="ocr_pixel_gap_split";part.original_text=entry.text
            return part
        }
    }
    public static func read(_ frame:RGBFrame,ocr:VisionTextReader) throws -> [TextEvidence] {
        var entries=try ocr.read(frame),proposals=[[Float]]()
        for e in QueueTextRules.classify(entries) {
            guard e.score>=0.7,TextRules.matches("^[0-9]{2,}$",e.text),e.box.count == 4 else { continue }
            if e.text.count<7 && !["other","unconfirmed"].contains(e.queue_role ?? "") { continue }
            let b=bounds(e)
            func edge(_ a:Int,_ c:Int) -> Float {
                let x=e.box[a][0]-e.box[c][0],y=e.box[a][1]-e.box[c][1];return sqrt(x*x+y*y)
            }
            let h=min(edge(0,3),edge(1,2)),pad=0.5*(b[2]-b[0])
            let proposal=[max(0,(b[0]-pad).rounded(.toNearestOrEven)),max(0,(b[1]-4*h).rounded(.toNearestOrEven)),
                min(Float(frame.width),(b[2]+pad).rounded(.toNearestOrEven)),min(Float(frame.height),(b[3]+2*h).rounded(.toNearestOrEven))]
            if proposals.contains(where: { overlap(proposal,$0)>0.5 }) { continue }
            proposals.append(proposal);if proposals.count == 4 { break }
        }
        for box in proposals {
            let normalized=[box[0]/Float(frame.width),box[1]/Float(frame.height),box[2]/Float(frame.width),box[3]/Float(frame.height)]
            guard let (crop,actual)=frame.cropped(to:normalized) else { continue }
            for original in try ocr.read(crop) {
                var shifted=original
                shifted.box=original.box.map { [$0[0]+actual[0]*Float(frame.width),$0[1]+actual[1]*Float(frame.height)] }
                if let index=entries.firstIndex(where: { overlap(bounds($0),bounds(shifted))>0.6 }) { entries[index]=shifted }
                else { entries.append(shifted) }
            }
        }
        return QueueTextRules.classify(entries.flatMap { split($0,frame:frame) })
    }
}
