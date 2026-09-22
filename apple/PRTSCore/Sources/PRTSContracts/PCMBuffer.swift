import Foundation

public struct Utterance {
    public let start: SessionTime
    public let end: SessionTime
    public let samples: [Float]
    public let role: AudioRole
}

/// Small energy segmenter matching the Python reference's thresholds.
/// It does not identify speakers; supply user/ambient from the frontend contract.
public final class PCMBuffer {
    private var pending: [Float] = []
    private var preroll: [(SessionTime, [Float])] = []
    private var start: SessionTime?
    private var lastVoice: SessionTime = 0
    private var expected: SessionTime?
    private var sequence: UInt64?
    private var role: AudioRole = .auto
    public init() {}
    public var isCollecting: Bool { start != nil }
    public func reset() {
        pending.removeAll(keepingCapacity: true); preroll.removeAll()
        start=nil; expected=nil; sequence=nil
    }
    /// Call serially on the audio ingestion queue.
    public func append(_ chunk: PCMChunk) -> Utterance? {
        precondition(chunk.samples.count <= 16_000)
        if let seq=sequence, let stamp=expected,
           chunk.sequence != seq + 1 || abs(chunk.timestamp-stamp) > 0.08 { reset() }
        if start != nil && chunk.role != role { reset() }
        sequence=chunk.sequence
        let end=chunk.timestamp+Double(chunk.samples.count)/16000
        expected=end
        let energy=chunk.samples.reduce(Float(0)) { $0+$1*$1 } / Float(max(1,chunk.samples.count))
        let voiced=sqrt(energy) >= 0.012
        if start == nil {
            preroll.append((chunk.timestamp,chunk.samples))
            while preroll.count > 1 && chunk.timestamp-preroll[0].0 > 0.18 { preroll.removeFirst() }
            guard voiced else { return nil }
            start=preroll[0].0; pending=preroll.flatMap { $0.1 }; preroll.removeAll(); role=chunk.role
        } else { pending.append(contentsOf: chunk.samples) }
        if voiced { lastVoice=end }
        let begin=start!
        if chunk.endUtterance || end-lastVoice >= 0.45 || end-begin >= 12 {
            let result=Utterance(start:begin,end:end,samples:pending,role:role)
            pending=[]; start=nil
            return lastVoice-begin >= 0.18 ? result : nil
        }
        return nil
    }
}
