import Foundation

/// Session seconds from one monotonic clock. Never mix wall-clock/Unix/GPS time.
public typealias SessionTime = Double

public enum AudioRole: String, Codable { case user, ambient, auto }
public struct PCMChunk {
    public let timestamp: SessionTime
    public let sequence: UInt64
    public let samples: [Float] // 16 kHz, mono, [-1, 1], <= 1 second
    public let role: AudioRole
    public let endUtterance: Bool
    public let echoCancelled: Bool
    public init(timestamp: SessionTime, sequence: UInt64, samples: [Float],
                role: AudioRole = .auto, endUtterance: Bool = false, echoCancelled: Bool = false) {
        self.timestamp=timestamp; self.sequence=sequence; self.samples=samples
        self.role=role; self.endUtterance=endUtterance; self.echoCancelled=echoCancelled
    }
}

public struct SpatialSample {
    public let timestamp: SessionTime
    public let worldFrameID: String
    public let intrinsics: [Double]? // row-major 3x3, upright original pixel coordinates
    public let cameraToWorld: [Double]? // row-major 4x4; local right-handed metres
    public let tracking: String
    public let depthMetres: [Float]?
    public let depthWidth: Int
    public let depthHeight: Int
    public init(timestamp: SessionTime, worldFrameID: String = "", intrinsics: [Double]? = nil,
                cameraToWorld: [Double]? = nil, tracking: String = "unavailable",
                depthMetres: [Float]? = nil, depthWidth: Int = 0, depthHeight: Int = 0) {
        self.timestamp=timestamp; self.worldFrameID=worldFrameID; self.intrinsics=intrinsics
        self.cameraToWorld=cameraToWorld; self.tracking=tracking; self.depthMetres=depthMetres
        self.depthWidth=depthWidth; self.depthHeight=depthHeight
    }
}

/// RGB24, top-left origin; rotation/mirroring must already be applied.
public struct RGBFrame {
    public let timestamp: SessionTime
    public let sequence: UInt64
    public let width: Int
    public let height: Int
    public let bytes: Data
    public let spatial: SpatialSample?
    public init(timestamp: SessionTime, sequence: UInt64, width: Int, height: Int,
                bytes: Data, spatial: SpatialSample? = nil) {
        precondition(width > 0 && height > 0 && bytes.count == width * height * 3)
        self.timestamp=timestamp; self.sequence=sequence; self.width=width; self.height=height
        self.bytes=bytes; self.spatial=spatial
    }
}

public struct SpeechRequest: Codable {
    public let sequence: UInt64
    public let text: String
    public let priority: Int
    public let expires_s: SessionTime
    public let replace_group: String
    public init(sequence: UInt64, text: String, priority: Int, expires_s: SessionTime,
                replace_group: String) {
        self.sequence=sequence; self.text=text; self.priority=priority
        self.expires_s=expires_s; self.replace_group=replace_group
    }
}

/// Bounded camera ingress. A consumer takes the newest frame when it can run.
/// Task creation per camera callback is deliberately avoided.
public final class LatestFrameSlot {
    private let lock=NSLock()
    private var latest: RGBFrame?
    private var lastSequence: UInt64?
    private var lastTimestamp: SessionTime?
    public init() {}
    @discardableResult public func offer(_ frame: RGBFrame) -> Bool {
        lock.lock(); defer { lock.unlock() }
        if let seq=lastSequence, frame.sequence <= seq { return false }
        if let stamp=lastTimestamp, frame.timestamp < stamp { return false }
        latest=frame; lastSequence=frame.sequence; lastTimestamp=frame.timestamp
        return true
    }
    public func takeLatest() -> RGBFrame? {
        lock.lock(); defer { lock.unlock() }
        let result=latest; latest=nil; return result
    }
}
