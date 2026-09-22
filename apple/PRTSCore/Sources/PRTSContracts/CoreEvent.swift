import Foundation

public enum JSONValue: Codable {
    case string(String), number(Double), bool(Bool), array([JSONValue]), object([String:JSONValue]), null
    public init(from decoder: Decoder) throws {
        let c=try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let x=try? c.decode(Bool.self) { self = .bool(x) }
        else if let x=try? c.decode(Double.self) { self = .number(x) }
        else if let x=try? c.decode(String.self) { self = .string(x) }
        else if let x=try? c.decode([String:JSONValue].self) { self = .object(x) }
        else { self = .array(try c.decode([JSONValue].self)) }
    }
    public func encode(to encoder: Encoder) throws {
        var c=encoder.singleValueContainer()
        switch self {
        case .string(let x):try c.encode(x)
        case .number(let x):try c.encode(x)
        case .bool(let x):try c.encode(x)
        case .array(let x):try c.encode(x)
        case .object(let x):try c.encode(x)
        case .null:try c.encodeNil()
        }
    }
    public static func value<T:Encodable>(_ value:T) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(value))
    }
    public var string: String? { if case .string(let s)=self { return s };return nil }
    public var number: Double? { if case .number(let n)=self { return n };return nil }
    public var object: [String:JSONValue]? { if case .object(let d)=self { return d };return nil }
}

/// Flat JSON envelope matching the Python event protocol. Native typed fields
/// remain available without a JSON round trip to the frontend.
public struct CoreEvent: Encodable {
    public let sequence: UInt64
    public let emitted_s: SessionTime
    public let type: String
    public let payload: [String:JSONValue]
    public init(sequence: UInt64, emitted: SessionTime, type: String, payload: [String:JSONValue] = [:]) {
        self.sequence=sequence;self.emitted_s=emitted;self.type=type;self.payload=payload
    }
    public func encode(to encoder: Encoder) throws {
        var all=payload
        all["schema_version"] = .number(1);all["sequence"] = .number(Double(sequence))
        all["emitted_s"] = .number(emitted_s);all["type"] = .string(type)
        try all.encode(to:encoder)
    }
    public var speechRequest: SpeechRequest? {
        guard type == "speech_request",let text=payload["text"]?.string,
              let priority=payload["priority"]?.number,let expires=payload["expires_s"]?.number,
              let group=payload["replace_group"]?.string else { return nil }
        return SpeechRequest(sequence:sequence,text:text,priority:Int(priority),expires_s:expires,replace_group:group)
    }
}
