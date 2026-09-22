import Foundation
import PRTSContracts
import prts_vlm

public enum NativeModelError: Error { case runtime(String), closed }
public struct GeneratedText {
    public let text: String
    public let finishReason: String
    public let processingSeconds: Double
}

private final class TextAccumulator {
    var bytes=Data()
    var text=""
    let onDelta: ((String) -> Void)?
    init(_ onDelta: ((String) -> Void)?) { self.onDelta=onDelta }
    func append(_ pointer: UnsafePointer<UInt8>, count: Int) {
        bytes.append(pointer,count:count)
        // Wait for a complete UTF-8 prefix instead of replacing partial Hanzi.
        if let decoded=String(data:bytes,encoding:.utf8) {
            let delta=String(decoded.dropFirst(text.count)); text=decoded
            if !delta.isEmpty { onDelta?(delta) }
        }
    }
}

/// Runs the exact C ABI tested by the Windows/Python reference. Call off main.
/// Apple Metal/build/resource performance still requires device validation.
public final class NativeVisionLanguageModel {
    private var handle: OpaquePointer?
    private let runLock=NSLock()
    public init(model: URL, projector: URL, threads: Int32 = 4,
                imageSlices: Int32 = 1, useMetal: Bool = true, deterministic: Bool = true) throws {
        handle=model.path.withCString { m in
            projector.path.withCString { p in
                var config=prts_vlm_config(model_path:m,projector_path:p,context_tokens:4096,
                    threads:threads,image_slices:imageSlices,use_gpu:useMetal ? 1 : 0)
                return prts_vlm_create(&config)
            }
        }
        guard handle != nil else { throw NativeModelError.runtime(Self.lastError()) }
        prts_vlm_set_deterministic(handle,deterministic ? 1 : 0)
    }
    private static func lastError() -> String {
        guard let value=prts_vlm_last_error() else { return "Native VLM error" }
        return String(cString:value)
    }
    public func generate(prompt: String, frame: RGBFrame? = nil, maxTokens: Int32 = 180,
                         jsonSchema: String? = nil,
                         onDelta: ((String) -> Void)? = nil) throws -> GeneratedText {
        runLock.lock(); defer { runLock.unlock() }
        guard let handle else { throw NativeModelError.closed }
        let output=TextAccumulator(onDelta)
        let user=Unmanaged.passUnretained(output).toOpaque()
        let start=ProcessInfo.processInfo.systemUptime
        let callback: prts_text_callback = { pointer, length, context in
            guard let pointer, let context else { return }
            Unmanaged<TextAccumulator>.fromOpaque(context).takeUnretainedValue().append(pointer,count:Int(length))
        }
        let code: Int32=withExtendedLifetime(output) {
            prompt.withCString { text in
                func run(_ bytes: UnsafePointer<UInt8>?, _ width: UInt32, _ height: UInt32) -> Int32 {
                    if let jsonSchema {
                        return jsonSchema.withCString { schema in
                            prts_vlm_run_json(handle,text,schema,bytes,width,height,maxTokens,callback,user)
                        }
                    }
                    return prts_vlm_run(handle,text,bytes,width,height,maxTokens,callback,user)
                }
                if let frame {
                    return frame.bytes.withUnsafeBytes { buffer in
                        run(buffer.bindMemory(to:UInt8.self).baseAddress,UInt32(frame.width),UInt32(frame.height))
                    }
                }
                return run(nil,0,0)
            }
        }
        guard code >= 0 else { throw NativeModelError.runtime(Self.lastError()) }
        return GeneratedText(text:output.text.trimmingCharacters(in:.whitespacesAndNewlines),
            finishReason:code == 0 ? "stop" : code == 1 ? "cancelled" : "length",
            processingSeconds:ProcessInfo.processInfo.systemUptime-start)
    }
    /// Compute cancellation is checked after vision encoding and between tokens.
    /// Stop frontend playback immediately without waiting for encoding to finish.
    public func cancel() { if let handle { prts_vlm_cancel(handle) } }
    /// Stop submitting work before close. Existing synchronous generation finishes.
    public func close() {
        cancel(); runLock.lock(); defer { runLock.unlock() }
        if let handle { prts_vlm_destroy(handle); self.handle=nil }
    }
    deinit { close() }
}
