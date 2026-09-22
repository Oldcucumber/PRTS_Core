import Foundation
import onnxruntime
import prts_vlm

public struct FloatTensor {
    public let values: [Float]
    public let shape: [Int64]
    public init(values: [Float], shape: [Int64]) { self.values=values; self.shape=shape }
}

/// Uses the same linked ONNX Runtime as sherpa-onnx. Its package release is
/// 1.28.2 while its pinned iOS binary is 1.28.1; report runtimeVersion at startup.
/// CPU C ABI is tested on Windows. CoreML selection and this Swift bridge must
/// be compiled and compared on the team's Mac/device before claiming parity.
public final class ONNXModel {
    private var handle: OpaquePointer?
    private let lock=NSLock()
    public let runtimeVersion: String
    public init(model: URL, threads: Int32 = 4, useCoreML: Bool = false) throws {
        guard let apiBase=OrtGetApiBase() else { throw NativeModelError.runtime("ONNX Runtime unavailable") }
        runtimeVersion=String(cString:apiBase.pointee.GetVersionString())
        handle=model.path.withCString { path in
            prts_onnx_create(UnsafeRawPointer(apiBase),path,threads,useCoreML ? 1 : 0)
        }
        guard handle != nil else { throw NativeModelError.runtime(Self.lastError()) }
    }
    private static func lastError() -> String {
        guard let message=prts_onnx_last_error() else { return "ONNX inference error" }
        return String(cString:message)
    }
    public func run(_ input: FloatTensor) throws -> FloatTensor {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw NativeModelError.closed }
        var result=prts_float_tensor()
        let code=input.values.withUnsafeBufferPointer { values in
            input.shape.withUnsafeBufferPointer { shape in
                prts_onnx_run(handle,values.baseAddress,Int64(values.count),shape.baseAddress,Int32(shape.count),&result)
            }
        }
        guard code == 0, let data=result.data else { throw NativeModelError.runtime(Self.lastError()) }
        let values=Array(UnsafeBufferPointer(start:data,count:Int(result.element_count)))
        let shape=withUnsafeBytes(of:result.shape) { raw in
            Array(raw.bindMemory(to:Int64.self).prefix(Int(result.rank)))
        }
        return FloatTensor(values:values,shape:shape)
    }
    public func close() {
        lock.lock(); defer { lock.unlock() }
        if let handle { prts_onnx_destroy(handle); self.handle=nil }
    }
    deinit { close() }
}
