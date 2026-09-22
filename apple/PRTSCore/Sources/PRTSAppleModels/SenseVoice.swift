import Foundation
import PRTSContracts
import SherpaOnnx

/// Sentence recognition on a background serial queue. The frontend continues
/// supplying PCM while this worker decodes the preceding bounded utterance.
public final class SenseVoice {
    private let recognizer: SherpaOnnxOfflineRecognizer
    private let lock=NSLock()
    public init(model: URL, tokens: URL, threads: Int = 4) throws {
        for file in [model,tokens] where !FileManager.default.fileExists(atPath:file.path) {
            throw CocoaError(.fileNoSuchFile,userInfo:[NSFilePathErrorKey:file.path])
        }
        let voice=sherpaOnnxOfflineSenseVoiceModelConfig(model:model.path,language:"zh",
                                                        useInverseTextNormalization:true)
        let models=sherpaOnnxOfflineModelConfig(tokens:tokens.path,numThreads:threads,
                                               provider:"cpu",senseVoice:voice)
        var config=sherpaOnnxOfflineRecognizerConfig(
            featConfig:sherpaOnnxFeatureConfig(sampleRate:16000,featureDim:80),modelConfig:models)
        recognizer=SherpaOnnxOfflineRecognizer(config:&config)
    }
    public func transcribe(_ utterance: Utterance) -> String {
        lock.lock(); defer { lock.unlock() }
        return recognizer.decode(samples:utterance.samples,sampleRate:16000).text
    }
}
