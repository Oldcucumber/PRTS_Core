import AVFoundation
import Foundation
import PRTSContracts

/// Frontend event consumer; all methods must run on its main queue.
/// It controls speech/cue output only, not microphone capture or app UI.
public final class AppleSpeechOutput: NSObject, AVSpeechSynthesizerDelegate, AVAudioPlayerDelegate {
    private let synthesizer=AVSpeechSynthesizer()
    private let clock: () -> SessionTime
    private var pending: [SpeechRequest]=[]
    private var active: SpeechRequest?
    private var activeUtterance: AVSpeechUtterance?
    private var deadline: Timer?
    private var cuePlayer: AVAudioPlayer?
    private let cues: [String:URL]
    public var playbackChanged: ((Bool) -> Void)?
    public var playbackEvent: ((UInt64,String,SessionTime) -> Void)?
    public static func bundledCues() -> [String:URL] {
        Dictionary(uniqueKeysWithValues:["target_found","attention","arrived"].compactMap { name in
            Bundle.module.url(forResource:name,withExtension:"wav",subdirectory:"cues").map { (name,$0) }
        })
    }
    public init(cues: [String:URL] = AppleSpeechOutput.bundledCues(), clock: @escaping () -> SessionTime) {
        self.cues=cues; self.clock=clock
        super.init(); synthesizer.delegate=self
    }
    public func accept(_ request: SpeechRequest) {
        precondition(Thread.isMainThread)
        guard request.expires_s > clock() else { return }
        pending.removeAll { $0.replace_group == request.replace_group || $0.expires_s <= clock() }
        pending.append(request)
        if let active, request.priority > active.priority || active.replace_group == request.replace_group {
            synthesizer.stopSpeaking(at:.immediate)
            // Delegate clears the old utterance and starts the selected pending one.
        } else if active == nil { speakNext() }
    }
    public func cancelAll() {
        precondition(Thread.isMainThread)
        pending.removeAll(); deadline?.invalidate(); deadline=nil
        if let active { playbackEvent?(active.sequence,"cancelled",clock()) }
        active=nil; activeUtterance=nil
        synthesizer.stopSpeaking(at:.immediate); cuePlayer?.stop(); cuePlayer=nil
        playbackChanged?(false)
    }
    public func playCue(_ name: String, expires: SessionTime) throws {
        precondition(Thread.isMainThread)
        guard expires > clock(), let url=cues[name] else { return }
        let player=try AVAudioPlayer(contentsOf:url)
        cuePlayer=player; player.delegate=self; player.play(); playbackChanged?(true)
    }
    private func speakNext() {
        pending.removeAll { $0.expires_s <= clock() }
        pending.sort { $0.priority == $1.priority ? $0.sequence < $1.sequence : $0.priority > $1.priority }
        guard !pending.isEmpty else { playbackChanged?(cuePlayer?.isPlaying == true); return }
        let request=pending.removeFirst()
        let utterance=AVSpeechUtterance(string:request.text)
        utterance.voice=AVSpeechSynthesisVoice(language:"zh-CN")
        active=request; activeUtterance=utterance
        synthesizer.speak(utterance); playbackChanged?(true)
        deadline=Timer.scheduledTimer(withTimeInterval:max(0.01,request.expires_s-clock()),repeats:false) { [weak self] _ in
            self?.synthesizer.stopSpeaking(at:.immediate)
        }
    }
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        if activeUtterance === utterance, let active { playbackEvent?(active.sequence,"started",clock()) }
    }
    private func finished(_ utterance: AVSpeechUtterance, status: String) {
        guard activeUtterance === utterance else { return }
        if let active { playbackEvent?(active.sequence,status,clock()) }
        deadline?.invalidate(); deadline=nil; active=nil; activeUtterance=nil
        speakNext()
    }
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        finished(utterance,status:"finished")
    }
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        finished(utterance,status:"cancelled")
    }
    public func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        if cuePlayer === player { cuePlayer=nil; playbackChanged?(active != nil) }
    }
}
