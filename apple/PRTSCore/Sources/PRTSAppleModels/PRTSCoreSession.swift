import Foundation
import PRTSContracts

public struct CoreModelFiles {
    public let semantic:URL,semanticManifest:URL,detector:URL,detectorManifest:URL
    public let language:URL,projector:URL,asr:URL,asrTokens:URL
    public init(semantic: URL, semanticManifest: URL, detector: URL, detectorManifest: URL,
                language: URL, projector: URL, asr: URL, asrTokens: URL) {
        self.semantic=semantic;self.semanticManifest=semanticManifest;self.detector=detector;self.detectorManifest=detectorManifest
        self.language=language;self.projector=projector;self.asr=asr;self.asrTokens=asrTokens
    }
}

/// In-process core entry point. Construct on a background queue, push timestamped
/// sensor buffers, and consume events. No camera/microphone/UI is opened here.
/// This Apple implementation needs Xcode/device replay validation before release.
public final class PRTSCoreSession {
    private struct Query { let text:String;let epoch:UInt64;let timestamp:SessionTime }
    private let state=DispatchQueue(label:"prts.state")
    private let vision=DispatchQueue(label:"prts.vision",qos:.userInitiated)
    private let speech=DispatchQueue(label:"prts.language",qos:.userInitiated)
    private let recognition=DispatchQueue(label:"prts.asr",qos:.userInitiated)
    private let audioIngress=DispatchQueue(label:"prts.audio.ingress",qos:.userInitiated)
    private let events=DispatchQueue(label:"prts.events")
    private let work=DispatchGroup()
    private let frameSlot=LatestFrameSlot(),segmenter=PCMBuffer(),tasks=TaskState(),guide=LocalGuide()
    private let perception:NativePerception,language:NativeVisionLanguageModel,asr:SenseVoice
    private let ocr=VisionTextReader(),maps:MapService?
    private lazy var waitingBrain=VisualWaitingBrain(model:language)
    private var ambientObservations=[AmbientObservation]()
    private let clock:()->SessionTime,onEvent:(CoreEvent)->Void
    private var cameraSignal:DispatchSourceUserDataAdd!
    private var running=true,visionBusy=false,languageBusy=false,asrBusy=false,locationBusy=false
    private var playback=false // audioIngress only
    private var latest:RGBFrame?,lastVisionSequence:UInt64?,lastEvidenceSequence:UInt64?
    private var epoch:UInt64=0,sequence:UInt64=0,mapVersion:UInt64=0
    private var pendingQuery:Query?,pendingUtterance:(Utterance,UInt64)?
    private var location:LocationFix?,pendingLocation:LocationFix?,tracker:RouteTracker?,routeHint:RouteUpdate?
    private var destinations=[MapDestination](),candidateArrival:WaitRequest?,activeArrival:WaitRequest?
    private var lastNavigationMode=""
    private var lastGuide="",lastGuideAt = -Double.infinity,lastRoute="",lastRouteAt = -Double.infinity
    public init(files: CoreModelFiles, maps: MapService? = nil, useMetal: Bool = true, useCoreML: Bool = false,
                clock: @escaping ()->SessionTime = { ProcessInfo.processInfo.systemUptime },
                onEvent: @escaping (CoreEvent)->Void) throws {
        self.clock=clock;self.onEvent=onEvent;self.maps=maps
        perception=try NativePerception(semantic:files.semantic,semanticManifest:files.semanticManifest,
            detector:files.detector,detectorManifest:files.detectorManifest,useCoreML:useCoreML)
        language=try NativeVisionLanguageModel(model:files.language,projector:files.projector,useMetal:useMetal)
        asr=try SenseVoice(model:files.asr,tokens:files.asrTokens)
        cameraSignal=DispatchSource.makeUserDataAddSource(queue:state)
        cameraSignal.setEventHandler { [weak self] in
            guard let self,self.running else { return }
            if let frame=self.frameSlot.takeLatest() { self.latest=frame }
            self.scheduleVision();self.scheduleLanguage()
        }
        cameraSignal.resume()
    }
    @discardableResult public func pushVideo(_ frame: RGBFrame) -> Bool {
        let accepted=frameSlot.offer(frame)
        if accepted { cameraSignal.add(data:1) }
        return accepted
    }
    public func setPlayback(_ active: Bool) { audioIngress.async { self.playback=active } }
    public func pushAudio(_ chunk: PCMChunk) {
        audioIngress.async {
            if self.playback && chunk.role != .user && !chunk.echoCancelled { self.segmenter.reset();return }
            let starting = !self.segmenter.isCollecting
            let utterance=self.segmenter.append(chunk)
            if starting && (self.segmenter.isCollecting || utterance != nil) && chunk.role == .user {
                self.state.async { if self.running { self.interrupt("user_speech_started") } }
            }
            if let utterance {
                self.state.async {
                    guard self.running else { return }
                    if self.pendingUtterance != nil { self.emit("input_dropped",["input":.string("audio_utterance")]) }
                    self.pendingUtterance=(utterance,self.epoch);self.scheduleASR()
                }
            }
        }
    }
    public func pushText(_ text: String, role: AudioRole = .user, timestamp: SessionTime? = nil) {
        state.async {
            guard self.running else { return }
            if role == .user { self.interrupt("user_input") }
            self.handleText(text,role:role,timestamp:timestamp ?? self.clock())
        }
    }
    public func pushLocation(_ fix: LocationFix) {
        state.async { guard self.running else { return };self.pendingLocation=fix;self.scheduleLocation() }
    }
    /// Completion follows workers; no synchronous wait on the main/audio thread.
    public func close(completion: @escaping ()->Void = {}) {
        state.async {
            guard self.running else { completion();return }
            self.running=false;self.epoch += 1;self.mapVersion += 1;self.cameraSignal.cancel()
            self.pendingQuery=nil;self.pendingUtterance=nil;self.pendingLocation=nil
            self.language.cancel();self.emit("speech_cancel",["reason":.string("core_closed")])
            self.work.notify(queue:.global(qos:.utility)) { self.language.close();completion() }
        }
    }
    private func encoded<T:Encodable>(_ value:T) -> JSONValue {
        // Values originate in these typed model/contract structures; errors remain visible.
        do { return try JSONValue.value(value) }
        catch { return .object(["encoding_error":.string(String(describing:error))]) }
    }
    @discardableResult private func emit(_ type: String, _ payload: [String:JSONValue] = [:]) -> UInt64 {
        sequence += 1
        let event=CoreEvent(sequence:sequence,emitted:clock(),type:type,payload:payload)
        events.async { self.onEvent(event) }
        return sequence
    }
    private func say(_ text:String,source:String,priority:Int=30,ttl:Double=12,cue:String?=nil) {
        let expires=clock()+ttl
        let id=emit("speech_request",["text":.string(text),"source":.string(source),"priority":.number(Double(priority)),
            "expires_s":.number(expires),"replace_group":.string(["guidance","route_progress"].contains(source) ? "navigation" : source)])
        if let cue { emit("sound_cue",["cue":.string(cue),"related_sequence":.number(Double(id)),"expires_s":.number(expires)]) }
    }
    private func interrupt(_ reason:String) {
        epoch += 1;pendingQuery=nil;language.cancel()
        emit("speech_cancel",["reason":.string(reason)])
    }
    private func notice(_ event:TaskNotice) {
        emit(event.type,encoded(event).object ?? [:])
        say(event.text,source:event.type,priority:event.type == "target_observed" ? 70 : 30,
            ttl:12,cue:event.type == "target_observed" ? "target_found" : nil)
    }
    private func fail(_ worker:String,_ error:Error) {
        // Map adapters redact URLs/credentials before returning an error.
        emit("request_error",["worker":.string(worker),"text":.string(String(describing:error))])
    }
    private func handleText(_ text:String,role:AudioRole,timestamp:SessionTime) {
        let intent=CommandRouter.parse(text,waiting:tasks.wait,hasCandidates:!destinations.isEmpty)
        if role == .ambient || (role == .auto && intent == nil) {
            ambientObservations.append(AmbientObservation(text:text,observation_s:timestamp))
            ambientObservations=Array(ambientObservations.suffix(4));scheduleLanguage()
            return
        }
        if let intent,intent.action != "ask" { apply(intent);return }
        pendingQuery=Query(text:text,epoch:epoch,timestamp:timestamp);scheduleLanguage()
    }
    private func apply(_ intent:TaskIntent) {
        emit("intent",encoded(intent).object ?? [:])
        switch intent.action {
        case "destination": search(intent.query ?? "",arrival:intent.arrival_wait)
        case "confirm_destination": confirm(intent.index ?? 0)
        case "stop":
            lastNavigationMode="";lastGuide=""
            mapVersion += 1;tracker=nil;routeHint=nil;activeArrival=nil;candidateArrival=nil;destinations=[]
            notice(tasks.command(intent,now:clock()))
        default:
            notice(tasks.command(intent,now:clock()))
            scheduleVision();scheduleLanguage()
        }
    }
    private func scheduleASR() {
        guard running,!asrBusy,let (utterance,generation)=pendingUtterance else { return }
        pendingUtterance=nil;asrBusy=true;work.enter()
        recognition.async {
            let text=self.asr.transcribe(utterance)
            self.state.async {
                self.asrBusy=false;defer { self.work.leave();self.scheduleASR() }
                guard self.running else { return }
                self.emit("transcript",["text":.string(text),"role":.string(utterance.role.rawValue),
                    "observation_s":.number(utterance.start),"end_s":.number(utterance.end)])
                guard utterance.role == .ambient || generation == self.epoch else { return }
                self.handleText(text,role:utterance.role,timestamp:utterance.start)
            }
        }
    }
    private func scheduleVision() {
        guard running,tasks.navigating,!visionBusy,let frame=latest,frame.sequence != lastVisionSequence else { return }
        visionBusy=true;lastVisionSequence=frame.sequence;let hint=routeHint;work.enter()
        vision.async {
            let result=Result { () -> (PerceptionObservation,LocalGuidance) in
                let observation=try self.perception.observe(frame)
                return (observation,try self.guide.update(observation,route:hint))
            }
            self.state.async {
                self.visionBusy=false;defer { self.work.leave();self.scheduleVision() }
                guard self.running else { return }
                do {
                    let (observation,original)=try result.get(),grid=observation.semantic,age=self.clock()-frame.timestamp
                    let dense:JSONValue = .object(["width":.number(Double(grid.width)),"height":.number(Double(grid.height)),
                        "class_ids":.string(Data(grid.classIDs).base64EncodedString()),"labels":self.encoded(grid.classNames),
                        "encoding":.string("base64-u8-rowmajor"),"coordinate_space":.string("image_normalized")])
                    self.emit("scene",["frame_sequence":.number(Double(frame.sequence)),"observation_s":.number(frame.timestamp),
                        "age_s":.number(age),"semantic_grid":dense,"detections":self.encoded(observation.detections),
                        "guidance":self.encoded(original),"processing_s":.number(observation.processingSeconds)])
                    guard self.tasks.navigating else { return }
                    var published=original
                    if age > 2 || age < 0 { published.status="WAIT";published.direction="UNKNOWN";published.path=[]
                        published.mode="observation_wait";published.forward_scan=nil
                        published.reason="stale_observation";published.text="画面处理尚未跟上，请先等待" }
                    var payload=self.encoded(published).object ?? [:]
                    payload["observation_s"] = .number(frame.timestamp);payload["frame_sequence"] = .number(Double(frame.sequence))
                    self.emit("guidance",payload)
                    let mode=published.mode,entered=mode == "free_forward" && self.lastNavigationMode != mode
                    let key=mode+":"+published.status+":"+published.direction+":"+published.reason
                    let changed=key != self.lastGuide
                    if ["free_forward","sidewalk"].contains(mode) { self.lastNavigationMode=mode }
                    if changed || entered || (mode != "free_forward" && self.clock()-self.lastGuideAt >= 4) {
                        if published.status == "STOP",changed {
                            self.emit("speech_cancel",["reason":.string("navigation_hazard"),"preserve_wait_task":.bool(true)])
                        }
                        if mode != "free_forward" || entered || (published.status == "STOP" && changed) {
                            var text=published.text
                            if entered {
                                text="进入自由前进模式，持续观察正前方障碍"
                                if published.status == "STOP" { text += "。"+published.text }
                            }
                            let priority=changed ? (["STOP","WAIT"].contains(published.status) ? 80 : 40) : 20
                            self.say(text,source:"guidance",priority:priority,ttl:entered ? 12 : 3)
                        }
                        self.lastGuide=key;self.lastGuideAt=self.clock()
                    }
                } catch { self.fail("vision",error) }
            }
        }
    }
    private func scheduleLanguage() {
        guard running,!languageBusy else { return }
        if let query=pendingQuery {
            pendingQuery=nil
            guard let frame=latest else { say("尚未收到相机画面",source:"answer");return }
            languageBusy=true;work.enter()
            speech.async {
                let result=Result { () -> (TaskIntent?,String,[TextEvidence]) in
                    if CommandRouter.parse(query.text) == nil {
                        let response=try self.language.generate(prompt:"提取任务为 JSON。action 为 ask、navigate、stop、wait、cancel 之一；kind 为 bus、train、number 或空；target 保留完整字母和数字；direction 没有则空。只提取本次用户明确要求，不补全目标。用户："+query.text,maxTokens:100)
                        if response.finishReason == "stop",let begin=response.text.firstIndex(of:"{"),let end=response.text.lastIndex(of:"}"),begin <= end,
                           let intent=try? JSONDecoder().decode(TaskIntent.self,from:Data(response.text[begin...end].utf8)),intent.action != "ask" {
                            if intent.action != "wait" || TextRules.identifiers(query.text).contains(intent.target ?? "") { return (intent,"",[]) }
                        }
                    }
                    let texts=try self.ocr.read(frame)
                    let lines=texts.filter { $0.score >= 0.8 }.sorted {
                        let ay=$0.box.first?[1] ?? 0,by=$1.box.first?[1] ?? 0
                        return ay == by ? ($0.box.first?[0] ?? 0) < ($1.box.first?[0] ?? 0) : ay < by
                    }.map(\.text)
                    if TextRules.matches("写|读|文字|牌|标识|告示|标志|内容",query.text),!lines.isEmpty {
                        return (nil,"画面可辨认的文字："+lines.joined(separator:"；")+"。",texts)
                    }
                    let response=try self.language.generate(prompt:"你是 PRTS 离线视觉助手。用简洁中文回答，保留文字的否定和条件；不猜测遮挡文字或未测量距离，不把灯色当作通行许可。当前 OCR："+lines.joined(separator:"；")+"。用户："+query.text,frame:frame)
                    return (nil,response.finishReason == "cancelled" ? "" : response.text,texts)
                }
                self.state.async {
                    self.languageBusy=false;defer { self.work.leave();self.scheduleLanguage() }
                    guard self.running,query.epoch == self.epoch else { return }
                    do {
                        let (intent,text,ocr)=try result.get()
                        if let intent { self.apply(intent);return }
                        guard !text.isEmpty else { return }
                        let age=self.clock()-frame.timestamp
                        self.emit("answer",["text":.string(text),"ocr":self.encoded(ocr),"observation_s":.number(frame.timestamp),
                            "frame_sequence":.number(Double(frame.sequence)),"age_s":.number(age),"retrospective":.bool(age > 2)])
                        let spoken=(age > 2 ? "根据约\(Int(age.rounded()))秒前的画面，" : "")+text
                        self.say(spoken,source:"answer",ttl:max(12,min(60,Double(spoken.count)/3+5)))
                    } catch { self.fail("language",error) }
                }
            }
            return
        }
        guard let task=tasks.wait,!task.notified,let frame=latest,frame.sequence != lastEvidenceSequence,
              clock()-frame.timestamp <= 2 else { return }
        languageBusy=true;lastEvidenceSequence=frame.sequence;work.enter()
        let ambient=ambientObservations.filter { $0.observation_s >= task.started && frame.timestamp-$0.observation_s >= 0 && frame.timestamp-$0.observation_s <= 45 }
        speech.async {
            let result=Result { () -> (BrainObservation,[TextEvidence]) in
                let texts=try self.ocr.read(frame)
                return (try self.waitingBrain.observe(frame:frame,task:task,texts:texts,ambient:ambient),texts)
            }
            self.state.async {
                self.languageBusy=false;defer { self.work.leave();self.scheduleLanguage() }
                guard self.running,self.tasks.wait?.version == task.version else { return }
                do {
                    let (brain,texts)=try result.get(),age=self.clock()-brain.evidence_s
                    self.emit("target_evidence",["ocr":self.encoded(texts),
                        "observation_s":.number(frame.timestamp),"frame_sequence":.number(Double(frame.sequence)),"age_s":.number(self.clock()-frame.timestamp)])
                    var payload=self.encoded(brain).object ?? [:]
                    payload["observation_s"] = .number(brain.evidence_s);payload["age_s"] = .number(age)
                    payload["frame_sequence"] = .number(Double(frame.sequence));self.emit("brain_observation",payload)
                    if age >= 0,age <= (task.kind == "number" ? 45 : 30),var result=self.tasks.applyBrain(decision:brain.decision,taskVersion:task.version,
                        observation:brain.observation,now:brain.evidence_s) {
                        let channel=brain.observation_channel == "environment_transcript" ? "环境广播" : "画面"
                        if age > 2 { result.text="约\(Int(age.rounded()))秒前的\(channel)中，"+result.text }
                        self.notice(result)
                    }
                } catch { self.fail("evidence",error) }
            }
        }
    }
    private func search(_ query:String,arrival:WaitRequest?) {
        guard let maps else { emit("map_unavailable");say("尚未配置地图服务",source:"navigation");return }
        mapVersion += 1;let version=mapVersion,near=location?.point;work.enter()
        Task {
            let result:Result<[MapDestination],Error>
            do { result = .success(try await maps.search(query,near:near)) } catch { result = .failure(error) }
            self.state.async {
                defer { self.work.leave() };guard self.running,self.mapVersion == version else { return }
                do {
                    self.destinations=try result.get();self.candidateArrival=arrival
                    let text=self.destinations.isEmpty ? "没有找到匹配地点，请换一个名称" : "请选择目的地："+self.destinations.enumerated().map { "\($0.offset+1)，\($0.element.name)，\($0.element.address)" }.joined(separator:"；")
                    self.emit("destination_candidates",["candidates":self.encoded(self.destinations),"needs_confirmation":.bool(true),"text":.string(text)])
                    self.say(text,source:"destination_candidates")
                } catch { self.fail("map_search",error) }
            }
        }
    }
    private func confirm(_ choice:Int) {
        let index=choice == 0 && destinations.count == 1 ? 1 : choice
        guard (1...max(1,destinations.count)).contains(index),index <= destinations.count else {
            say("请选择候选列表中的一个地点",source:"navigation");return
        }
        guard let maps,let location,clock()-location.timestamp_s <= 5,location.accuracy_m <= 25 else {
            say("需要更新当前位置再规划路线",source:"navigation");return
        }
        mapVersion += 1;let version=mapVersion,destination=destinations[index-1],arrival=candidateArrival;work.enter()
        Task {
            let result:Result<WalkingRoute,Error>
            do { result = .success(try await maps.walking(origin:location.point,destination:destination)) } catch { result = .failure(error) }
            self.state.async {
                defer { self.work.leave() };guard self.running,self.mapVersion == version else { return }
                do {
                    let route=try result.get();self.tracker=try RouteTracker(route:route);self.activeArrival=arrival
                    self.destinations=[];self.candidateArrival=nil;self.lastRoute="";self.routeHint=nil
                    _=self.tasks.command(TaskIntent(action:"navigate"),now:self.clock())
                    self.emit("route_started",["route":self.encoded(route),"version":.number(Double(version))])
                    self.say("已确认目的地，开始按步行路线引导",source:"route_started");self.scheduleVision()
                } catch { self.fail("route",error) }
            }
        }
    }
    private func scheduleLocation() {
        guard running,!locationBusy,let fix=pendingLocation else { return }
        pendingLocation=nil;locationBusy=true;work.enter()
        Task {
            let result:Result<LocationFix,Error>
            do {
                let point:MapPoint
                if fix.point.crs == "GCJ02" { point=fix.point }
                else if let maps=self.maps { point=try await maps.convert(fix.point) }
                else { throw NavigationError.unavailable("此位置需要地图坐标转换") }
                result = .success(LocationFix(point:point,timestamp:fix.timestamp_s,accuracy:fix.accuracy_m,
                    cameraHeading:fix.camera_heading_deg,headingAccuracy:fix.heading_accuracy_deg))
            } catch { result = .failure(error) }
            self.state.async {
                self.locationBusy=false;defer { self.work.leave();self.scheduleLocation() }
                guard self.running else { return }
                do {
                    let normalized=try result.get()
                    if let old=self.location,normalized.timestamp_s < old.timestamp_s { return }
                    self.location=normalized
                    guard self.tasks.navigating,let tracker=self.tracker else { return }
                    let event=tracker.update(normalized,now:self.clock());self.routeHint=event
                    self.emit("route_progress",self.encoded(event).object ?? [:])
                    let key="\(event.status):\(event.step_index ?? -1):\((event.distance_to_maneuver_m ?? 100) <= 20)"
                    if key != self.lastRoute || self.clock()-self.lastRouteAt >= 15 {
                        self.say(event.text,source:"route_progress",priority:50,ttl:6,cue:event.status == "arrived" ? "arrived" : nil)
                        self.lastRoute=key;self.lastRouteAt=self.clock()
                    }
                    if event.status == "arrived",let arrival=self.activeArrival {
                        self.activeArrival=nil
                        self.notice(self.tasks.command(TaskIntent(action:"wait",kind:arrival.kind,target:arrival.target,direction:arrival.direction),now:self.clock()))
                        self.scheduleLanguage()
                    }
                } catch { self.fail("location",error) }
            }
        }
    }
}
