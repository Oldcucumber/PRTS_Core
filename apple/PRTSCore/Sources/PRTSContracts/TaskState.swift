import Foundation

public enum TextRules {
    public static func matches(_ pattern: String, _ text: String) -> Bool {
        text.range(of: pattern, options: .regularExpression) != nil
    }
    public static func captures(_ pattern: String, _ text: String) -> [[String]] {
        guard let regex=try? NSRegularExpression(pattern:pattern) else { return [] }
        let ns=text as NSString
        return regex.matches(in:text,range:NSRange(location:0,length:ns.length)).map { match in
            (0..<match.numberOfRanges).map { i in
                let range=match.range(at:i)
                return range.location == NSNotFound ? "" : ns.substring(with:range)
            }
        }
    }
    public static func normalized(_ text: String) -> String {
        let traditional=Array("請號車來綠紅燈開導航繼續暫這個時剛幫換櫃臺輛進達線轉臨專")
        let simplified=Array("请号车来绿红灯开导航继续暂这个时刚帮换柜台辆进达线转临专")
        let table=Dictionary(uniqueKeysWithValues:zip(traditional,simplified))
        return String(text.precomposedStringWithCompatibilityMapping.map { table[$0] ?? $0 })
    }
    /// Complete route/queue identifiers, with no route dictionary or requested-target repair.
    public static func identifiers(_ input: String) -> [String] {
        var text=normalized(input).uppercased()
        let digits:[Character:Int] = ["零":0,"〇":0,"一":1,"幺":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9]
        let units:[Character:Int] = ["十":10,"百":100,"千":1000]
        let regex=try! NSRegularExpression(pattern:"[零〇一幺二两三四五六七八九十百千]+")
        for match in regex.matches(in:text,range:NSRange(text.startIndex...,in:text)).reversed() {
            guard let range=Range(match.range,in:text) else { continue }
            let old=String(text[range]),replacement:String
            if !old.contains(where: { units[$0] != nil }) {
                replacement=old.map { String(digits[$0]!) }.joined()
            } else {
                var total=0,current=0
                for c in old {
                    if let digit=digits[c] { current=digit }
                    else if let unit=units[c] { total += (current == 0 ? 1 : current)*unit; current=0 }
                }
                replacement=String(total+current)
            }
            text.replaceSubrange(range,with:replacement)
        }
        text=text.replacingOccurrences(of:"(?<![A-Z])([A-Z]{1,2})\\s+(?=\\d)",with:"$1",options:.regularExpression)
        text=text.replacingOccurrences(of:"(?<![A-Z])([A-Z]{1,4})\\s*[-‐‑–]\\s*(?=\\d)",with:"$1",options:.regularExpression)
        return captures("(?<![A-Z0-9夜快特临专])[A-Z夜快特临专]*[0-9]+[A-Z]*(?![A-Z0-9])",text).map { $0[0] }
    }
    public static func directionMatches(_ requested: String, _ observed: String) -> Bool {
        func clean(_ value:String) -> String {
            normalized(value).lowercased().replacingOccurrences(of:"[\\s，,。./、→>\\-]+",with:"",options:.regularExpression)
        }
        return requested.isEmpty || clean(observed).contains(clean(requested))
    }
}

public struct WaitRequest: Codable, Equatable {
    public var kind: String
    public var target: String
    public var direction: String
    public init(kind: String, target: String, direction: String = "") {
        self.kind=kind; self.target=target; self.direction=direction
    }
}

public struct TaskIntent: Codable {
    public var action: String
    public var kind: String?
    public var target: String?
    public var direction: String?
    public var query: String?
    public var index: Int?
    public var arrival_wait: WaitRequest?
    public init(action: String, kind: String? = nil, target: String? = nil, direction: String? = nil,
                query: String? = nil, index: Int? = nil, arrivalWait: WaitRequest? = nil) {
        self.action=action; self.kind=kind; self.target=target; self.direction=direction
        self.query=query; self.index=index; self.arrival_wait=arrivalWait
    }
}

public struct WaitingTask: Codable {
    public let kind: String
    public let target: String
    public let direction: String
    public let version: UInt64
    public let started: SessionTime
    public var notified: Bool
    public var candidate_notified: Bool
}

/// OCR boxes use upright-image pixel coordinates, including crop offsets.
public struct TextEvidence: Codable {
    public var text: String
    public var score: Float
    public var box: [[Float]]
    public var vehicle_label: String?
    public var vehicle_id: String?
    public var text_role: String?
    public var route_verification_attempted: Bool = false
    public var route_verified: Bool = false
    public var route_candidate: Bool = false
    public var queue_verification_attempted: Bool = false
    public var queue_role: String?
    public var queue_heading: String?
    public var queue_heading_box: [[Float]]?
    public var number_source: String?
    public var original_text: String?
    public init(text: String, score: Float, box: [[Float]] = [], vehicleLabel: String? = nil,
                vehicleID: String? = nil, textRole: String? = nil) {
        self.text=text; self.score=score; self.box=box; self.vehicle_label=vehicleLabel
        self.vehicle_id=vehicleID; self.text_role=textRole
    }
    private enum CodingKeys:String,CodingKey {
        case text,score,box,vehicle_label,vehicle_id,text_role,route_verification_attempted,route_verified,route_candidate
        case queue_verification_attempted,queue_role,queue_heading,queue_heading_box,number_source,original_text
    }
    public init(from decoder: Decoder) throws {
        let c=try decoder.container(keyedBy:CodingKeys.self)
        text=try c.decode(String.self,forKey:.text);score=try c.decode(Float.self,forKey:.score)
        box=try c.decodeIfPresent([[Float]].self,forKey:.box) ?? []
        vehicle_label=try c.decodeIfPresent(String.self,forKey:.vehicle_label)
        vehicle_id=try c.decodeIfPresent(String.self,forKey:.vehicle_id)
        text_role=try c.decodeIfPresent(String.self,forKey:.text_role)
        route_verification_attempted=try c.decodeIfPresent(Bool.self,forKey:.route_verification_attempted) ?? false
        route_verified=try c.decodeIfPresent(Bool.self,forKey:.route_verified) ?? false
        route_candidate=try c.decodeIfPresent(Bool.self,forKey:.route_candidate) ?? false
        queue_verification_attempted=try c.decodeIfPresent(Bool.self,forKey:.queue_verification_attempted) ?? false
        queue_role=try c.decodeIfPresent(String.self,forKey:.queue_role)
        queue_heading=try c.decodeIfPresent(String.self,forKey:.queue_heading)
        queue_heading_box=try c.decodeIfPresent([[Float]].self,forKey:.queue_heading_box)
        number_source=try c.decodeIfPresent(String.self,forKey:.number_source)
        original_text=try c.decodeIfPresent(String.self,forKey:.original_text)
    }
}

public struct VehicleObservation: Codable {
    public let label: String
    public let observation_id: String
    public init(label: String, observationID: String) { self.label=label; self.observation_id=observationID }
}

public struct TaskNotice: Codable {
    public let type: String
    public var text: String
    public var wait: WaitingTask?
    public var source: String?
    public var observation_s: SessionTime?
    public var evidence_text: String?
    public var vehicle_id: String?
    public init(type: String, text: String) { self.type=type; self.text=text }
}

/// Call on one state queue. Neural observations never mutate the current task.
public final class TaskState {
    public private(set) var navigating=false
    public private(set) var wait: WaitingTask?
    public private(set) var version: UInt64=0
    public init() {}
    public func command(_ intent: TaskIntent, now: SessionTime) -> TaskNotice {
        if intent.action == "continue_wait" {
            guard let task=wait else { return TaskNotice(type:"need_target",text:"请说明要等的线路或号码") }
            return TaskNotice(type:"wait_continued",text:(task.notified ? "已经提醒过" : "会继续留意")+task.target)
        }
        version += 1
        switch intent.action {
        case "navigate": navigating=true; return TaskNotice(type:"navigation_started",text:"已开始观察人行道和障碍")
        case "stop": navigating=false; return TaskNotice(type:"navigation_stopped",text:"已停止局部引导")
        case "cancel": wait=nil; return TaskNotice(type:"wait_cancelled",text:"已取消等待")
        case "wait":
            let kind=intent.kind ?? "", ids=Array(Set(TextRules.identifiers(intent.target ?? "")))
            guard ["bus","train","number"].contains(kind),ids.count == 1 else {
                return TaskNotice(type:kind == "light" ? "unsupported_task" : "need_target",
                    text:kind == "light" ? "持续灯色与行进方向关联尚未验证" : "请说明要等的完整线路或号码")
            }
            let target=ids[0],direction=intent.direction ?? ""
            wait=WaitingTask(kind:kind,target:target,direction:direction,version:version,started:now,
                             notified:false,candidate_notified:false)
            return TaskNotice(type:"wait_started",text:"正在等待\(target)\(direction.isEmpty ? "" : "往"+direction+"方向")，识别到后提醒")
        default: return TaskNotice(type:"question_received",text:"正在查看当前画面")
        }
    }
    public func applyBrain(decision:String,taskVersion:UInt64,observation:String,now:SessionTime) -> TaskNotice? {
        guard decision == "MATCH",var task=wait,!task.notified,task.version == taskVersion else { return nil }
        task.notified=true;wait=task
        let text=task.kind == "number" ? "观察到\(task.target)的叫号信息" : "观察到\(task.target)路\(task.kind == "bus" ? "公交车" : "列车")"
        var notice=TaskNotice(type:"target_observed",text:text)
        notice.wait=task;notice.source="multimodal_waiting_brain";notice.observation_s=now;notice.evidence_text=observation
        return notice
    }
    /// Historical OCR/rule comparison. The current session uses applyBrain instead.
    public func observe(texts: [TextEvidence], vehicles: [VehicleObservation], now: SessionTime,
                        audio: String = "", audioStart: SessionTime? = nil) -> TaskNotice? {
        guard var task=wait,!task.notified else { return nil }
        let strong=texts.filter { $0.score >= 0.70 },visual=strong.map(\.text).joined(separator:" ")
        let exactVisual=strong.contains { TextRules.identifiers($0.text).contains(task.target) }
        let exactAudio=TextRules.identifiers(audio).contains(task.target) && (audioStart ?? task.started) >= task.started
        var source:String?,evidence=visual,candidate:String?,vehicleID:String?
        if task.kind == "number" {
            if exactAudio && QueueTextRules.calledInAudio(audio).contains(task.target) { source="audio" }
            else if texts.contains(where: { $0.queue_verification_attempted }) {
                for entry in strong where TextRules.identifiers(entry.text).contains(task.target) {
                    if entry.queue_role == "called" { source="visual";evidence=entry.text;break }
                    if entry.queue_role == "unconfirmed" { candidate=entry.text }
                }
            }
            else if exactVisual && TextRules.matches("号|號|窗口|柜台|櫃台|请|請|就诊|就診|取餐",visual)
                && !TextRules.matches("未叫|等待|候诊|候診|排队|排隊|取号|取號",visual) { source="visual" }
        } else {
            for vehicle in vehicles where vehicle.label == task.kind {
                let associated=texts.filter { ($0.score >= 0.70 || $0.route_verified || $0.route_candidate)
                    && $0.vehicle_label == vehicle.label && $0.vehicle_id == vehicle.observation_id }
                let vehicleText=associated.filter { $0.text_role != "vehicle_body" }.map(\.text).joined(separator:" ")
                for entry in associated where TextRules.identifiers(entry.text).contains(task.target) {
                    if entry.route_candidate { candidate=vehicleText; continue }
                    if entry.route_verification_attempted && !entry.route_verified { continue }
                    if entry.text_role == "vehicle_body" { continue }
                    if TextRules.directionMatches(task.direction,vehicleText) {
                        source="visual"; evidence=vehicleText; vehicleID=vehicle.observation_id; break
                    }
                    candidate=vehicleText
                }
                if source != nil { break }
            }
            if exactAudio && TextRules.matches("到站|进站|進站|到达|到達",audio)
                && !TextRules.matches("未|没|沒|即将|即將|将要|將要|还有|還有|不是|并非|並非",audio)
                && TextRules.directionMatches(task.direction,audio) { source="audio" }
        }
        if let source {
            task.notified=true; wait=task
            let text=task.kind == "number" ? (source == "audio" ? "广播中叫到了" : "叫号屏显示")+task.target
                : "观察到目标"+task.target+(task.direction.isEmpty ? "" : "，方向"+task.direction)
            var result=TaskNotice(type:"target_observed",text:text)
            result.wait=task;result.source=source;result.observation_s=now
            result.evidence_text=source == "audio" ? audio : evidence;result.vehicle_id=vehicleID
            return result
        }
        if let candidate,!task.candidate_notified {
            task.candidate_notified=true;wait=task
            let text=task.kind == "number" ? "看到了\(task.target)，尚未确认是否已经叫到" :
                (task.direction.isEmpty ? "画面中可能是\(task.target)，请调整相机以确认号码"
                : "识别到\(task.target)，尚未确认是否往\(task.direction)方向")
            var result=TaskNotice(type:"target_candidate",text:text)
            result.wait=task;result.source="visual";result.observation_s=now;result.evidence_text=candidate
            return result
        }
        return nil
    }
}
