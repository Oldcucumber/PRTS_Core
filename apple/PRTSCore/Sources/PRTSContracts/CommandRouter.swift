import Foundation

/// Explicit controls use task semantics; free-form questions use the local model.
public enum CommandRouter {
    public static func parse(_ input: String, waiting: WaitingTask? = nil, hasCandidates: Bool = false) -> TaskIntent? {
        let t=TextRules.normalized(input).replacingOccurrences(of:"\\s+",with:"",options:.regularExpression)
        func has(_ pattern:String) -> Bool { TextRules.matches(pattern,t) }
        if waiting != nil && has("^(?:叫到时|到时|到了|来了)(?:再)?(?:提醒|告诉|通知)我[。！!，,]?$") {
            return TaskIntent(action:"continue_wait")
        }
        if has("(不要|别).{0,2}(停止|暂停).{0,5}(导航|指路|引导|方向提示)") { return TaskIntent(action:"navigate") }
        if has("(不要|不用|别|停止|暂停|取消).{0,8}(导航|指路|带路|引导|方向提示|提示方向|报方向|方向指引)") { return TaskIntent(action:"stop") }
        if hasCandidates {
            if let item=TextRules.captures("^(?:选|选择|就去|就是|确认|第)?([一二三四五12345])(?:个|处|家)?[。！!，,]?$",t).first {
                let index=Int(item[1]) ?? ((Array("一二三四五").firstIndex(of:item[1].first!) ?? -1)+1)
                return TaskIntent(action:"confirm_destination",index:index)
            }
            if has("^(确认|就这个|就是这个|好的|可以)[。！!，,]?$") { return TaskIntent(action:"confirm_destination",index:0) }
        }
        if let destination=TextRules.captures("(?:带我去|导航到|我想要去|我想去|我要去|带我到|前往)(.+)",t).first,
           !has("不想去|不要去|不是去") {
            let split=try! NSRegularExpression(pattern:"[，,。；;]|到了(?:以后|之后)?|到站后")
            let raw=destination[1],range=NSRange(raw.startIndex...,in:raw)
            var name=raw,arrival:WaitRequest?
            if let first=split.firstMatch(in:raw,range:range) {
                let ns=raw as NSString;name=ns.substring(to:first.range.location)
                let tail=ns.substring(from:first.range.location+first.range.length)
                if let next=parse(tail,waiting:waiting),next.action == "wait" {
                    arrival=WaitRequest(kind:next.kind ?? "",target:next.target ?? "",direction:next.direction ?? "")
                }
            }
            name=name.replacingOccurrences(of:"[吧啊呀呢？?！!。]+$",with:"",options:.regularExpression)
            if !name.isEmpty { return TaskIntent(action:"destination",query:name,arrivalWait:arrival) }
        }
        let isWaiting=has("等待|等候|提醒|通知|留意|盯着|叫到|到站|来了|来时|到了|到时|等.{0,15}(路|号|车|灯|线)")
        if !isWaiting && has("能不能走|可以.{0,18}(走|通行|通过).{0,2}[吗么]?|能否.{0,8}(走|通行)|能走吗")
            && !has("开始导航|恢复导航|开启导航") { return TaskIntent(action:"ask") }
        if has("取消|撤销|不等|别等|停止等待|停止等候|不用.{0,8}(提醒|通知)|不要.{0,8}(提醒|通知)") { return TaskIntent(action:"cancel") }
        if has("停下|停一停|暂停|停止.{0,5}(导航|指路|引导|走)|别.{0,4}(走|指路|导航)|不要.{0,5}(走|指路|导航)") { return TaskIntent(action:"stop") }
        if !isWaiting && has("开始|继续|恢复|开启|带我|帮我|往前|向前") && has("导航|带路|指路|引导|走|前进") { return TaskIntent(action:"navigate") }
        var kind:String?
        if has("叫号|叫到|排号|排队|取餐|就诊|候诊|挂号|窗口|柜台|我的号") { kind="number" }
        else if has("公交|公车|巴士|公共汽车|路车|路的车") { kind="bus" }
        else if has("地铁|列车|火车|号线|轻轨") { kind="train" }
        else if has("红绿灯|信号灯|绿灯|红灯|黄灯") { kind="light" }
        let modify=has("改成|换成|改为|改等|换等|换一下")
        if modify && kind == nil { kind=waiting?.kind }
        if let kind,isWaiting || modify {
            var target=""
            if kind == "light" { target=TextRules.captures("(绿|红|黄)灯",t).last?[1] ?? "" }
            else {
                let suffix=kind == "bus" ? "路" : kind == "train" ? "号线|线" : "号"
                let chunks=TextRules.captures("([A-Za-z夜快特临专]*[0-9零〇一幺二两三四五六七八九十百千]+[A-Za-z]*)(?:"+suffix+")",t)
                var choices=chunks.compactMap { TextRules.identifiers($0[1]).first }
                if choices.isEmpty { choices=TextRules.identifiers(t) }
                target=modify ? (choices.last ?? "") : Set(choices).count == 1 ? choices[0] : ""
            }
            let direction=["bus","train"].contains(kind)
                ? (TextRules.captures("(?:开往|往|去往)(.+?)(?:方向|的|[，,。]|$)",t).first?[1] ?? "") : ""
            return TaskIntent(action:"wait",kind:kind,target:target,direction:direction)
        }
        if isWaiting && has("等|提醒|通知|留意") { return TaskIntent(action:"wait",kind:"",target:"") }
        if has("什么|哪里|哪儿|在哪|写的|写着|读一下|读出来|看一下|看看|能不能|可以|有没有|前面|眼前|周围") { return TaskIntent(action:"ask") }
        return nil
    }
}
