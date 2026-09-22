import Foundation

/// Target-independent OCR column and announcement semantics.
/// Model OCR quality and field recognition must be tested separately on Apple.
public enum QueueTextRules {
    public static func headingRole(_ input: String) -> String? {
        let text=input.precomposedStringWithCompatibilityMapping.uppercased()
            .replacingOccurrences(of:"[\\s_\\-‐‑–:：]+",with:"",options:.regularExpression)
        let patterns:[(String,String)] = [
            ("waiting","等待|等候|候诊|候診|候餐|未叫|取号|取號|排队|排隊|WAITING|PREPARING|PENDING|NOTREADY"),
            ("called","取餐|取药|取藥|叫号|叫號|当前号|當前號|正在服务|NOWSERVING|NOWCALLING|PICKUP|READYFOR|CALLEDNUMBER"),
            ("counter","窗口|柜台|櫃台|诊室|診室|COUNTER|DESK|ROOM|LOKET|SCHALTER"),
            ("other","价格|價格|金额|金額|汇率|匯率|利率|PRICE|SELL|BUY|CNY|EUR|USD|VALAS|SALDO|BUNGA"),
            ("queue_label","号码|號碼|票号|票號|QUEUENUMBER|TICKETNUMBER|NOMOR|ANTRIAN|ANTREAN|UWNUMMER|^NUMBER$")
        ]
        return patterns.first { TextRules.matches($0.1,text) }?.0
    }
    public static func numericRow(_ text: String) -> Bool {
        let value=text.precomposedStringWithCompatibilityMapping.uppercased()
        if TextRules.matches("(?<![A-Z0-9])[0-9]+\\s*[-/]\\s*[0-9]+",value) { return false }
        return TextRules.matches("[0-9]",value) && TextRules.matches("^[A-Z0-9\\s,，、;/\\-‐‑–]+$",value)
    }
    private static func headingDistance(_ row:TextEvidence,_ heading:TextEvidence,sameLine:Bool=true) -> Float? {
        guard row.box.count == 4,heading.box.count == 4 else { return nil }
        let dx=heading.box[1][0]-heading.box[0][0],dy=heading.box[1][1]-heading.box[0][1]
        let length=max(1,sqrt(dx*dx+dy*dy)),ax=dx/length,ay=dy/length
        func projected(_ e:TextEvidence) -> [Float] {
            let x=e.box.map { $0[0]*ax+$0[1]*ay },y=e.box.map { -$0[0]*ay+$0[1]*ax }
            return [x.min()!,y.min()!,x.max()!,y.max()!]
        }
        let r=projected(row),h=projected(heading),rh=max(1,r[3]-r[1]),hh=max(1,h[3]-h[1])
        let rx=(r[0]+r[2])/2,ry=(r[1]+r[3])/2,hx=(h[0]+h[2])/2,hy=(h[1]+h[3])/2
        let scale=max(rh,hh),horizontal=max(0,min(r[2],h[2])-max(r[0],h[0]))/max(1,min(r[2]-r[0],h[2]-h[0]))
        if abs(ry-hy)<0.65*scale && rx>hx {
            return sameLine && r[0]-h[2]<6*scale ? max(0,r[0]-h[2])/scale+0.1 : nil
        }
        if horizontal>=0.45 && ry>hy && ry-hy<6*scale { return (ry-hy)/scale }
        return nil
    }
    public static func classify(_ entries:[TextEvidence]) -> [TextEvidence] {
        let headers=entries.filter { $0.score>=0.65 && headingRole($0.text) != nil }
        return entries.map { entry in
            var e=entry;e.queue_verification_attempted=true;e.queue_role="other"
            if let own=headingRole(e.text),["counter","waiting","other","queue_label"].contains(own) {
                e.queue_role=own;return e
            }
            guard e.score>=0.7,numericRow(e.text) else { return e }
            let candidates:[(Float,TextEvidence)]=headers.compactMap { h in
                headingDistance(e,h).map { ($0,h) }
            }
            guard let heading=candidates.min(by: { $0.0<$1.0 })?.1,var role=headingRole(heading.text) else { return e }
            if role == "queue_label" {
                let active=headers.contains { h in
                    guard headingRole(h.text) == "counter" else { return false }
                    if headingDistance(heading,h,sameLine:false) != nil { return true }
                    return TextRules.identifiers(h.text).isEmpty && headingDistance(h,heading) != nil
                        && headingDistance(h,heading,sameLine:false) == nil
                }
                role=active ? "called" : "unconfirmed"
            } else if role == "counter" && !TextRules.identifiers(heading.text).isEmpty
                && headingDistance(e,heading,sameLine:false) != nil { role="called" }
            e.queue_role=role;e.queue_heading=heading.text;e.queue_heading_box=heading.box
            return e
        }
    }
    private static func before(_ pattern:String,_ value:String) -> String {
        guard let range=value.range(of:pattern,options:.regularExpression) else { return value }
        return String(value[..<range.lowerBound])
    }
    public static func calledInAudio(_ text:String) -> [String] {
        let separator="\u{001F}"
        let clauses=text.replacingOccurrences(of:"[，,。；;！？!?]+",with:separator,options:.regularExpression).components(separatedBy:separator)
        var result=[String]()
        for clause in clauses {
            if TextRules.matches("未|还没|還沒|尚未|不是|并非|並非|取消|过号|過號|跳过|跳過|等待|排队|排隊|请问|請問|有没有|有沒有|是否|哪里|哪裡",clause) { continue }
            var spoken:String?
            let actions="前往|到|至|去|就诊|就診|取餐|取药|取藥|办理|辦理"
            if let imperative=TextRules.captures("(?:请|請)(.+)",clause).first,
               TextRules.matches(actions,imperative[1]) {
                spoken=before(actions,imperative[1])
                if TextRules.identifiers(spoken!).isEmpty {
                    let prefix=before("请|請",clause)
                    if !TextRules.matches("窗口|柜台|櫃台|诊室|診室",prefix) { spoken=prefix }
                }
            } else if let label=TextRules.captures("(?i)(?:叫到|叫号|叫號|呼叫|NOW\\s+(?:SERVING|CALLING))\\s*(.+)",clause).first {
                spoken=before("(?i)前往|到|至|去|窗口|柜台|櫃台|COUNTER|DESK",label[1])
            }
            if let spoken,!TextRules.matches("(?<![A-Z0-9])[0-9]+\\s*[-/]\\s*[0-9]+",spoken.uppercased()) {
                for id in TextRules.identifiers(spoken) where !result.contains(id) { result.append(id) }
            }
        }
        return result
    }
}
