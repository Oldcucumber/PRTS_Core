import Foundation
import PRTSContracts

public struct LocalGuidance: Codable {
    public var status: String
    public var direction: String
    public var reason: String
    public var text: String
    public var path: [[Double]]
    public var mode="sidewalk"
    public var forward_scan:ForwardScan?=nil
    public let baseline_path: [[Double]]
    public let path_coverage: Double
    public let baseline_coverage: Double
    public let path_heading_deg: Double?
    public let checked_pixels: Int
    public let outside_free_pixels: Int
    public let direction_confirmed: Bool
    public let route_hint_status: String
    public let coordinate_space="image_normalized"
    public let spatial_registration="unavailable"
    public let temporal_method="current_mask_with_direction_hysteresis"
}

/// Recomputes the entire path in the current mask. This Apple source uses temporal
/// direction hysteresis, but does not claim Python's optical-flow registration.
public final class LocalGuide {
    private var lastTime:SessionTime?,lastDirection:String?,pending:String?
    private var pendingSince=0.0,pendingCount=0
    private let forwardHalfAngle:Double,horizontalFOV:Double
    public init(forwardHalfAngle:Double=15,horizontalFOV:Double=60) { self.forwardHalfAngle=forwardHalfAngle;self.horizontalFOV=horizontalFOV }
    public func reset() { lastTime=nil;lastDirection=nil;pending=nil;pendingCount=0 }
    public func update(_ observation: PerceptionObservation, route: RouteUpdate? = nil) throws -> LocalGuidance {
        let grid=observation.semantic,w=grid.width,h=grid.height,t=observation.timestamp
        let nonmonotonic=lastTime != nil && t <= lastTime!
        if nonmonotonic || (lastTime != nil && t-lastTime! > 2) { reset() }
        lastTime=t
        var free=grid.walkable
        for item in observation.detections where item.blocking {
            let b=item.box,pad=Float(0.006)+0.045*(b[3]-b[1])
            let x1=max(0,Int(floor(Double((b[0]-pad)*Float(w)))))
            let x2=min(w,Int(ceil(Double((b[2]+pad)*Float(w)))))
            let y1=max(0,Int(floor(Double((b[1]-0.01)*Float(h)))))
            let y2=min(h,Int(ceil(Double((b[3]+0.025)*Float(h)))))
            if x1 < x2 && y1 < y2 { for y in y1..<y2 { for x in x1..<x2 { free[y*w+x]=0 } } }
        }
        var preference=0,routeState="absent"
        if let route {
            if route.status != "following" { routeState="inactive" }
            else if let bearing=route.relative_bearing_deg,bearing.isFinite {
                if abs(bearing)>75 { routeState="outside_forward_view" }
                else { preference=bearing < -15 ? -1 : bearing > 15 ? 1 : 0;routeState="lateral_preference_only" }
            } else { routeState="camera_heading_unavailable" }
        }
        let baseline=try CorridorPlanner.search(freeMask:grid.walkable,width:w,height:h,preference:preference)
        let corridor=try CorridorPlanner.search(freeMask:free,width:w,height:h,
                                               anchorX:baseline.pixels.first.map { Int($0.x) },preference:preference)
        var path=corridor.normalized,direction="UNKNOWN",status="WAIT",reason="no_continuous_sidewalk",confirmed=false
        if nonmonotonic { path=[];reason="nonmonotonic_timestamp" }
        else if corridor.coverage < 0.55 || corridor.pixels.count < 5 {
            path=[]
            if baseline.coverage >= 0.55 { status="STOP";reason="detector_blocks_corridor" }
        } else if !corridor.valid { path=[];reason="path_validation_failed" }
        else if let heading=corridor.headingDegrees {
            var raw=heading < -18 ? "LEFT" : heading > 18 ? "RIGHT" : "FORWARD"
            if lastDirection == "LEFT" && heading < -10 { raw="LEFT" }
            if lastDirection == "RIGHT" && heading > 10 { raw="RIGHT" }
            if lastDirection == nil || raw == lastDirection { direction=raw;confirmed=true;pending=nil;pendingCount=0 }
            else {
                if pending != raw { pending=raw;pendingSince=t;pendingCount=1 } else { pendingCount += 1 }
                if pendingCount >= 2 && t-pendingSince >= 0.15 { direction=raw;confirmed=true;pending=nil;pendingCount=0 }
            }
            if confirmed { lastDirection=direction }
            status="CANDIDATE";reason=confirmed ? "current_sidewalk" : "direction_confirming"
            if preference != 0 && heading*Double(preference) < -18 {
                status="WAIT";direction="UNKNOWN";confirmed=false;reason="route_corridor_conflict"
                routeState="conflicts_with_visible_corridor";path=[]
            }
        }
        if path.isEmpty { lastDirection=nil;pending=nil;pendingCount=0 }
        var text=direction == "LEFT" ? "候选通道向左延伸" : direction == "RIGHT" ? "候选通道向右延伸"
            : direction == "FORWARD" ? "前方有连续人行道候选通道"
            : path.isEmpty ? "暂未找到连续人行道，请先等待" : "正在确认前方转向"
        if status == "STOP" { text="前方通道被障碍占用，请先停下等待" }
        if reason == "route_corridor_conflict" { text="眼前通道与路线方向不一致，请先等待确认路线" }
        var result=LocalGuidance(status:status,direction:direction,reason:reason,text:text,path:path,
            baseline_path:baseline.normalized,path_coverage:corridor.coverage,baseline_coverage:baseline.coverage,
            path_heading_deg:corridor.headingDegrees,checked_pixels:corridor.checkedPixels,
            outside_free_pixels:corridor.outsideMaskPixels,direction_confirmed:confirmed,route_hint_status:routeState)
        if reason == "no_continuous_sidewalk" {
            let scan=ForwardScan.observe(observation,halfAngle:forwardHalfAngle,horizontalFOV:horizontalFOV)
            result.mode="free_forward";result.forward_scan=scan
            result.status=scan.obstacles.isEmpty ? "FREE" : "STOP"
            result.reason=scan.obstacles.isEmpty ? "free_forward" : "forward_obstacle"
            result.text=scan.obstacles.isEmpty ? "自由前进模式，持续观察正前方障碍" : "正前方检测到障碍，请留意"
        }
        return result
    }
}
