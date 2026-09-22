import Foundation

public struct MapPoint: Codable, Equatable {
    public let longitude: Double
    public let latitude: Double
    public let crs: String
    public init(longitude: Double, latitude: Double, crs: String = "GCJ02") {
        self.longitude=longitude; self.latitude=latitude; self.crs=crs
    }
    public var query: String { String(format:"%.6f,%.6f",locale:Locale(identifier:"en_US_POSIX"),longitude,latitude) }
    public func distance(to b: MapPoint) -> Double {
        let radians=Double.pi/180,latitude=(self.latitude+b.latitude)/2*radians
        return hypot((b.longitude-longitude)*radians*cos(latitude),(b.latitude-self.latitude)*radians)*6371008.8
    }
    public func bearing(to b: MapPoint) -> Double {
        let angle=atan2((b.longitude-longitude)*cos((latitude+b.latitude)/2 * .pi/180),b.latitude-latitude)*180 / .pi
        return (angle+360).truncatingRemainder(dividingBy:360)
    }
    public static func signedAngle(_ degrees: Double) -> Double {
        let r=(degrees+180).truncatingRemainder(dividingBy:360)
        return (r < 0 ? r+360 : r)-180
    }
}

public struct LocationFix: Codable {
    public let point: MapPoint
    public let timestamp_s: SessionTime
    public let accuracy_m: Double
    public let camera_heading_deg: Double?
    public let heading_accuracy_deg: Double?
    public init(point: MapPoint, timestamp: SessionTime, accuracy: Double,
                cameraHeading: Double? = nil, headingAccuracy: Double? = nil) {
        self.point=point;self.timestamp_s=timestamp;self.accuracy_m=accuracy
        self.camera_heading_deg=cameraHeading;self.heading_accuracy_deg=headingAccuracy
    }
}
public struct MapDestination: Codable {
    public let id: String
    public let name: String
    public let point: MapPoint
    public let address: String
    public init(id: String, name: String, point: MapPoint, address: String = "") {
        self.id=id;self.name=name;self.point=point;self.address=address
    }
}
public struct RouteStep: Codable {
    public let points: [MapPoint]
    public let instruction: String
    public let action: String
    public let road: String
    public let walk_type: String
    public init(points: [MapPoint], instruction: String = "", action: String = "", road: String = "", walkType: String = "0") {
        self.points=points;self.instruction=instruction;self.action=action;self.road=road;self.walk_type=walkType
    }
}
public struct WalkingRoute: Codable {
    public let destination: MapDestination
    public let steps: [RouteStep]
    public let provider: String
    public init(destination: MapDestination, steps: [RouteStep], provider: String) {
        self.destination=destination;self.steps=steps;self.provider=provider
    }
}
public struct RouteUpdate: Codable {
    public var status: String
    public var text: String
    public var step_index: Int?
    public var progress_m: Double?
    public var remaining_m: Double?
    public var cross_track_m: Double?
    public var distance_to_destination_m: Double?
    public var distance_to_maneuver_m: Double?
    public var desired_heading_deg: Double?
    public var relative_bearing_deg: Double?
    public var action: String?
    public var walk_type: String?
    public var position_accuracy_m: Double?
    public var coordinate_frame="compass"
    public init(status: String, text: String) { self.status=status;self.text=text }
}
public enum NavigationError: Error { case invalidRoute(String), unavailable(String) }

/// Geometry and progress only. A compass preference is never a camera projection.
public final class RouteTracker {
    private struct Segment { let a:MapPoint;let b:MapPoint;let start:Double;let length:Double;let step:Int }
    public let route: WalkingRoute
    private var segments=[Segment]()
    public private(set) var length=0.0
    private var progress=0.0,lastTime:SessionTime?
    public init(route: WalkingRoute) throws {
        self.route=route
        var last:MapPoint?
        for (i,step) in route.steps.enumerated() {
            guard step.points.count >= 2 else { throw NavigationError.invalidRoute("路线步骤缺少点串") }
            if let last,last.crs != step.points[0].crs || last.distance(to:step.points[0]) > 3 {
                throw NavigationError.invalidRoute("路线步骤不连通")
            }
            for (a,b) in zip(step.points,step.points.dropFirst()) {
                guard a.crs == b.crs else { throw NavigationError.invalidRoute("路线坐标系混用") }
                let size=a.distance(to:b)
                if size >= 0.05 { segments.append(Segment(a:a,b:b,start:length,length:size,step:i));length += size }
            }
            last=step.points.last
        }
        guard !segments.isEmpty,route.destination.point.crs == segments[0].a.crs else {
            throw NavigationError.invalidRoute("路线为空或目的地坐标系不一致")
        }
    }
    public func update(_ fix: LocationFix, now: SessionTime) -> RouteUpdate {
        if fix.point.crs != segments[0].a.crs { return RouteUpdate(status:"coordinate_mismatch",text:"位置与路线的坐标系不一致") }
        if now-fix.timestamp_s > 5 || fix.timestamp_s > now+0.5 || (lastTime != nil && fix.timestamp_s < lastTime!) {
            return RouteUpdate(status:"location_stale",text:"位置尚未更新")
        }
        if fix.accuracy_m < 0 || fix.accuracy_m > 25 { return RouteUpdate(status:"location_uncertain",text:"定位精度不足，暂不更新转向") }
        struct Candidate { let cross:Double;let delta:Double;let along:Double;let segment:Segment }
        var candidates=[Candidate]()
        for segment in segments {
            let a=segment.a,b=segment.b,c=cos((a.latitude+b.latitude)/2 * .pi/180)
            let dx=(b.longitude-a.longitude)*c,dy=b.latitude-a.latitude
            let px=(fix.point.longitude-a.longitude)*c,py=fix.point.latitude-a.latitude
            let t=min(1,max(0,(px*dx+py*dy)/(dx*dx+dy*dy))),along=segment.start+segment.length*t
            if let lastTime {
                let advance=max(15,(fix.timestamp_s-lastTime)*3+2*fix.accuracy_m)
                if along < progress-12 || along > progress+advance { continue }
            }
            let projected=MapPoint(longitude:a.longitude+(b.longitude-a.longitude)*t,
                                   latitude:a.latitude+(b.latitude-a.latitude)*t,crs:a.crs)
            candidates.append(Candidate(cross:fix.point.distance(to:projected),delta:abs(along-progress),along:along,segment:segment))
        }
        guard let best=candidates.min(by: { a,b in
            let ar=(a.cross*10).rounded(.toNearestOrEven),br=(b.cross*10).rounded(.toNearestOrEven)
            return ar == br ? a.delta < b.delta : ar < br
        }) else { return RouteUpdate(status:"relocalize",text:"位置变化过大，需要重新定位") }
        if best.cross > max(12,2*fix.accuracy_m) {
            var result=RouteUpdate(status:"off_route",text:"已偏离原路线，需要重新规划")
            result.cross_track_m=best.cross;return result
        }
        progress=max(0,best.along);lastTime=fix.timestamp_s
        let remaining=length-progress,destinationGap=fix.point.distance(to:route.destination.point)
        let ended=remaining <= 5 && fix.point.distance(to:segments.last!.b) <= 6 && fix.accuracy_m <= 8
        let arrived=ended && destinationGap <= max(15,2*fix.accuracy_m)
        let index=best.segment.step,step=route.steps[index]
        let stepEnd=segments.filter { $0.step == index }.map { $0.start+$0.length }.max()!
        let toTurn=max(0,stepEnd-progress),heading=best.segment.a.bearing(to:best.segment.b)
        let next=index+1 < route.steps.count ? route.steps[index+1] : nil
        var action=step.action
        if action.isEmpty,let next {
            let delta=MapPoint.signedAngle(next.points[0].bearing(to:next.points.last!)-heading)
            action=delta > 30 ? "右转" : delta < -30 ? "左转" : "直行"
        }
        let text=arrived ? "已到达目的地附近" : ended ? "已到达地图路线终点，尚未确认目的地位置"
            : next != nil && toTurn <= 20 ? "沿当前路线前行，约\(Int(toTurn.rounded(.toNearestOrEven)))米后\(action)"
            : step.instruction.isEmpty ? "沿当前路线前行" : step.instruction
        var result=RouteUpdate(status:arrived ? "arrived" : ended ? "route_end_unconfirmed" : "following",text:text)
        result.step_index=index;result.progress_m=progress;result.remaining_m=remaining;result.cross_track_m=best.cross
        result.distance_to_destination_m=destinationGap;result.distance_to_maneuver_m=toTurn
        result.desired_heading_deg=heading;result.action=action;result.walk_type=step.walk_type;result.position_accuracy_m=fix.accuracy_m
        if let direction=fix.camera_heading_deg,let accuracy=fix.heading_accuracy_deg,accuracy >= 0 && accuracy <= 25 {
            result.relative_bearing_deg=MapPoint.signedAngle(heading-direction)
        }
        return result
    }
}

public protocol MapService {
    func convert(_ point: MapPoint) async throws -> MapPoint
    func search(_ query: String, near: MapPoint?) async throws -> [MapDestination]
    func walking(origin: MapPoint, destination: MapDestination) async throws -> WalkingRoute
}
