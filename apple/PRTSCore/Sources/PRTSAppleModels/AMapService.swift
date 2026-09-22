import Foundation
import PRTSContracts

/// Optional network adapter. Key is supplied by the host app, never in the package.
/// Sensor/model processing does not depend on this service being available.
public final class AMapService: MapService {
    private let key:String,city:String
    private let session:URLSession
    public init(key: String, city: String = "") {
        self.key=key;self.city=city
        let config=URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest=8;config.urlCache=nil
        self.session=URLSession(configuration:config)
    }
    private func request(_ endpoint: String, _ parameters: [String:String]) async throws -> [String:Any] {
        guard !key.isEmpty else { throw NavigationError.unavailable("尚未配置高德 Web 服务 Key") }
        var parts=URLComponents(string:"https://restapi.amap.com/"+endpoint)!
        var params=parameters;params["key"]=key;params["output"]="JSON"
        parts.queryItems=params.sorted { $0.key < $1.key }.map { URLQueryItem(name:$0.key,value:$0.value) }
        let data:Data
        do {
            let (body,response)=try await session.data(from:parts.url!)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw NavigationError.unavailable("地图服务暂不可用")
            }
            data=body
        } catch { throw NavigationError.unavailable("地图请求失败，请检查服务连接") }
        guard let payload=try JSONSerialization.jsonObject(with:data) as? [String:Any],
              String(describing:payload["status"] ?? "") == "1" else {
            throw NavigationError.unavailable("地图未返回可用结果，请检查 Key 和服务权限")
        }
        return payload
    }
    private func point(_ text: String) throws -> MapPoint {
        let parts=text.split(separator:",").compactMap { Double($0) }
        guard parts.count == 2 else { throw NavigationError.unavailable("地图坐标格式无效") }
        return MapPoint(longitude:parts[0],latitude:parts[1])
    }
    public func convert(_ point: MapPoint) async throws -> MapPoint {
        if point.crs == "GCJ02" { return point }
        guard point.crs == "WGS84" else { throw NavigationError.unavailable("不支持此坐标系") }
        let payload=try await request("v3/assistant/coordinate/convert",["locations":point.query,"coordsys":"gps"])
        return try self.point((payload["locations"] as? String) ?? "")
    }
    public func search(_ query: String, near: MapPoint?) async throws -> [MapDestination] {
        var params=["keywords":query,"offset":"5","page":"1","extensions":"base"]
        var endpoint="v3/place/text"
        if !city.isEmpty { params["city"]=city;params["citylimit"]="true" }
        if let near {
            params["location"]=try await convert(near).query
            params["radius"]="5000";params["sortrule"]="weight";endpoint="v3/place/around"
        }
        let payload=try await request(endpoint,params)
        let records=(payload["pois"] as? [[String:Any]]) ?? []
        return try records.prefix(5).compactMap { p in
            guard let location=p["location"] as? String,!location.isEmpty else { return nil }
            return MapDestination(id:(p["id"] as? String) ?? "",name:(p["name"] as? String) ?? "",
                point:try point(location),address:(p["address"] as? String) ?? "")
        }
    }
    public func walking(origin: MapPoint, destination: MapDestination) async throws -> WalkingRoute {
        let origin=try await convert(origin)
        guard destination.point.crs == "GCJ02" else { throw NavigationError.unavailable("目的地坐标系无效") }
        // Use the confirmed coordinate. A POI ID can select another entrance.
        let payload=try await request("v5/direction/walking",["origin":origin.query,
            "destination":destination.point.query,"show_fields":"polyline,navi,cost"])
        let container=payload["route"] as? [String:Any]
        let first=(container?["paths"] as? [[String:Any]])?.first
        var steps=[RouteStep]()
        let records=(first?["steps"] as? [[String:Any]]) ?? []
        for item in records {
            let points=try ((item["polyline"] as? String) ?? "").split(separator:";").map { try point(String($0)) }
            if points.count < 2 { continue }
            let navi=(item["navi"] as? [String:Any]) ?? [:]
            let road=(item["road_name"] as? String) ?? (item["road"] as? String) ?? ""
            steps.append(RouteStep(points:points,instruction:(item["instruction"] as? String) ?? "",
                action:(navi["action"] as? String) ?? "",road:road,
                walkType:String(describing:navi["walk_type"] ?? "0")))
        }
        guard !steps.isEmpty else { throw NavigationError.unavailable("步行路线缺少几何数据") }
        return WalkingRoute(destination:destination,steps:steps,provider:"amap")
    }
}
