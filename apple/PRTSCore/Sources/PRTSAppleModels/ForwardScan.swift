import Foundation

public struct ForwardObstacle: Codable {
    public let label:String,box:[Double],source:String
}
public struct ForwardScan: Codable {
    public let half_angle_deg:Double,horizontal_fov_deg:Double
    public let angle_source="assumed_horizontal_fov",coordinate_space="image_normalized"
    public let sector_box:[Double],obstacles:[ForwardObstacle],assessment:String
    public let metric_distance_available=false

    static func observe(_ observation:PerceptionObservation,halfAngle:Double,horizontalFOV:Double) -> ForwardScan {
        let fraction=tan(halfAngle * .pi/180)/tan(horizontalFOV * .pi/360)
        let left=max(0,0.5-fraction/2),right=min(1,0.5+fraction/2),top=0.55
        var obstacles=[ForwardObstacle]()
        for item in observation.detections where item.blocking && item.score >= 0.35 {
            let b=item.box.map(Double.init)
            let overlap=max(0,min(right,b[2])-max(left,b[0]))*max(0,min(1,b[3])-max(top,b[1]))
            if overlap/((right-left)*(1-top)) >= 0.01 {
                obstacles.append(ForwardObstacle(label:item.label,box:b,source:"detector"))
            }
        }
        let grid=observation.semantic,w=grid.width,h=grid.height
        let x0=Int(left*Double(w)),x1=min(w,Int(ceil(right*Double(w)))),y0=Int(top*Double(h))
        let minimum=max(8,Int(Double((x1-x0)*(h-y0))*0.01))
        let ids=Set(grid.classNames.filter { obstacleNames.contains($0.value.lowercased()) }.map(\.key))
        var occupied=[Bool](repeating:false,count:w*h),visited=occupied
        for y in y0..<h { for x in x0..<x1 {
            let i=y*w+x;occupied[i]=ids.contains(Int(grid.classIDs[i])) && grid.confidence[i] >= 0.60
        } }
        for seed in 0..<occupied.count where occupied[seed] && !visited[seed] {
            var queue=[seed],index=0,minX=seed%w,maxX=minX,minY=seed/w,maxY=minY,labels=Set<String>()
            visited[seed]=true
            while index < queue.count {
                let i=queue[index],x=i%w,y=i/w;index+=1
                minX=min(minX,x);maxX=max(maxX,x);minY=min(minY,y);maxY=max(maxY,y)
                labels.insert(grid.classNames[Int(grid.classIDs[i])] ?? "unknown")
                for dy in -1...1 { for dx in -1...1 {
                    let nx=x+dx,ny=y+dy
                    if nx >= 0 && nx < w && ny >= 0 && ny < h {
                        let j=ny*w+nx
                        if occupied[j] && !visited[j] { visited[j]=true;queue.append(j) }
                    }
                } }
            }
            if queue.count >= minimum {
                obstacles.append(ForwardObstacle(label:labels.sorted().joined(separator:"/"),
                    box:[Double(minX)/Double(w),Double(minY)/Double(h),Double(maxX+1)/Double(w),Double(maxY+1)/Double(h)],source:"semantic"))
            }
        }
        return ForwardScan(half_angle_deg:halfAngle,horizontal_fov_deg:horizontalFOV,
            sector_box:[left,top,right,1],obstacles:obstacles,
            assessment:obstacles.isEmpty ? "no_obstacle_detected" : "obstacle_detected")
    }
    // Kept aligned with Python FIXED and DYNAMIC semantic groups.
    private static let obstacleNames:Set<String>=["animal", "banner", "barrier", "bench", "bicycle", "bicyclist", "bike rack", "billboard", "bird", "boat", "bridge", "building", "bus", "bus stop", "car", "caravan", "catch basin", "cctv camera", "construction-bridge", "construction-building", "construction-door", "construction-fenceguardrail", "construction-stairs", "construction-tunnel", "construction-wall", "curb", "ego vehicle", "fence", "fire hydrant", "flat-curb", "ground animal", "guard rail", "guard rail/road barrier", "hand rail", "human-person", "human-rider", "junction box", "mailbox", "motorcycle", "motorcyclist", "nature-vegetation", "object-pole", "object-trafficlight", "object-trafficsign", "obstacle", "on rails", "opening-door", "opening-gate", "other rider", "other vehicle", "pedestrian", "person", "phone booth", "pole", "pothole", "rider", "stairs", "street light", "traffic light", "traffic sign", "traffic sign (back)", "traffic sign (front)", "traffic sign frame", "trailer", "train", "trash can", "tree", "tree trunk", "truck", "tunnel", "utility pole", "vegetation", "vehicle", "vehicle-bicycle", "vehicle-bus", "vehicle-car", "vehicle-caravan", "vehicle-cartrailer", "vehicle-motorcycle", "vehicle-tramtrain", "vehicle-truck", "wall", "wall/fence", "water", "water body", "wheeled slow"]
}
