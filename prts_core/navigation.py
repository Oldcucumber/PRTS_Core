"""Destination confirmation and walking-route progress, independent of model runtimes.

All route geometry carries its coordinate system. Image-space guidance is a separate
consumer of route hints; GPS coordinates are never projected directly onto an image.
"""
from dataclasses import asdict, dataclass, field
import json
import math
import os
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass(frozen=True)
class Point:
    longitude: float
    latitude: float
    crs: str = 'GCJ02'

    def query(self):
        return f'{self.longitude:.6f},{self.latitude:.6f}'


@dataclass(frozen=True)
class Location:
    point: Point
    timestamp_s: float
    accuracy_m: float
    camera_heading_deg: float | None = None
    heading_accuracy_deg: float | None = None


@dataclass
class Destination:
    id: str
    name: str
    point: Point
    address: str = ''


@dataclass
class Step:
    points: list[Point]
    instruction: str = ''
    action: str = ''
    road: str = ''
    walk_type: str = '0'


@dataclass
class WalkingRoute:
    destination: Destination
    steps: list[Step]
    provider: str
    provenance: dict = field(default_factory=dict)


class MapUnavailable(RuntimeError):
    pass


def distance(a, b):
    if a.crs != b.crs:
        raise ValueError('Location and route coordinate systems differ')
    lat = math.radians((a.latitude + b.latitude) / 2)
    dx = math.radians(b.longitude - a.longitude) * math.cos(lat) * 6371008.8
    dy = math.radians(b.latitude - a.latitude) * 6371008.8
    return math.hypot(dx, dy)


def bearing(a, b):
    lat = math.radians((a.latitude + b.latitude) / 2)
    return math.degrees(math.atan2((b.longitude-a.longitude)*math.cos(lat), b.latitude-a.latitude)) % 360


def signed_angle(angle):
    return (angle + 180) % 360 - 180


def parse_point(value, crs='GCJ02'):
    lon, lat = map(float, value.split(','))
    return Point(lon, lat, crs)


def parse_amap_walking(payload, destination):
    paths = payload.get('route', {}).get('paths', [])
    if str(payload.get('status')) != '1' or not paths:
        raise MapUnavailable('高德未返回可用步行路线')
    steps = []
    for item in paths[0].get('steps', []):
        points = [parse_point(p) for p in item.get('polyline', '').split(';') if p]
        if len(points) < 2:
            continue
        navi = item.get('navi') or {}
        steps.append(Step(points, item.get('instruction', ''), navi.get('action', ''),
                          item.get('road_name', item.get('road', '')), str(navi.get('walk_type', '0'))))
    if not steps:
        raise MapUnavailable('步行路线缺少几何数据，需要请求 polyline 字段')
    return WalkingRoute(destination, steps, 'amap', {'api': 'v5/direction/walking', 'crs': 'GCJ02'})


class AMapProvider:
    """Official web-service adapter. Keys and complete request URLs are never logged."""
    def __init__(self, key=None, city='', timeout_s=8):
        self.key = key or os.environ.get('AMAP_WEB_KEY', '')
        self.city = city
        self.timeout_s = timeout_s

    def _request(self, endpoint, **params):
        if not self.key:
            raise MapUnavailable('尚未配置高德 Web 服务 Key')
        query = urlencode(dict(params, key=self.key, output='JSON'))
        try:
            with urlopen('https://restapi.amap.com/' + endpoint + '?' + query, timeout=self.timeout_s) as r:
                payload = json.load(r)
        except (OSError, ValueError):
            raise MapUnavailable('地图请求失败，请检查服务连接') from None
        if str(payload.get('status')) != '1':
            raise MapUnavailable('地图服务拒绝本次请求，状态码：' + str(payload.get('infocode', 'unknown')))
        return payload

    def search(self, query, location=None):
        params = dict(keywords=query, offset=5, page=1, extensions='base')
        endpoint='v3/place/text'
        if self.city:
            params.update(city=self.city, citylimit='true')
        if location is not None:
            point=self.convert(location.point if isinstance(location,Location) else location)
            params.update(location=point.query(),radius=5000,sortrule='weight')
            endpoint='v3/place/around'
        payload = self._request(endpoint, **params)
        return [Destination(p['id'], p['name'], parse_point(p['location']),
                            p.get('address') if isinstance(p.get('address'), str) else '')
                for p in payload.get('pois', []) if p.get('location')][:5]

    def convert(self, point):
        if point.crs == 'GCJ02':
            return point
        if point.crs != 'WGS84':
            raise ValueError('AMap adapter supports WGS84 and GCJ02')
        result = self._request('v3/assistant/coordinate/convert', locations=point.query(), coordsys='gps')
        return parse_point(result['locations'])

    def walking(self, origin, destination):
        origin = self.convert(origin)
        payload = self._request('v5/direction/walking', origin=origin.query(),
                                destination=destination.point.query(),
                                show_fields='polyline,navi,cost')
        # The confirmed coordinate is authoritative. Supplying destination_id
        # too can choose a different POI entrance (86 m away in the live audit).
        return parse_amap_walking(payload, destination)


class RouteProgress:
    def __init__(self, route):
        self.route = route
        self.segments = []
        length = 0.
        last = None
        for i, step in enumerate(route.steps):
            if last is not None and distance(last, step.points[0]) > 3:
                raise ValueError('Route steps are disconnected')
            for a, b in zip(step.points, step.points[1:]):
                size = distance(a, b)
                if size < .05:
                    continue
                self.segments.append((a, b, length, size, i))
                length += size
            last = step.points[-1]
        if not self.segments:
            raise ValueError('Walking route has no nonzero segments')
        self.length = length
        self.progress = 0.
        self.last_time = None
        self.arrived = False

    def update(self, fix, now):
        if fix.point.crs != self.segments[0][0].crs:
            return dict(status='coordinate_mismatch', text='位置与路线的坐标系不一致')
        if now - fix.timestamp_s > 5 or fix.timestamp_s > now + .5:
            return dict(status='location_stale', text='位置尚未更新')
        if fix.accuracy_m < 0 or fix.accuracy_m > 25:
            return dict(status='location_uncertain', text='定位精度不足，暂不更新转向')
        if self.last_time is not None and fix.timestamp_s < self.last_time:
            return dict(status='location_stale', text='忽略早于当前进度的位置')
        candidates = []
        for a, b, start, length, step in self.segments:
            latitude = math.radians((a.latitude+b.latitude)/2)
            dx, dy = (b.longitude-a.longitude)*math.cos(latitude), b.latitude-a.latitude
            px, py = (fix.point.longitude-a.longitude)*math.cos(latitude), fix.point.latitude-a.latitude
            t = min(1., max(0., (px*dx+py*dy)/(dx*dx+dy*dy)))
            projected = Point(a.longitude+(b.longitude-a.longitude)*t, a.latitude+(b.latitude-a.latitude)*t, a.crs)
            along = start+length*t
            # Do not jump to a distant branch when a route crosses itself.
            if self.last_time is not None:
                advance = max(15., (fix.timestamp_s-self.last_time)*3 + 2*fix.accuracy_m)
                if along < self.progress-12 or along > self.progress+advance:
                    continue
            candidates.append((distance(fix.point, projected), abs(along-self.progress), along, step, a, b))
        if not candidates:
            return dict(status='relocalize', text='位置变化过大，需要重新定位')
        cross_track, _, along, step_index, a, b = min(candidates, key=lambda x: (round(x[0], 1), x[1]))
        if cross_track > max(12., 2*fix.accuracy_m):
            return dict(status='off_route', cross_track_m=cross_track, text='已偏离原路线，需要重新规划')
        self.progress = max(0., along)
        self.last_time = fix.timestamp_s
        remaining = self.length-along
        end = self.segments[-1][1]
        route_ended = remaining <= 5 and distance(fix.point, end) <= 6 and fix.accuracy_m <= 8
        destination_gap=distance(fix.point,self.route.destination.point)
        self.arrived = route_ended and destination_gap <= max(15.,2*fix.accuracy_m)
        step = self.route.steps[step_index]
        step_end = max(s[2]+s[3] for s in self.segments if s[4] == step_index)
        to_turn = max(0., step_end-along)
        heading = bearing(a, b)
        relative = None
        if fix.camera_heading_deg is not None and fix.heading_accuracy_deg is not None and 0 <= fix.heading_accuracy_deg <= 25:
            relative = signed_angle(heading-fix.camera_heading_deg)
        next_step = self.route.steps[step_index+1] if step_index+1 < len(self.route.steps) else None
        action = step.action
        if not action and next_step:
            delta = signed_angle(bearing(next_step.points[0], next_step.points[-1])-heading)
            action = '右转' if delta > 30 else '左转' if delta < -30 else '直行'
        text = '已到达目的地附近' if self.arrived else (
            f'沿当前路线前行，约{round(to_turn)}米后{action}' if next_step and to_turn <= 20
            else step.instruction or '沿当前路线前行')
        status='arrived' if self.arrived else 'following'
        if route_ended and not self.arrived:
            status='route_end_unconfirmed';text='已到达地图路线终点，尚未确认目的地位置'
        return dict(status=status, step_index=step_index,
                    progress_m=along, remaining_m=remaining, cross_track_m=cross_track,
                    distance_to_destination_m=destination_gap,
                    distance_to_maneuver_m=to_turn, desired_heading_deg=heading,
                    relative_bearing_deg=relative, action=action, walk_type=step.walk_type,
                    text=text, coordinate_frame='compass', position_accuracy_m=fix.accuracy_m)


class Navigation:
    """State transitions stay synchronous; the runtime calls network work off vision."""
    def __init__(self, provider=None):
        self.provider = provider or AMapProvider()
        self.candidates = []
        self.route = None
        self.tracker = None
        self.pending_wait = None
        self.candidate_wait = None
        self.version = 0

    def search(self, query, location=None, arrival_wait=None):
        self.version += 1
        self.candidates = self.provider.search(query, location)
        self.candidate_wait = arrival_wait
        choices = [dict(index=i+1, **asdict(p)) for i, p in enumerate(self.candidates)]
        text = '；'.join(f'{p["index"]}，{p["name"]}，{p["address"]}' for p in choices)
        return dict(type='destination_candidates', candidates=choices, needs_confirmation=True,
                    text=('请选择目的地：'+text) if choices else '没有找到匹配地点，请换一个名称')

    def confirm(self, index, origin):
        if not 1 <= index <= len(self.candidates):
            return dict(type='need_destination_choice', text='请选择候选列表中的一个地点')
        route = self.provider.walking(origin, self.candidates[index-1])
        tracker = RouteProgress(route)
        self.route,self.tracker = route,tracker
        self.pending_wait,self.candidate_wait = self.candidate_wait,None
        self.candidates = []
        self.version += 1
        return dict(type='route_started', route=asdict(self.route), route_version=self.version,
                    text='开始前往'+self.route.destination.name)

    def cancel(self):
        self.candidates = []
        self.route = None
        self.tracker = None
        self.pending_wait = None
        self.candidate_wait = None
        self.version += 1

    def update(self, fix, now):
        if self.tracker is None:
            return None
        result = self.tracker.update(fix, now)
        result.update(type='route_progress', route_version=self.version)
        if result['status'] == 'arrived' and self.pending_wait:
            result['arrival_wait'] = self.pending_wait
            self.pending_wait = None
        return result
