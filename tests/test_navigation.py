"""Constructed route tests exercise geometry/state, not live-map accuracy."""
import unittest
from prts_core.navigation import (Point, Location, Destination, Step, WalkingRoute,
                                  RouteProgress, Navigation, parse_amap_walking, AMapProvider, MapUnavailable)


class RouteTests(unittest.TestCase):
    def route(self):
        a, b, c = Point(114., 30.), Point(114., 30.0005), Point(114.0005, 30.0005)
        return WalkingRoute(Destination('fixture', '测试站', c),
                            [Step([a,b]), Step([b,c])], 'constructed_fixture')

    def test_progress_turn_and_arrival(self):
        tracker = RouteProgress(self.route())
        start = tracker.update(Location(Point(114.,30.),0,2,0,3),0)
        self.assertEqual(start['status'],'following')
        turn = tracker.update(Location(Point(114.,30.0004),30,2,0,3),30)
        self.assertEqual(turn['action'],'右转')
        self.assertLess(turn['distance_to_maneuver_m'],12)
        self.assertEqual(turn['relative_bearing_deg'],0)
        end = tracker.update(Location(Point(114.0005,30.0005),70,2,90,3),70)
        self.assertEqual(end['status'],'arrived')

    def test_bad_coordinates_stale_fix_and_off_route(self):
        tracker = RouteProgress(self.route())
        self.assertEqual(tracker.update(Location(Point(114,30,'WGS84'),0,2),0)['status'],'coordinate_mismatch')
        self.assertEqual(tracker.update(Location(Point(114,30),0,2),8)['status'],'location_stale')
        self.assertEqual(tracker.update(Location(Point(114,30),0,50),0)['status'],'location_uncertain')
        self.assertEqual(tracker.update(Location(Point(114.002,30),0,2),0)['status'],'off_route')

    def test_jump_cannot_finish_route(self):
        tracker = RouteProgress(self.route())
        tracker.update(Location(Point(114,30),0,2),0)
        result = tracker.update(Location(Point(114.0005,30.0005),.1,2),.1)
        self.assertNotEqual(result['status'],'arrived')

    def test_snapped_route_end_is_not_the_confirmed_destination(self):
        route=self.route()
        route.destination=Destination('offset','站点',Point(114.0005,30.0013))
        class Provider:
            def search(self,*args):return [route.destination]
            def walking(self,*args):return route
        nav=Navigation(Provider());nav.search('站点',arrival_wait={'kind':'bus','target':'R7'})
        nav.confirm(1,Point(114.,30.))
        end=route.steps[-1].points[-1]
        event=nav.update(Location(end,80,2),80)
        self.assertEqual(event['status'],'route_end_unconfirmed')
        self.assertGreater(event['distance_to_destination_m'],80)
        self.assertNotIn('arrival_wait',event)
        self.assertIsNotNone(nav.pending_wait)

    def test_amap_parser_requests_geometric_evidence(self):
        route = self.route()
        payload = {'status':'1','route':{'paths':[{'steps':[
            {'polyline':'114,30;114,30.0005','instruction':'向北','navi':{'action':'右转','walk_type':'0'}},
            {'polyline':'114,30.0005;114.0005,30.0005','instruction':'向东'}]}]}}
        actual = parse_amap_walking(payload,route.destination)
        self.assertEqual(len(actual.steps),2)
        self.assertEqual(actual.steps[0].action,'右转')
        payload['route']['paths'][0]['steps']=[{'instruction':'没有几何数据'}]
        with self.assertRaises(MapUnavailable):parse_amap_walking(payload,route.destination)

    def test_unconfirmed_destination_cannot_change_current_arrival_wait(self):
        original=self.route()
        next_route=WalkingRoute(Destination('next','下一站',Point(114.001,30.0005)),
                               [Step([original.destination.point,Point(114.001,30.0005)])],'constructed_fixture')
        class Provider:
            def search(self,query,location):return [original.destination if query=='原站' else next_route.destination]
            def walking(self,origin,destination):return original if destination.id=='fixture' else next_route
        nav=Navigation(Provider())
        nav.search('原站',arrival_wait={'kind':'bus','target':'R7A'})
        nav.confirm(1,Point(114,30))
        nav.search('下一站',arrival_wait={'kind':'bus','target':'K203'})
        event=nav.update(Location(original.destination.point,70,2),70)
        self.assertEqual(event['arrival_wait']['target'],'R7A')
        self.assertEqual(nav.candidate_wait['target'],'K203')
        nav.confirm(1,original.destination.point)
        event=nav.update(Location(next_route.destination.point,100,2),100)
        self.assertEqual(event['arrival_wait']['target'],'K203')

    def test_choice_required_and_arrival_wait_once(self):
        route = self.route()
        class Provider:
            calls = 0
            def search(self,query,location): return [route.destination,Destination('other','同名站',route.destination.point)]
            def walking(self,origin,destination): self.calls+=1; return route
        provider = Provider(); nav = Navigation(provider)
        event = nav.search('测试站',arrival_wait={'kind':'bus','target':'17'})
        self.assertTrue(event['needs_confirmation']);self.assertEqual(provider.calls,0)
        self.assertEqual(nav.confirm(3,Point(114,30))['type'],'need_destination_choice')
        nav.confirm(1,Point(114,30));self.assertEqual(provider.calls,1)
        fix = Location(route.destination.point,70,2)
        event = nav.update(fix,70);self.assertEqual(event['arrival_wait']['target'],'17')
        self.assertNotIn('arrival_wait',nav.update(fix,70))
        nav.cancel();self.assertIsNone(nav.update(fix,70))


if __name__=='__main__':unittest.main()
