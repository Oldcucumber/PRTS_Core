"""Geometric regressions; constructed masks are not perception accuracy truth."""
import unittest
import cv2
import numpy as np
from prts_core.guidance import Guide,search_corridor,validate_path,path_heading
from prts_core.semantics import group_labels,region_records


class CorridorTests(unittest.TestCase):
    def observation(self,mask):return {'sidewalk_class':mask,'sidewalk':mask.astype(np.float32),'detections':[]}
    def corridor(self,sign):
        mask=np.zeros((180,320),np.uint8)
        pts=np.array([[160,179],[160,155],[160+sign*90,100],[160+sign*90,50]],np.int32)
        cv2.polylines(mask,[pts],False,1,64)
        return mask.astype(bool)

    def test_full_polyline_cannot_cross_hole_between_valid_vertices(self):
        free=np.ones((90,160),bool);free[45,80]=False
        self.assertFalse(validate_path([(80,70),(80,20)],free)['valid'])
        path,coverage=search_corridor(free)
        self.assertGreater(coverage,.55);self.assertTrue(validate_path(path,free)['valid'])

    def test_left_to_right_turn_is_confirmed_without_frozen_arrow(self):
        frame=np.zeros((180,320,3),np.uint8);g=Guide()
        left=g(frame,self.observation(self.corridor(-1)),0)
        transition=g(frame,self.observation(self.corridor(1)),.3)
        right=g(frame,self.observation(self.corridor(1)),.6)
        self.assertEqual(left['direction'],'LEFT')
        self.assertEqual(transition['direction'],'UNKNOWN')
        self.assertEqual(transition['direction_raw'],'RIGHT')
        self.assertEqual(right['direction'],'RIGHT')
        self.assertTrue(right['path_validation']['valid'])

    def test_invalid_current_path_stops_immediately(self):
        frame=np.zeros((180,320,3),np.uint8);g=Guide();mask=self.corridor(-1)
        g(frame,self.observation(mask),0);mask[150:155]=False
        r=g(frame,self.observation(mask),.3)
        self.assertEqual(r['direction'],'UNKNOWN');self.assertEqual(r['path'],[])

    def test_slow_cpu_interval_clears_history_but_uses_fresh_geometry(self):
        frame=np.zeros((180,320,3),np.uint8);g=Guide()
        g(frame,self.observation(self.corridor(-1)),0)
        r=g(frame,self.observation(self.corridor(1)),6)
        self.assertEqual(r['direction'],'RIGHT');self.assertTrue(r['path_validation']['valid'])
        stale=g(frame,self.observation(self.corridor(-1)),5)
        self.assertEqual(stale['direction'],'UNKNOWN')

    def test_route_without_camera_heading_cannot_move_candidate(self):
        frame=np.zeros((180,320,3),np.uint8);p=self.observation(np.ones((180,320),bool))
        a=Guide()(frame,p,0);b=Guide()(frame,p,0,route_hint={'status':'following','desired_heading_deg':270,'relative_bearing_deg':None})
        self.assertEqual(a['path'],b['path']);self.assertEqual(b['route_hint_status'],'camera_heading_unavailable')

    def test_sharp_bend_stays_connected(self):
        mask=np.zeros((180,320),np.uint8)
        cv2.polylines(mask,[np.array([[160,179],[160,143],[270,125],[270,50]],np.int32)],False,1,36)
        path,coverage=search_corridor(mask.astype(bool))
        self.assertGreaterEqual(coverage,.55);self.assertTrue(validate_path(path,mask)['valid'])
        self.assertEqual(path[0][0],160)
        r=Guide()(np.zeros((180,320,3),np.uint8),self.observation(mask.astype(bool)),0)
        self.assertEqual(r['detector_path_shift'],0)
        self.assertEqual(r['status'],'CANDIDATE')

    def test_heading_uses_fixed_rows_not_path_length(self):
        points=[(round(160+(169-y)*.5),y) for y in range(169,69,-1)]
        truncated=[p for p in points if p[1]>=100]
        a=path_heading(points,(180,320));b=path_heading(truncated,(180,320))
        self.assertAlmostEqual(a['heading_deg'],b['heading_deg'])


class RegionTests(unittest.TestCase):
    def test_raw_taxonomies_and_unknowns(self):
        labels={0:'flat-sidewalk',1:'nature-vegetation',2:'human-person',3:'flat-road',4:'nature-terrain',5:'train'}
        np.testing.assert_array_equal(group_labels(np.arange(6,dtype=np.uint8),labels),[1,3,4,2,0,4])

    def test_region_holes_preserved_and_vegetation_not_relabelled(self):
        classes=np.ones((60,80),np.uint8);classes[20:40,30:50]=0
        r=region_records(classes,{0:'sidewalk',1:'vegetation'})
        vegetation=next(x for x in r if x['label']=='vegetation')
        self.assertEqual(vegetation['group'],'fixed_obstacle');self.assertTrue(vegetation['holes'])

if __name__=='__main__':unittest.main()
