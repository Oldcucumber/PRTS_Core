"""Deterministic logic regressions. These are not perception or ASR accuracy claims."""
import unittest
import numpy as np
from prts_core.intent import route, ordered_ocr
from prts_core.tasks import Tasks, identifiers
from prts_core.guidance import Guide, search_corridor


class IntentTests(unittest.TestCase):
    def test_controls_and_questions(self):
        for text, action in [
            ('别再给我方向提示了','stop'),('暂停方向指引','stop'),('不用继续报方向','stop'),
            ('不要停止导航，继续看路','navigate'),('请开始导航，沿人行道走','navigate'),
            ('我可以沿着人行道往前走吗？','ask'),('帮我看看前面的路能不能走？','ask'),
            ('看看门口的牌子','ask'),('取消导航','stop'),('取消等车提醒','cancel')]:
            with self.subTest(text=text):self.assertEqual(route(text)['action'],action)

    def test_wait_entities_and_replacement(self):
        for text, kind, target in [
            ('帮我等十七路公交，到了提醒','bus','17'),('等三号线进站','train','3'),
            ('我的号是B二百零六，叫到我时告诉我','number','B206')]:
            with self.subTest(text=text):self.assertEqual((route(text)['kind'],route(text)['target']),(kind,target))
        self.assertEqual(route('改成B207号',{'kind':'number'})['target'],'B207')
        self.assertEqual(route('帮我等公交车')['target'],'')

    def test_layout_preserves_subject_before_prohibition(self):
        def item(t,x,y,w,h):return dict(text=t,score=.99,box=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        rows=ordered_ocr([item('禁止入内',120,60,40,150),item('施工车辆',40,65,40,150),item('注意',40,10,120,25)])
        self.assertEqual(rows,['注意','施工车辆 / 禁止入内'])


class WaitingTests(unittest.TestCase):
    def setUp(self):
        self.tasks=Tasks();self.tasks.command(dict(action='wait',kind='number',target='B206'),10)

    def test_numbers(self):
        self.assertEqual(identifiers('B 二百零六、二十一、A一零八'),['B206','21','A108'])

    def test_wrong_stale_negative_and_deduplicated_audio(self):
        self.assertIsNone(self.tasks.observe([],[],12,'请B260号到1号窗口',11))
        self.assertIsNone(self.tasks.observe([],[],12,'请B206号到1号窗口',9))
        self.assertIsNone(self.tasks.observe([],[],12,'B206号还未叫到，请等待',11))
        self.assertEqual(self.tasks.observe([],[],12,'请B206号到1号窗口',11)['source'],'audio')
        self.assertIsNone(self.tasks.observe([],[],13,'请B206号到1号窗口',12))

    def test_replacement_and_cancel(self):
        self.tasks.command(dict(action='wait',kind='number',target='B207'),15)
        self.assertIsNone(self.tasks.observe([],[],16,'请B206号到1号窗口',15))
        self.tasks.command(dict(action='cancel'),17)
        self.assertIsNone(self.tasks.observe([],[],18,'请B207号到1号窗口',17))

    def test_unrelated_route_sign_is_not_bus(self):
        self.tasks.command(dict(action='wait',kind='bus',target='17'),10)
        sign=dict(text='17路',score=.99)
        self.assertIsNone(self.tasks.observe([sign],[dict(label='bus')],12))
        self.assertIsNone(self.tasks.observe([],[],12,'17路即将到站',11))
        sign['vehicle_label']='bus'
        self.assertEqual(self.tasks.observe([sign],[dict(label='bus')],12)['source'],'visual')


class PlannerTests(unittest.TestCase):
    def perception(self, ground, boxes=()):
        return dict(sidewalk_class=ground,sidewalk=ground.astype('float32'),
                    detections=[dict(box=b,label='person',class_id=0,score=.9,blocking=True) for b in boxes])

    def test_real_causality_in_planner_with_constructed_masks(self):
        # Constructed geometry only: tests path search, not whether a model sees obstacles.
        frame=np.zeros((180,320,3),np.uint8);ground=np.ones((180,320),bool)
        straight=Guide()(frame,self.perception(ground),0)
        detour=Guide()(frame,self.perception(ground,[[.44,.50,.58,.76]]),0)
        stop=Guide()(frame,self.perception(ground,[[.0,.55,1,.70]]),0)
        self.assertEqual(straight['direction'],'FORWARD')
        self.assertEqual(detour['status'],'DETOUR');self.assertGreater(detour['detector_path_shift'],.035)
        self.assertEqual(detour['path'][0],detour['baseline_path'][0])
        self.assertEqual(stop['status'],'STOP');self.assertGreaterEqual(stop['baseline_coverage'],.55)

    def test_no_stale_boxes_and_no_false_obstacle_cause(self):
        frame=np.zeros((180,320,3),np.uint8);ground=np.zeros((180,320),bool)
        ground[:,100:220]=True;ground[120:140,:]=False
        guide=Guide();r=guide(frame,self.perception(ground,[[.0,.50,.10,.99]]),0)
        self.assertEqual(r['status'],'FREE')
        r=guide(frame,self.perception(np.ones_like(ground)),.5)
        self.assertEqual(r['detections'],[]);self.assertEqual(r['status'],'CANDIDATE')

    def test_no_bridge_over_gap(self):
        ground=np.ones((180,320),bool);ground[140:142]=False
        _,coverage=search_corridor(ground);self.assertLess(coverage,.55)

    def test_no_teleport_to_remote_side_corridor(self):
        ground=np.zeros((180,320),bool);ground[:,20:85]=True
        path,coverage=search_corridor(ground)
        self.assertEqual(path,[]);self.assertEqual(coverage,0)


if __name__=='__main__':unittest.main()
