import Foundation
import XCTest
@testable import PRTSContracts

final class PythonParityTests: XCTestCase {
    struct IdentifierCase: Decodable { let input:String;let expected:[String] }
    struct IntentCase: Decodable { let input:String;let expected:TaskIntent? }
    struct Observation: Decodable { let texts:[TextEvidence];let vehicles:[VehicleObservation];let now:Double;let expected:TaskNotice? }
    struct WaitingCase: Decodable { let command:TaskIntent;let observations:[Observation] }
    struct Fix: Decodable { let input:LocationFix;let now:Double;let expected:RouteUpdate }
    struct QueueObservation: Decodable { let target:String;let expected:TaskNotice? }
    struct QueueLayout: Decodable {
        let id:String;let input:[TextEvidence];let roles:[String];let observations:[QueueObservation]
    }
    struct Fixture: Decodable {
        let identifiers:[IdentifierCase];let intents:[IntentCase];let waiting:[WaitingCase]
        let route:WalkingRoute;let route_fixes:[Fix]
        let queue_audio:[IdentifierCase];let queue_layouts:[QueueLayout]
    }
    func fixture() throws -> Fixture {
        let url=try XCTUnwrap(Bundle.module.url(forResource:"python-parity",withExtension:"json"))
        return try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
    }
    func testCompleteIdentifiersAndCommandParity() throws {
        let fixtures=try fixture()
        for row in fixtures.identifiers { XCTAssertEqual(TextRules.identifiers(row.input),row.expected,row.input) }
        for row in fixtures.intents {
            let actual=CommandRouter.parse(row.input)
            XCTAssertEqual(actual?.action,row.expected?.action,row.input)
            XCTAssertEqual(actual?.kind ?? "",row.expected?.kind ?? "",row.input)
            XCTAssertEqual(actual?.target ?? "",row.expected?.target ?? "",row.input)
            XCTAssertEqual(actual?.direction ?? "",row.expected?.direction ?? "",row.input)
            XCTAssertEqual(actual?.query,row.expected?.query,row.input)
            XCTAssertEqual(actual?.arrival_wait,row.expected?.arrival_wait,row.input)
        }
    }
    func testVehicleAssociationAndPartialEvidenceParity() throws {
        for scenario in try fixture().waiting {
            let state=TaskState();_=state.command(scenario.command,now:0)
            for row in scenario.observations {
                let actual=state.observe(texts:row.texts,vehicles:row.vehicles,now:row.now)
                XCTAssertEqual(actual?.type,row.expected?.type)
                XCTAssertEqual(actual?.text,row.expected?.text)
                XCTAssertEqual(actual?.source,row.expected?.source)
                XCTAssertEqual(actual?.vehicle_id,row.expected?.vehicle_id)
                XCTAssertEqual(actual?.wait?.notified,row.expected?.wait?.notified)
            }
        }
    }
    func testRecordedRouteWith627SyntheticFixes() throws {
        let fixtures=try fixture(),tracker=try RouteTracker(route:fixtures.route)
        XCTAssertEqual(fixtures.route_fixes.count,627)
        for row in fixtures.route_fixes {
            let actual=tracker.update(row.input,now:row.now),expected=row.expected
            XCTAssertEqual(actual.status,expected.status)
            XCTAssertEqual(actual.step_index,expected.step_index)
            XCTAssertEqual(actual.action,expected.action)
            XCTAssertEqual(try XCTUnwrap(actual.progress_m),try XCTUnwrap(expected.progress_m),accuracy:0.0001)
            XCTAssertEqual(try XCTUnwrap(actual.relative_bearing_deg),try XCTUnwrap(expected.relative_bearing_deg),accuracy:0.0001)
        }
    }
    func testQueueFieldAndAnnouncementParity() throws {
        let fixtures=try fixture()
        for row in fixtures.queue_audio { XCTAssertEqual(QueueTextRules.calledInAudio(row.input),row.expected,row.input) }
        for layout in fixtures.queue_layouts {
            let classified=QueueTextRules.classify(layout.input)
            XCTAssertEqual(classified.map { $0.queue_role ?? "" },layout.roles,layout.id)
            for row in layout.observations {
                let state=TaskState()
                _=state.command(TaskIntent(action:"wait",kind:"number",target:row.target),now:0)
                let actual=state.observe(texts:classified,vehicles:[],now:1)
                XCTAssertEqual(actual?.type,row.expected?.type,layout.id+"/"+row.target)
                XCTAssertEqual(actual?.text,row.expected?.text,layout.id+"/"+row.target)
            }
        }
    }
    func testCancellationReplacementAndQueueAudio() {
        let state=TaskState()
        _=state.command(TaskIntent(action:"wait",kind:"number",target:"B205"),now:2)
        XCTAssertNil(state.observe(texts:[],vehicles:[],now:3,audio:"请 B205 号到窗口",audioStart:1))
        XCTAssertNil(state.observe(texts:[],vehicles:[],now:4,audio:"B205 号还未叫到，请等待",audioStart:3))
        _=state.command(TaskIntent(action:"wait",kind:"number",target:"A108"),now:5)
        XCTAssertNil(state.observe(texts:[],vehicles:[],now:6,audio:"请 B205 号到窗口",audioStart:5.5))
        XCTAssertEqual(state.observe(texts:[],vehicles:[],now:7,audio:"请 A108 号到窗口",audioStart:6)?.type,"target_observed")
        _=state.command(TaskIntent(action:"cancel"),now:8)
        XCTAssertNil(state.observe(texts:[],vehicles:[],now:9,audio:"请 A108 号到窗口",audioStart:8))
    }
    func testSplitSpeechTailPreservesWaitingIdentity() throws {
        let state=TaskState()
        _=state.command(TaskIntent(action:"wait",kind:"number",target:"215"),now:1)
        let version=state.version
        let intent=try XCTUnwrap(CommandRouter.parse("到时提醒我",waiting:state.wait))
        XCTAssertEqual(intent.action,"continue_wait")
        XCTAssertEqual(state.command(intent,now:2).type,"wait_continued")
        XCTAssertEqual(state.version,version)
        XCTAssertEqual(state.wait?.started,1)
        XCTAssertEqual(state.observe(texts:[],vehicles:[],now:3,audio:"请215号取餐",audioStart:2)?.type,"target_observed")
        _=state.command(intent,now:4)
        XCTAssertEqual(state.wait?.notified,true)
        XCTAssertNil(state.observe(texts:[],vehicles:[],now:5,audio:"请215号取餐",audioStart:4))
        _=state.command(TaskIntent(action:"cancel"),now:6)
        XCTAssertEqual(state.command(intent,now:7).type,"need_target")
        XCTAssertEqual(QueueTextRules.calledInAudio("请739-8到窗口办理"),[])
    }
}
