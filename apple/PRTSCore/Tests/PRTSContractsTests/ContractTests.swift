import XCTest
@testable import PRTSContracts

final class ContractTests: XCTestCase {
    func testCameraKeepsLatestAndRejectsEarlierSequence() {
        let slot=LatestFrameSlot()
        func frame(_ i: UInt64) -> RGBFrame {
            RGBFrame(timestamp:Double(i),sequence:i,width:1,height:1,bytes:Data([0,0,0]))
        }
        XCTAssertTrue(slot.offer(frame(4))); XCTAssertTrue(slot.offer(frame(7)))
        XCTAssertFalse(slot.offer(frame(6)))
        XCTAssertEqual(slot.takeLatest()?.sequence,7); XCTAssertNil(slot.takeLatest())
    }
    func testPCMSilenceEndsUtteranceAndMissingChunkResetsIt() {
        let audio=PCMBuffer(); var result: Utterance?
        for i in 0..<10 {
            let r=audio.append(PCMChunk(timestamp:Double(i)*0.1,sequence:UInt64(i),
                samples:Array(repeating:i<4 ? 0.08 : 0,count:1600),role:.user))
            if let r { result=r }
        }
        XCTAssertEqual(result?.start,0)
        XCTAssertEqual(result?.role,.user)
        let reset=PCMBuffer()
        _=reset.append(PCMChunk(timestamp:0,sequence:0,samples:Array(repeating:0.08,count:3200)))
        let next=reset.append(PCMChunk(timestamp:1,sequence:9,samples:Array(repeating:0.08,count:3200),endUtterance:true))
        XCTAssertEqual(next?.start,1)
    }
}
