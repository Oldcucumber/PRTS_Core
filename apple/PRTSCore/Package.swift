// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Run Foundation-only migration tests on a Mac without downloading model runtimes.
let contractsOnly=ProcessInfo.processInfo.environment["PRTS_CONTRACTS_ONLY"] == "1"

let package = Package(
    name: "PRTSCore",
    platforms: [.iOS(.v16),.macOS(.v13)],
    products: [.library(name: "PRTSContracts", targets: ["PRTSContracts"])] + (contractsOnly ? [] : [
        .library(name: "PRTSAppleModels", targets: ["PRTSAppleModels"])
    ]),
    dependencies: contractsOnly ? [] : [
        .package(url: "https://github.com/k2-fsa/sherpa-onnx", exact: "1.13.8"),
        .package(url: "https://github.com/csukuangfj/onnxruntime-libs", exact: "1.28.2")
    ],
    targets: [ .target(name: "PRTSContracts"),
        .testTarget(name: "PRTSContractsTests", dependencies: ["PRTSContracts"], resources:[.process("Resources")])
    ] + (contractsOnly ? [] : [
        .binaryTarget(name: "prts_vlm", path: "Artifacts/prts_vlm.xcframework"),
        .target(name: "PRTSAppleModels", dependencies: [
            "PRTSContracts", "prts_vlm",
            .product(name: "sherpa-onnx", package: "sherpa-onnx"),
            .product(name: "onnxruntime-ios", package: "onnxruntime-libs")
        ], resources:[.copy("Resources/cues")],
        linkerSettings: [.linkedFramework("CoreML"),.linkedFramework("CoreFoundation"),.linkedLibrary("c++")])
    ])
)
