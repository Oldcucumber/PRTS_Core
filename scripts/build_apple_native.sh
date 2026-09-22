#!/bin/bash
# Run on the team's Mac with Xcode and CMake. No signing or deployment occurs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REV=b29c606e28a01b1bc8c1351026a0fa6e616bf6c4
VENDOR="$ROOT/outputs/vendor/llama-b10964-apple"
BUILD="$ROOT/outputs/apple-native"
ARTIFACT="$ROOT/apple/PRTSCore/Artifacts/prts_vlm.xcframework"
test "$(uname -s)" = Darwin || { echo "This build needs macOS and Xcode."; exit 1; }
xcodebuild -version
PRTS_CONTRACTS_ONLY=1 swift test --package-path "$ROOT/apple/PRTSCore"
mkdir -p "$BUILD" "$(dirname "$VENDOR")" "$(dirname "$ARTIFACT")"
if [ ! -d "$VENDOR/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$VENDOR"
fi
git -C "$VENDOR" checkout --detach "$REV"
test "$(git -C "$VENDOR" rev-parse HEAD)" = "$REV"
for SDK in iphoneos iphonesimulator; do
    cmake -S "$ROOT/native" -B "$BUILD/$SDK" -G Xcode \
        -DLLAMA_SOURCE_DIR="$VENDOR" -DPRTS_OFFICIAL_RUNTIME=ON -DPRTS_CPU_REPACK=OFF -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT="$SDK" -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD/$SDK" --config Release --target prts_vlm --parallel 4
done
# Refuse to overwrite an earlier artifact; keep it available for comparison.
test ! -e "$ARTIFACT" || { echo "Artifact already exists: $ARTIFACT"; exit 1; }
xcodebuild -create-xcframework \
    -framework "$BUILD/iphoneos/Release-iphoneos/prts_vlm.framework" \
    -framework "$BUILD/iphonesimulator/Release-iphonesimulator/prts_vlm.framework" \
    -output "$ARTIFACT"
cd "$ROOT/apple/PRTSCore"
xcodebuild -scheme PRTSAppleModels -destination 'generic/platform=iOS Simulator' \
    -derivedDataPath "$BUILD/SwiftDerivedData" \
    build CODE_SIGNING_ALLOWED=NO
