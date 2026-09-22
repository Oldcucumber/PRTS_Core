// Measurement-only helper. DXGI reports this process's local/nonlocal usage,
// including DirectML allocations; no NVIDIA/CUDA telemetry library is required.
#include <windows.h>
#include <dxgi1_4.h>
#include <stdint.h>
#include <cstdio>
#include <vector>

struct adapters {
    std::vector<IDXGIAdapter3 *> items;
    std::vector<LUID> identifiers;
    HRESULT result = E_FAIL;
    adapters() {
        IDXGIFactory1 * factory = nullptr;
        result = CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&factory));
        if (FAILED(result)) return;
        for (UINT i = 0;; ++i) {
            IDXGIAdapter1 * base = nullptr;
            if (factory->EnumAdapters1(i, &base) == DXGI_ERROR_NOT_FOUND) break;
            if (!base) break;
            DXGI_ADAPTER_DESC1 description = {};
            base->GetDesc1(&description);
            char name[512] = {};
            WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, name, sizeof(name), nullptr, nullptr);
            std::fprintf(stderr, "DXGI adapter: %s; flags=%u; LUID=%08lx:%08lx\n", name, description.Flags,
                         static_cast<unsigned long>(description.AdapterLuid.HighPart), description.AdapterLuid.LowPart);
            if (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { base->Release(); continue; }
            bool duplicate = false;
            for (const auto & id : identifiers)
                duplicate |= id.HighPart == description.AdapterLuid.HighPart && id.LowPart == description.AdapterLuid.LowPart;
            if (duplicate) { base->Release(); continue; }
            IDXGIAdapter3 * adapter = nullptr;
            if (SUCCEEDED(base->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void **>(&adapter)))) {
                items.push_back(adapter);
                identifiers.push_back(description.AdapterLuid);
            }
            base->Release();
        }
        factory->Release();
        result = items.empty() ? E_NOINTERFACE : S_OK;
    }
    ~adapters() { for (auto p : items) p->Release(); }
};

extern "C" __declspec(dllexport) int32_t prts_gpu_memory_for_adapter(uint32_t index, uint64_t * local, uint64_t * nonlocal) {
    // Keep process-lifetime COM references: this diagnostic DLL can be unloaded
    // during CRT teardown, after DXGI has already begun shutting down.
    static adapters & devices = *new adapters;
    *local = *nonlocal = 0;
    if (FAILED(devices.result)) return devices.result;
    if (index >= devices.items.size()) return E_INVALIDARG;
    {
        auto p = devices.items[index];
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        HRESULT code = p->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
        if (FAILED(code)) return code;
        *local += info.CurrentUsage;
        code = p->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info);
        if (FAILED(code)) return code;
        *nonlocal += info.CurrentUsage;
    }
    return S_OK;
}

extern "C" __declspec(dllexport) int32_t prts_gpu_memory(uint64_t * local, uint64_t * nonlocal) {
    // ORT DirectML defaults to DXGI device_id=0. Measure that adapter only.
    return prts_gpu_memory_for_adapter(0, local, nonlocal);
}
