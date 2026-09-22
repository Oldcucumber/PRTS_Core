"""Optional desktop measurement, separate from the portable inference ABI."""
import json
from pathlib import Path
import threading
import time
import psutil


class ResourceMonitor:
    def __init__(self, output, dxgi_library=None):
        self.output=Path(output);self.rows=[];self.done=threading.Event()
        self.dxgi=None
        if dxgi_library:
            from .windows_memory import WindowsGPUMemory
            self.dxgi=WindowsGPUMemory(dxgi_library)
        self.process=psutil.Process();self.start=time.monotonic()
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def _run(self):
        with self.output.open('w',encoding='utf-8') as file:
            while not self.done.wait(.1):
                info=self.process.memory_info();resident=info.rss
                for child in self.process.children(recursive=True):
                    try:resident+=child.memory_info().rss
                    except psutil.NoSuchProcess:pass
                row=dict(elapsed_s=time.monotonic()-self.start,process_tree_rss=resident,
                         peak_working_set=getattr(info,'peak_wset',info.rss),
                         gpu_local_usage=0,gpu_nonlocal_usage=0)
                if self.dxgi:row.update(self.dxgi.sample())
                row['conservative_envelope']=resident+row['gpu_local_usage']+row['gpu_nonlocal_usage']
                self.rows.append(row);file.write(json.dumps(row)+'\n')

    def close(self):
        self.done.set();self.thread.join()
        peak=max((r['conservative_envelope'] for r in self.rows),default=0)
        return dict(samples=len(self.rows),peak_conservative_desktop_envelope_bytes=peak,
                    with_assumed_frontend_2gb=peak+2_000_000_000,
                    peak_process_tree_rss_bytes=max((r['process_tree_rss'] for r in self.rows),default=0),
                    dxgi_measured=self.dxgi is not None,
                    method='100 ms process-tree RSS plus simultaneous DXGI device-0 local/nonlocal allocations; possible shared-memory double counting. Frontend 2 GB is a reservation, not a measurement. Not an Apple footprint.')
