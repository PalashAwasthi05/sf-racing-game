"""Run a command with a memory ceiling: read its physical footprint (the kernel's phys_footprint, which counts MPS /
unified memory and compressed pages, via proc_pid_rusage) every second, log it every two minutes, and kill the
command if it passes the ceiling instead of letting it swap. The exit line gives the lifetime peak footprint.
Usage: work/geoenv/bin/python work/watch_mem.py <limit_GB> <log> <command...>   (readers: 16 GB; see NEXT_AGENT.md)"""
import ctypes, os, subprocess, sys, time

LIBC = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
RUSAGE_INFO_V4 = 4


class RUsageV4(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16), ('v', ctypes.c_uint64 * 35)]


def footprint(pid):
    """(current, lifetime peak) physical footprint in bytes, or (0, 0) once the process is gone."""
    r = RUsageV4()
    if LIBC.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(RUSAGE_INFO_V4), ctypes.byref(r)) != 0: return 0., 0.
    return float(r.v[7]), float(r.v[28])                  # ri_phys_footprint, ri_lifetime_max_phys_footprint


def main():
    limit, log = float(sys.argv[1]) * 1e9, open(sys.argv[2], 'a')
    p = subprocess.Popen(sys.argv[3:], stdout=log, stderr=subprocess.STDOUT)
    peak, t0, last = 0., time.time(), 0.
    while p.poll() is None:
        now, life = footprint(p.pid); peak = max(peak, life, now)
        if now > limit:
            p.kill(); print(f'watchdog: killed at {now / 1e9:.1f} GB after {(time.time() - t0) / 60:.1f} min', file=log, flush=True); break
        if time.time() - last > float(os.environ.get('WATCH_LOG_S', 120)): print(f'watchdog: {now / 1e9:.1f} GB (peak {peak / 1e9:.1f}) at {(time.time() - t0) / 60:.1f} min', file=log, flush=True); last = time.time()
        time.sleep(1)
    code = p.wait()
    print(f'watchdog: exit {code}, peak {peak / 1e9:.1f} GB, {(time.time() - t0) / 60:.1f} min', file=log, flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
