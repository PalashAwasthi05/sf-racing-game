"""Run an isolated macOS benchmark, accounting for its server, browser and GPU helpers."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

MIB = 1024 ** 2
LIB = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)


class Usage(ctypes.Structure):
    # rusage_info_v0, from sys/resource.h
    _fields_ = [('uuid', ctypes.c_uint8 * 16), ('fields', ctypes.c_uint64 * 10)]


def footprint(pid):
    usage = Usage()
    if LIB.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0:
        return usage.fields[7], usage.fields[8]  # footprint and process start time
    return None


def system_state():
    pressure, size = ctypes.c_int(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    if LIB.sysctlbyname(b'kern.memorystatus_vm_pressure_level', ctypes.byref(pressure), ctypes.byref(size), None, 0):
        raise RuntimeError('Memory pressure monitor unavailable')
    stats = subprocess.check_output(['/usr/bin/vm_stat'], text=True, timeout=2)
    page = re.search(r'page size of (\d+) bytes', stats)
    pages = dict((k, int(v)) for k, v in re.findall(r'^Pages (\w+):\s+(\d+)', stats, re.M))
    available = (pages['free'] + pages['inactive']) * int(page.group(1))
    return pressure.value, available


def members(root, known, measured=None):
    # Groups cover adopted workers; parent links cover Chrome's independent groups.
    lines = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,stat='], text=True, timeout=2)
    table = {int(p): (int(pp), int(group)) for p, pp, group, state in
             (line.split() for line in lines.splitlines()) if 'Z' not in state}
    owned = {p for p, (_, g) in table.items() if g == root}
    for p, birth in known.items():
        if p in table:
            value = footprint(p)
            if value and value[1] == birth:
                owned.add(p)
    while True:
        more = {p for p, (parent, _) in table.items() if parent in owned}
        if more <= owned:
            break
        owned |= more
    total, live = 0, {}
    for p in owned:
        value = footprint(p)
        if value is None:
            # Exit races are normal. If a live process is unreadable stop the run.
            try:
                os.kill(p, 0)
            except ProcessLookupError:
                continue
            if not footprint(p):
                status = subprocess.run(['/bin/ps', '-p', str(p), '-o', 'stat='], capture_output=True, text=True)
                if not status.stdout.strip() or 'Z' in status.stdout:
                    continue
                raise RuntimeError('Cannot read a live workload process')
            value = footprint(p)
            if not value:
                raise RuntimeError('Cannot measure workload')
        total += value[0]
        live[p] = value[1]
        if measured is not None:
            measured[p] = value[0]
    return total, live


def process_role(pid, executable, root):
    """Classify executable names internally; never persist names or arguments."""
    name = Path(executable.strip()).name.lower()
    if pid == root:
        return 'driver'
    if 'gpu' in name:
        return 'gpu'
    if 'renderer' in name:
        return 'renderer'
    if ('chrome' in name or 'chromium' in name) and 'helper' not in name:
        return 'browser'
    if name.startswith('python'):
        return 'server'
    return 'helper'


def process_breakdown(root, measured, seconds):
    # Only inspect the isolated workload. macOS can name its GPU process simply "Chrome Helper".
    names = subprocess.run(['/bin/ps', '-p', ','.join(str(p) for p in sorted(measured)),
                            '-o', 'pid=,comm='], capture_output=True, text=True, timeout=2).stdout if measured else ''
    roles = {}
    for line in names.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            roles[int(parts[0])] = process_role(int(parts[0]), parts[1], root)
    # The lightweight Chromium shell uses its own executable for all children.
    helpers = [str(pid) for pid, role in roles.items() if role in ('helper', 'browser')]
    if helpers:
        helper_types = subprocess.run(['/bin/ps', '-p', ','.join(helpers), '-o', 'pid=,args='],
                                      capture_output=True, text=True, timeout=2).stdout
        for line in helper_types.splitlines():
            match = re.match(r'^\s*(\d+)\s+.*?\s--type=(gpu-process|renderer)(?:\s|$)', line)
            if match:
                roles[int(match[1])] = 'gpu' if match[2] == 'gpu-process' else 'renderer'
    totals = {role: 0 for role in ('driver', 'browser', 'renderer', 'gpu', 'server', 'helper')}
    processes = []
    for pid, byte_count in sorted(measured.items()):
        role = roles.get(pid, 'helper')
        totals[role] += byte_count
        processes.append({'pid': pid, 'role': role, 'footprintMiB': round(byte_count/MIB, 2)})
    return {'secondsSinceStart': round(seconds, 2), 'rolesMiB': {r: round(v/MIB, 2) for r, v in totals.items()},
            'processes': processes}


def stop(proc, known):
    # The session belongs solely to this job. Recheck process birth before
    # signalling helpers that started a session of their own.
    try:
        _, live = members(proc.pid, known)
        known.update(live)
    except Exception:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid, birth in known.items():
        current = footprint(pid)
        if current and current[1] == birth:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    proc.wait(timeout=3)


def merge_drive_report(destination, memory):
    """Keep a single reviewable report even when the watchdog killed the browser."""
    try:
        drive = json.loads(destination.read_text())
    except (OSError, ValueError):
        drive = {'scope': 'drive-stream', 'partial': True}
    rows = memory['samples']
    origin = drive.get('timeOriginEpochMs')
    events = drive.get('events', [])

    def stats(values):
        values = sorted(values)
        return {'samples': len(values), 'medianMiB': values[len(values)//2] if values else None,
                'minMiB': min(values) if values else None, 'maxMiB': max(values) if values else None}

    def relative_ms(row):
        return memory['startedEpochMs'] + row[0] * 1000 - origin

    if origin is not None:
        crossings = drive.get('crossingChecks') or [dict(e) for e in events if e['type'] == 'crossing']
        for crossing in crossings:
            crossing['externalMemory'] = stats([r[1] for r in rows if abs(relative_ms(r)-crossing['ms']) <= 1000])
            role_rows = [r for r in memory.get('processRoleSamples', []) if
                         abs(memory['startedEpochMs']+1000*r['secondsSinceStart']-origin-crossing['ms']) <= 1000]
            crossing['externalMemoryByRole'] = {role: stats([r['rolesMiB'][role] for r in role_rows])
                for role in ('driver', 'browser', 'renderer', 'gpu', 'server', 'helper')}
        drive['crossingChecks'] = crossings
        evictions = [e for e in events if e['type'] == 'evicted']
        transients = [e for e in events if e['type'] in ('prepare', 'attached', 'crossing', 'evicted')]
        path_samples = drive.get('samples', [])
        settled = []
        for row in rows:
            ms = relative_ms(row)
            if not any(e['ms'] <= ms-2000 for e in evictions):
                continue
            if any(abs(e['ms']-ms) < 2000 for e in transients):
                continue
            near = min(path_samples, key=lambda s: abs(s['ms']-ms), default=None)
            if near is None or abs(near['ms']-ms) > 750 or near.get('loading') or not near.get('ready'):
                continue
            settled.append(row)
        count = len(settled)
        bands = [settled[:count//3], settled[count//3:2*count//3], settled[2*count//3:]]
        drive['settledMemory'] = {name: {**stats([r[1] for r in band]),
            'fromEpochMs': memory['startedEpochMs']+band[0][0]*1000 if band else None,
            'toEpochMs': memory['startedEpochMs']+band[-1][0]*1000 if band else None}
            for name, band in zip(('early', 'middle', 'late'), bands)}
        drive['settledMemory']['criterion'] = 'After first eviction; 2s away from prepare/attach/crossing/evict; nearest path sample ready and not loading'
        early, _, late = bands
        drive['settledMemory']['lateMinusEarlyMiB'] = (stats([r[1] for r in late])['medianMiB'] -
            stats([r[1] for r in early])['medianMiB']) if early and late else None
        turns = sorted(e['ms'] for e in drive.get('benchmarkEvents', []) if e['type'] == 'stopped-turnaround')
        groups = {}
        if turns:
            start_ms, end_ms = drive.get('startMs', 0), drive.get('endMs', float('inf'))
            residency_events = [e for e in events if e['type'] in ('attached', 'evicted')]
            ordered_path = sorted(path_samples, key=lambda s: s['ms'])
            path_index = 0
            for row in rows:
                ms = relative_ms(row)
                if ms < start_ms or ms > end_ms or (len(turns) > 1 and ms >= turns[1]):
                    continue
                while path_index+1 < len(ordered_path) and ordered_path[path_index+1]['ms'] <= ms:
                    path_index += 1
                near = ordered_path[path_index] if ordered_path else None
                if near is None or not 0 <= ms-near['ms'] <= 750 or not near.get('ready') or near.get('loading'):
                    continue
                if any(0 <= ms-e['ms'] <= 400 for e in residency_events):
                    continue
                active = tuple(sorted(near.get('active', [])))
                if not active:
                    continue
                leg = 'outbound' if ms < turns[0] else 'return'
                groups.setdefault(active, {'outbound': [], 'return': []})[leg].append(row[1])
        pairs, unmatched = [], []
        for active, legs in sorted(groups.items()):
            out, back = stats(legs['outbound']), stats(legs['return'])
            result = {'activePieces': list(active), 'outbound': out, 'return': back,
                      'deltaMiB': back['medianMiB']-out['medianMiB'] if out['samples'] and back['samples'] else None}
            (pairs if out['samples'] and back['samples'] else unmatched).append(result)
        drive['matchedVisitMemory'] = {'turnaroundMs': turns[0] if turns else None,
            'criterion': 'Same sorted active piece IDs on outbound and first return leg; ready and not loading; >400ms since attach/evict; preceding path sample within 750ms',
            'pairs': pairs, 'unmatched': unmatched}
    drive['externalMemory'] = memory
    if memory['reason'] != 'completed' or memory['exit'] != 0:
        drive['partial'] = True
    destination.write_text(json.dumps(drive))


def small_cpu_batch_allowed(command, cap_mib, reserve_mib):
    """A narrow exception for measured small CPU work, never the game/build."""
    if cap_mib > 384 or reserve_mib < 3584 or len(command) < 2:
        return False
    expected = Path(__file__).resolve().parents[1] / 'regen/sides.py'
    if Path(command[1]).resolve() != expected:
        return False
    try:
        pos = command.index('--max-buildings')
        return command[pos + 1] == '1'
    except (ValueError, IndexError):
        return False


def await_headroom(reserve_bytes, wait_seconds, maximum_pressure=1):
    deadline = time.monotonic() + wait_seconds
    green_since = None
    while True:
        level, available = system_state()
        now = time.monotonic()
        if not wait_seconds:
            return level, available, True
        if 1 <= level <= maximum_pressure and available >= reserve_bytes:
            if green_since is None:
                green_since = now
            if now - green_since >= 8:
                return level, available, True
        else:
            green_since = None
        if now >= deadline:
            return level, available, False
        time.sleep(min(2, deadline-now))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mib', type=int, required=True)
    cli.add_argument('--reserve-mib', type=int, default=3584)
    cli.add_argument('--timeout', type=int, default=90)
    cli.add_argument('--wait-for-green', type=int, default=0, metavar='SECONDS',
                     help='Before starting the workload, wait up to this long for eight continuous seconds of normal pressure and the usual reserve')
    cli.add_argument('--launch-headroom-mib', type=int, default=0,
                     help='Optionally require extra available memory at launch without changing the running cutoff')
    cli.add_argument('--report', type=Path, required=True)
    cli.add_argument('--merge-report', type=Path, help='Merge physical memory into an existing drive report, including after cutoff')
    cli.add_argument('--small-cpu-batch', action='store_true',
                     help='Only a single-building photo reader, <=384MiB: tolerate existing yellow pressure, never worsening pressure or a smaller system reserve')
    cli.add_argument('command', nargs=argparse.REMAINDER)
    args = cli.parse_args()
    if not args.command or args.mib <= 0 or args.reserve_mib <= 0 or args.timeout <= 0 or args.wait_for_green < 0 or args.launch_headroom_mib < 0:
        cli.error('Positive limits and a command are required')
    if args.small_cpu_batch and not small_cpu_batch_allowed(args.command, args.mib, args.reserve_mib):
        cli.error('Small CPU mode requires sides.py --max-buildings 1, <=384MiB cap and >=3584MiB reserve; never a browser or full build')
    monitor_ready = footprint(os.getpid()) is not None
    if args.wait_for_green and monitor_ready:
        print('Waiting for sustained safe headroom before launching the workload', flush=True)
    launch_min = max(args.reserve_mib, args.launch_headroom_mib)*MIB
    if args.small_cpu_batch:
        launch_min = max(launch_min, (args.reserve_mib + args.mib + 128)*MIB)
    maximum_pressure = 2 if args.small_cpu_batch else 1
    pressure, available, settled = await_headroom(launch_min, args.wait_for_green if monitor_ready else 0, maximum_pressure)
    started_epoch_ms = round(time.time() * 1000)
    preflight = {'pressureLevel': pressure, 'availableMiB': round(available/MIB), 'monitorReady': monitor_ready,
                 'policy': 'single-building CPU' if args.small_cpu_batch else 'normal-pressure'}
    if not settled or not monitor_ready or not 1 <= pressure <= maximum_pressure or available < launch_min:
        report = {'reason': 'preflight stopped', 'exit': 75, 'preflight': preflight, 'startedEpochMs': started_epoch_ms,
                  'limitMiB': args.mib, 'reserveMiB': args.reserve_mib, 'peakMiB': 0, 'samples': [], 'processRoleSamples': []}
        args.report.write_text(json.dumps(report))
        if args.merge_report:
            args.merge_report.write_text(json.dumps({'scope': 'drive-stream', 'partial': True, 'externalMemory': report}))
        print(f"Preflight stopped before launch: pressure {pressure}, available {available/MIB:.0f} MiB, "
              f"reserve {args.reserve_mib} MiB, monitor ready {monitor_ready}", flush=True)
        return 75
    proc = subprocess.Popen(args.command, start_new_session=True, env={**os.environ, "SF_MEMORY_GUARDED": "1",
                            "SF_GUARD_STARTED_EPOCH_MS": str(started_epoch_ms)})
    start, peak, known, samples, reason = time.monotonic(), 0, {}, [], 'completed'
    role_samples, last_role_time = [], -1.
    maximum_pressure = pressure  # once green, a return to yellow also stops this batch
    try:
        while proc.poll() is None:
            measured = {}
            total, known = members(proc.pid, known, measured)
            elapsed = time.monotonic() - start
            pressure, available = system_state()
            peak = max(peak, total)
            samples.append([round(elapsed, 2), round(total/MIB), round(available/MIB), len(known), pressure])
            if elapsed-last_role_time >= 1:
                role_samples.append(process_breakdown(proc.pid, measured, elapsed))
                last_role_time = elapsed
            if total >= args.mib*MIB:
                reason = 'workload footprint cutoff'
            elif not 1 <= pressure <= maximum_pressure or available < args.reserve_mib*MIB:
                reason = 'system headroom cutoff'
            elif elapsed >= args.timeout:
                reason = 'time cutoff'
            if reason != 'completed':
                break
            maximum_pressure = min(maximum_pressure, pressure)
            time.sleep(.15)
    except BaseException:
        reason = 'monitor interrupted'
        raise
    finally:
        stop(proc, known)
        report = {'reason': reason, 'exit': proc.returncode, 'startedEpochMs': started_epoch_ms, 'preflight': preflight,
                  'sampleColumns': ['secondsSinceStart', 'footprintMiB', 'availableMiB', 'processCount', 'pressureLevel'],
                  'limitMiB': args.mib, 'reserveMiB': args.reserve_mib,
                  'peakMiB': round(peak/MIB), 'samples': samples, 'processRoleSamples': role_samples,
                  'processRoleMethod': 'Owned process tree physical footprints; executable and Chromium process type roles; no executable names or command arguments retained'}
        args.report.write_text(json.dumps(report))
        if args.merge_report:
            merge_drive_report(args.merge_report, report)
        print(f'{reason}; peak workload {peak/MIB:.0f} MiB; limit {args.mib} MiB', flush=True)
    return proc.returncode if reason == 'completed' else 75


if __name__ == '__main__':
    sys.exit(main())
