import os
import queue
import signal
import shutil
import subprocess
import time
import unittest

from multiprocessing import Queue
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware.hw import Paths
from openpilot.tools.sim.bridge.common import QueueMessageType

SIM_DIR = os.path.join(BASEDIR, "openpilot/tools/sim")

class TestSimBridgeBase(OpenpilotTestCase):
  @classmethod
  def setup_class(cls):
    if cls is TestSimBridgeBase:
      raise unittest.SkipTest("Don't run this base class, run test_metadrive_bridge.py instead")

  def setup_method(self):
    self.processes = []

  def test_driving(self):
    # Startup manager and bridge.py. Check processes are running, then engage and verify.
    p_manager = subprocess.Popen("./launch_openpilot.sh", cwd=SIM_DIR, start_new_session=True)
    self.processes.append(p_manager)

    camera_services = ['modelV2', 'narrowRoadCameraState']
    sm = messaging.SubMaster(['selfdriveState', 'onroadEvents', 'managerState', *camera_services])
    q = Queue()
    bridge = self.create_bridge()
    p_bridge = bridge.run(q, retries=10)
    self.processes.append(p_bridge)

    max_time_per_step = 120

    # Wait for bridge to startup
    start_waiting = time.monotonic()
    while p_bridge.is_alive() and not bridge.started.value and time.monotonic() < start_waiting + max_time_per_step:
      time.sleep(0.1)
    assert p_bridge.exitcode is None, f"Bridge process should be running, but exited with code {p_bridge.exitcode}"
    assert bridge.started.value, "Bridge did not start before the deadline"

    start_time = time.monotonic()
    no_car_events_issues_once = False
    car_event_issues = []
    not_running = []
    while time.monotonic() < start_time + max_time_per_step:
      sm.update(100)

      not_running = [p.name for p in sm['managerState'].processes if not p.running and p.shouldBeRunning]
      car_event_issues = [event.name for event in sm['onroadEvents'] if any([event.noEntry, event.softDisable, event.immediateDisable])]

      if sm.all_alive() and len(car_event_issues) == 0 and len(not_running) == 0:
        no_car_events_issues_once = True
        break

    assert no_car_events_issues_once, \
                    f"Failed because no messages received, or CarEvents '{car_event_issues}' or processes not running '{not_running}'"

    start_time = time.monotonic()
    min_counts_control_active = 100
    control_active = 0

    while time.monotonic() < start_time + max_time_per_step:
      sm.update(100)

      if sm.updated['selfdriveState'] and sm['selfdriveState'].active:
        control_active += 1

        if control_active == min_counts_control_active:
          break

    assert min_counts_control_active == control_active, f"Simulator did not engage a minimal of {min_counts_control_active} steps was {control_active}"

    start_driving = time.monotonic()
    observed_frames = {s: set() for s in camera_services}
    deadline = start_driving + self.test_duration + 15
    while bridge.started.value and time.monotonic() < deadline:
      sm.update(100)
      for s in ['selfdriveState', *camera_services]:
        assert time.monotonic() - sm.logMonoTime[s] / 1e9 < 1, f"{s} stopped publishing while driving"
      for s in camera_services:
        if sm.updated[s]:
          assert time.monotonic() - sm[s].timestampEof / 1e9 < 1, f"{s} is using stale camera frames"
          observed_frames[s].add(sm[s].frameId)
      if sm.updated['selfdriveState']:
        assert sm['selfdriveState'].active, "openpilot disengaged while driving"
    assert not bridge.started.value, "Simulation failed to terminate before the deadline"
    observed_seconds = time.monotonic() - start_driving
    for s, frames in observed_frames.items():
      assert len(frames) >= 20 * (observed_seconds - 1), f"{s}: {len(frames)} unique frames in {observed_seconds:.2f}s"

    done_info = None
    while True:
      try:
        state = q.get(timeout=1)
      except queue.Empty:
        break
      if state.type == QueueMessageType.TERMINATION_INFO:
        done_info = state.info
        break
    assert done_info is not None, "Simulator exited without reporting its result"
    assert done_info.get("timeout"), f"Simulator ended before the driving duration elapsed: {done_info}"
    simulated, elapsed = done_info.pop('simulated_seconds'), done_info.pop('elapsed_seconds')
    assert abs(simulated - elapsed) < 1, f"Simulation is not realtime: {simulated:.2f}s simulated in {elapsed:.2f}s"
    print(f"Driving timing: {simulated=:.2f}s, {elapsed=:.2f}s, {observed_seconds=:.2f}s, frames={ {s: len(f) for s, f in observed_frames.items()} }")
    failure_states = [name for name, failed in done_info.items() if name != "timeout" and failed]
    assert len(failure_states) == 0, f"Simulator fails to finish a loop. Failure states: {failure_states}"

  def teardown_method(self):
    print("Test shutting down. CommIssues are acceptable")
    for p in reversed(self.processes):
      p.terminate()

    for p in reversed(self.processes):
      if isinstance(p, subprocess.Popen):
        try:
          p.wait(timeout=10)
        except subprocess.TimeoutExpired:
          os.killpg(p.pid, signal.SIGKILL)
          p.wait()
      else:
        p.join(timeout=10)
        if p.is_alive():
          p.kill()
          p.join()

    if destination := os.getenv("SIM_ARTIFACTS_DIR"):
      root = Path(Paths.log_root())
      for name in ("rlog.zst", "qlog.zst", "qcamera.ts"):
        for src in root.rglob(name):
          target = Path(destination) / src.relative_to(root)
          target.parent.mkdir(parents=True, exist_ok=True)
          shutil.copyfile(src, target)
