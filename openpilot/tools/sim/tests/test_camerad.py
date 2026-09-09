import unittest
from unittest.mock import patch

import numpy as np

from openpilot.tools.sim.lib import camerad
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


class TestCamerad(unittest.TestCase):
  def test_padded_frames_and_replay(self):
    with patch.object(camerad, 'W', 14), patch.object(camerad, 'H', 6), \
         patch.object(camerad.messaging, 'PubMaster'), patch.object(camerad, 'VisionIpcServer'):
      camera = camerad.Camerad(True)
      stride, y_height, uv_height, size = get_nv12_info(14, 6)
      self.assertEqual(camera.vipc_server.create_buffers_with_sizes.call_count, 2)
      self.assertEqual(camera.vipc_server.create_buffers_with_sizes.call_args.args[2:], (14, 6, size, stride, stride * y_height))
      # BT.601 integer values from the original simulator kernel.
      for rgb, values in [((0, 0, 0), (16, 128, 128)), ((255, 255, 255), (237, 128, 128)),
                          ((255, 0, 0), (82, 109, 184)), ((0, 255, 0), (145, 91, 81)), ((0, 0, 255), (42, 184, 119))]:
        frame = np.full((6, 14, 3), rgb, dtype=np.uint8)
        expected = np.zeros(size, dtype=np.uint8)
        expected[:stride * y_height].reshape(y_height, stride)[:6, :14] = values[0]
        expected[stride * y_height:stride * (y_height + uv_height)].reshape(uv_height, stride)[:3, :14] = np.tile(values[1:], 7)
        self.assertEqual(camera.rgb_to_yuv(frame), expected.tobytes())
      with patch.object(camerad.time, 'monotonic_ns', return_value=123456789):
        camera.cam_send_yuv_road(expected.tobytes())
      self.assertEqual(camera.vipc_server.send.call_args.args[2:], (0, 123456789, 123456789))
      state = camera.pm.send.call_args.args[1].narrowRoadCameraState
      self.assertEqual((state.frameId, state.timestampSof, state.timestampEof), (0, 123456789, 123456789))
