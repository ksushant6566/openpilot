import unittest

import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Context

from openpilot.selfdrive.modeld.compile_modeld import make_input_queues, make_policy_inputs


class TestPolicyInputs(unittest.TestCase):
  def test_history_across_jit_capture_and_replay(self):
    shapes = {'img': (1, 12, 2, 4), 'big_img': (1, 12, 2, 4), 'features_buffer': (1, 2, 3),
              'desire_pulse': (1, 3, 8), 'traffic_convention': (1, 2), 'action_t': (1, 2)}
    dtypes = {k: 'uint8' if 'img' in k else 'float16' for k in shapes}
    with Context(DEV='CPU'):
      queues, _, _ = make_input_queues(shapes, frame_skip=2, device='CPU', frame_copy_size=0)
      prepare = make_policy_inputs(shapes, 2, dtypes)

      @TinyJit
      def run(*args):
        inputs = prepare(*args)
        Tensor.realize(*inputs.values())
        return inputs

      histories = {k: q.numpy().copy() for k, q in queues.items() if k != 'packed_npy_inputs'}
      rng = np.random.default_rng(12)
      for _ in range(15):
        warped = rng.integers(0, 256, (2, 6, 2, 4), dtype=np.uint8)
        desire = rng.integers(0, 2, 8).astype(np.float32)
        features = rng.standard_normal(3).astype(np.float32)
        traffic, action = np.array([1, 0], np.float32), np.array([.2, .3], np.float32)
        packed = np.concatenate([desire, traffic, action, features])
        actual = run(Tensor(warped), *(queues[k] for k in histories), Tensor(packed))
        for key, newest in [('img_q', warped[:1]), ('big_img_q', warped[1:]),
                            ('feat_q', features.reshape(1, 1, 3)), ('desire_q', desire.reshape(1, 1, 8))]:
          histories[key] = np.concatenate([histories[key][1:], newest])
        expected = {'img': histories['img_q'][::2].reshape(shapes['img']),
                    'big_img': histories['big_img_q'][::2].reshape(shapes['big_img']),
                    'features_buffer': histories['feat_q'][::2].reshape(shapes['features_buffer']),
                    'desire_pulse': histories['desire_q'].reshape(3, 2, 8).max(axis=1)[None],
                    'traffic_convention': traffic[None], 'action_t': action[None]}
        for name, value in actual.items():
          np.testing.assert_array_equal(value.numpy(), expected[name].astype(dtypes[name]), err_msg=name)
