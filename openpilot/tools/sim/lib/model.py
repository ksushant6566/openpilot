"""CPU policy inference for the simulator; camera transforms and history use modeld's tinygrad code."""
import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, numpy_helper
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Context

from openpilot.selfdrive.modeld.compile_modeld import (NV12Frame, make_warp, make_policy_inputs, make_run_model,
                                                     nv12_copy_size, read_file_chunked_to_disk)
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
from openpilot.selfdrive.modeld.helpers import MODELS_DIR
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


def use_float32(model):
  """Expand fp16 values exactly; the CPU execution provider can then fuse these operators."""
  def convert_tensor(tensor):
    if tensor.data_type == TensorProto.FLOAT16:
      tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name))

  for tensor in model.graph.initializer:
    convert_tensor(tensor)
  for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
    if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
      value.type.tensor_type.elem_type = TensorProto.FLOAT
  for node in model.graph.node:
    for attr in node.attribute:
      if attr.type == onnx.AttributeProto.TENSOR:
        convert_tensor(attr.t)
      if node.op_type == 'Cast' and attr.name == 'to' and attr.i == TensorProto.FLOAT16:
        attr.i = TensorProto.FLOAT
  onnx.checker.check_model(model)
  return model


def load_sim_model(cam_w, cam_h):
  path = read_file_chunked_to_disk(MODELS_DIR / 'driving_supercombo.onnx')
  metadata = make_metadata_dict(path)
  options = onnxruntime.SessionOptions()
  options.intra_op_num_threads = 2
  options.inter_op_num_threads = 1
  options.add_session_config_entry('session.intra_op.allow_spinning', '0')
  session = onnxruntime.InferenceSession(use_float32(onnx.load(path)).SerializeToString(), sess_options=options,
                                        providers=['CPUExecutionProvider'])
  dtypes = {'tensor(uint8)': 'uint8', 'tensor(float)': 'float32'}
  input_dtypes = {inp.name: dtypes[inp.type] for inp in session.get_inputs()}
  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
  prepare_inputs = make_policy_inputs(metadata['input_shapes'], frame_skip, input_dtypes)

  def prepare_policy(*args):
    inputs = prepare_inputs(*args)
    Tensor.realize(*inputs.values())
    return inputs

  model_w, model_h = metadata['input_shapes']['img'][-1] * 2, metadata['input_shapes']['img'][-2] * 2
  prepare = TinyJit(make_run_model(make_warp(nv12, model_w, model_h), prepare_policy, metadata,
                                  nv12_copy_size(nv12.stride, nv12.y_height, nv12.uv_height)), prune=True)

  def run_model(**inputs):
    with Context(DEV='CPU'):
      policy_inputs = prepare(**inputs)
    output, = session.run(None, {name: value.numpy() for name, value in policy_inputs.items()})
    return Tensor(output, device='CPU'),

  return {'metadata': metadata, 'input_devices': {'model': 'CPU'}, 'run_model': {(cam_w, cam_h): run_model}}
