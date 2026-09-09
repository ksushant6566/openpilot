import os
import numpy as np

from metadrive.component.sensors.rgb_camera import RGBCamera
from panda3d.core import Texture, GraphicsOutput, loadPrcFileData


def configure_software_rendering():
  from metadrive.engine.core import engine_core
  from metadrive.third_party.simplepbr import init
  # The main view is unused; llvmpipe cannot allocate MetaDrive's 16-sample HDR buffer.
  engine_core.init = lambda **kwargs: init(**{**kwargs, 'msaa_samples': 0})
  loadPrcFileData('', 'framebuffer-multisample 0\nmultisamples 0')


class CopyRamRGBCamera(RGBCamera):
  """Camera which copies its content into RAM during the render process, for faster image grabbing."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cpu_texture = Texture()
    self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)

  def _setup_effect(self):
    if os.getenv('LIBGL_ALWAYS_SOFTWARE') != '1':
      return super()._setup_effect()
    # Render directly into RGB8 on CPU, keeping the road/lane texture shader.
    from metadrive.constants import CameraTagStateKey, Semantics
    from metadrive.engine.core.terrain import Terrain
    cam = self.get_cam().node()
    cam.setTagStateKey(CameraTagStateKey.RGB)
    cam.setTagState(Semantics.TERRAIN.label, Terrain.make_render_state(self.engine, 'terrain.vert.glsl', 'terrain.frag.glsl'))

  def get_rgb_array_cpu(self):
    origin_img = self.cpu_texture
    img = np.frombuffer(origin_img.getRamImageAs("RGB").getData(), dtype=np.uint8)
    img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 3))
    img = img[::-1]  # Flip on vertical axis
    return img


class RGBCameraWide(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    lens = self.get_lens()
    lens.setFov(120)
    lens.setNear(0.1)


class RGBCameraRoad(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    lens = self.get_lens()
    lens.setFov(40)
    lens.setNear(0.1)
