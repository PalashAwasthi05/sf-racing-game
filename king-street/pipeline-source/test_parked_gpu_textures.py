import unittest
import numpy as np
from PIL import Image
from parked_gpu_textures import bounded_mips, quality
from startup_gpu_textures import mip_images, quality as reference_quality


class ParkedGpuTexturesTest(unittest.TestCase):
    def test_bounded_filter_matches_existing_linear_light_mips(self):
        image=Image.fromarray(np.random.default_rng(12).integers(0,256,(64,32,4),dtype=np.uint8))
        for srgb in [False,True]:
            expected=list(mip_images(image,srgb=srgb))
            actual=list(bounded_mips(image,srgb))
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):
                np.testing.assert_array_equal(np.asarray(a),np.asarray(b))

    def test_histogram_gate_matches_reference_used_channel_thresholds(self):
        rng=np.random.default_rng(9)
        for size in [4,16,32,64]:
            a=rng.integers(0,256,(size,size,4),dtype=np.uint8)
            for limit in [0,1,3,6,18,40]:
                noise=rng.integers(-limit,limit+1,a.shape)
                b=np.clip(a.astype(np.int16)+noise,0,255).astype(np.uint8)
                b[...,3]=a[...,3]
                for roles in [{'baseColorTexture'},{'metallicRoughnessTexture','occlusionTexture'}]:
                    result=quality(Image.fromarray(a),Image.fromarray(b),roles)
                    expected=reference_quality(Image.fromarray(a),Image.fromarray(b),roles)
                    self.assertEqual(result is None,expected is None)
                    if result:
                        self.assertAlmostEqual(result['rmsByte'],expected['rmsByte'],places=4)
                        self.assertEqual(result['p99Byte'],expected['p99Byte'])
                        self.assertEqual(result['maxByte'],expected['maxByte'])

if __name__=='__main__':unittest.main()
