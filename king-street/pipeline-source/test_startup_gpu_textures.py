import io
import struct
import unittest
from PIL import Image
from startup_gpu_textures import compress_image, mip_images, quality


def png(image):
    data = io.BytesIO(); image.save(data, format='PNG'); return data.getvalue()


class StartupGpuTexturesTest(unittest.TestCase):
    def test_full_dxt1_chain_preserves_dimensions_and_is_eight_times_smaller(self):
        data, stats = compress_image(png(Image.new('RGBA', (64, 32), (132, 130, 132, 255))), {'baseColorTexture'})
        self.assertEqual(data[84:88], b'DXT1')
        self.assertEqual(struct.unpack_from('<I', data, 28)[0], 7)
        self.assertEqual(stats['mips'], 7)
        self.assertLess(stats['compressedMipBytes'], stats['rgbaMipBytes'] / 7)
        decoded = Image.open(io.BytesIO(data))
        self.assertEqual(decoded.size, (64, 32))
        self.assertLessEqual(stats['quality'][0]['rmsByte'], 3)

    def test_alpha_uses_dxt5_and_remains_within_one_byte(self):
        result = compress_image(png(Image.new('RGBA', (32, 32), (132, 130, 132, 128))), {'baseColorTexture'})
        self.assertIsNotNone(result)
        data, stats = result
        self.assertEqual(data[84:88], b'DXT5')
        self.assertEqual(Image.open(io.BytesIO(data)).convert('RGBA').getpixel((0, 0))[3], 128)

    def test_sensitive_normals_and_non_power_of_two_maps_keep_portable_image(self):
        self.assertIsNone(compress_image(png(Image.new('RGBA', (32, 32), (128, 128, 255, 255))), {'normalTexture'}))
        self.assertIsNone(compress_image(png(Image.new('RGBA', (30, 32), (132, 130, 132, 255))), {'baseColorTexture'}))

    def test_srgb_mips_average_in_linear_light(self):
        image = Image.new('RGBA', (2, 2), 'black')
        image.putpixel((1, 0), (255, 255, 255, 255)); image.putpixel((0, 1), (255, 255, 255, 255))
        levels = list(mip_images(image, srgb=True))
        self.assertEqual(levels[-1].getpixel((0, 0)), (188, 188, 188, 255))
        self.assertIsNone(quality(Image.new('RGBA', (4, 4), (0, 0, 0, 255)), Image.new('RGBA', (4, 4), (0, 0, 0, 250)), {'baseColorTexture'}))


if __name__ == '__main__': unittest.main()
