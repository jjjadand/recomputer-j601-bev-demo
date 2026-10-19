"""Numerical checks run on the Jetson GPU, with CPU references for CUDA kernels."""
import pathlib
import sys
import unittest


class GpuPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest("CUDA GPU required")
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "jetson"))
        import gpu_images
        import gpu_fusion
        import gpu_surround
        import numpy as np
        cls.cp, cls.np = cp, np
        cls.images, cls.fusion = gpu_images, gpu_fusion
        cls.surround = gpu_surround

    def test_foreground_mode_and_robust_box_reduction(self):
        cp, np = self.cp, self.np
        ego = np.column_stack((np.arange(15),np.full(15,2),np.full(15,1.5))).astype(np.float32)
        depth = cp.asarray([10]*8+[30]*7,dtype=cp.float32)
        pixels = [(cp.full(15,5,dtype=cp.float32),cp.full(15,5,dtype=cp.float32),depth)]*6
        detection = {"box":[0,0,10,10],"score":.9,"class_id":2,"label":"car"}
        fused, semantics = self.fusion.associate(cp.asarray(ego),pixels,[[detection],[],[],[],[],[]])
        self.assertEqual(len(fused),1)
        self.assertEqual(fused[0]["points"],8)
        np.testing.assert_allclose(fused[0]["center"],np.median(ego[:8],axis=0),atol=.001)
        extent = np.clip(np.quantile(ego[:8],.9,axis=0)-np.quantile(ego[:8],.1,axis=0),[.4,.4,.5],[8,8,5])
        np.testing.assert_allclose(fused[0]["size"],extent,atol=.001)
        np.testing.assert_array_equal(cp.asnumpy(semantics),[2]*8+[-1]*7)

    def test_insufficient_lidar_support_is_rejected(self):
        cp = self.cp
        ego = cp.asarray([[i,0,1] for i in range(4)],dtype=cp.float32)
        pixels = [(cp.full(4,5,dtype=cp.float32),cp.full(4,5,dtype=cp.float32),cp.full(4,10,dtype=cp.float32))]*6
        detection = {"box":[0,0,10,10],"score":.9,"class_id":2,"label":"car"}
        fused,semantics = self.fusion.associate(ego,pixels,[[detection],[],[],[],[],[]])
        self.assertFalse(fused)
        self.np.testing.assert_array_equal(cp.asnumpy(semantics),[-1]*4)

    def test_letterbox_channels_and_padding_match_cpu_reference(self):
        import cv2
        cp,np = self.cp,self.np
        image = np.random.default_rng(123).integers(0,256,(90,160,3),dtype=np.uint8)
        output = cp.empty((3,640,640),dtype=cp.float32)
        height,width,ratio,top,left = self.images.prepare(cp.asarray(image),output)
        expected = np.full((640,640,3),114,dtype=np.uint8)
        expected[top:top+round(height*ratio),left:left+round(width*ratio)] = cv2.resize(image,(round(width*ratio),round(height*ratio)))
        expected = expected[:,:,::-1].transpose(2,0,1).astype(np.float32)/255
        np.testing.assert_allclose(cp.asnumpy(output),expected,atol=1.01/255)

    def test_jpeg_round_trip_preserves_shape_and_channel_order(self):
        import cv2,base64
        cp,np = self.cp,self.np
        original = np.empty((64,128,3),dtype=np.uint8);original[:]=(20,90,220)
        ok,encoded=cv2.imencode('.jpg',original)
        self.assertTrue(ok)
        codec=self.images.NvJpeg()
        try:
            images=codec.decode([encoded]*6)
            result=codec.encode(images[0])
            decoded=cv2.imdecode(np.frombuffer(base64.b64decode(result.split(',')[1]),np.uint8),cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape,original.shape)
            np.testing.assert_allclose(decoded.mean(axis=(0,1)),original.mean(axis=(0,1)),atol=3)
        finally:codec.close()

    def test_batched_jpeg_preserves_each_image_shape_color_and_order(self):
        import cv2, base64
        cp, np = self.cp, self.np
        originals = [np.full((48+i*8, 80+i*16, 3), color, dtype=np.uint8)
                     for i, color in enumerate([(20, 40, 210), (190, 70, 20), (30, 200, 60)])]
        codec = self.images.NvJpeg()
        try:
            for _ in range(2):  # Reuse encoder states across frames.
                results = codec.encode_many([cp.asarray(image) for image in originals], [82, 90, 85])
                self.assertEqual(len(results), len(originals))
                for original, result in zip(originals, results):
                    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(result.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(decoded.shape, original.shape)
                    np.testing.assert_allclose(decoded.mean(axis=(0, 1)), original.mean(axis=(0, 1)), atol=3)
        finally:
            codec.close()

    def test_progressive_jpeg_uses_compatible_decode_backend(self):
        import cv2
        cp, np = self.cp, self.np
        original = np.full((96, 128, 3), (20, 90, 220), dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
        self.assertTrue(ok)
        codec = self.images.NvJpeg()
        try:
            images = codec.decode([encoded]*6)
            self.assertEqual(len(images), 6)
            np.testing.assert_allclose(cp.asnumpy(images[0]).mean(axis=(0, 1)), original.mean(axis=(0, 1)), atol=3)
            self.assertEqual(codec.last_decode_backend, 'nvJPEG hybrid')
        finally:
            codec.close()

    def test_ground_projection_matches_cpu_and_compensates_lidar_pose(self):
        import cv2
        cp, np = self.cp, self.np
        yy, xx = np.indices((128, 128))
        image = np.stack((xx, yy, np.full_like(xx, 44)), axis=-1).astype(np.uint8)
        ego_to_image = np.array([[0, -4, 0, 63.5], [-4, 0, 0, 63.5],
                                 [0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
        l2ego = np.array([[0, -1, 0, 1.2], [1, 0, 0, -.8],
                          [0, 0, 1, 1.84], [0, 0, 0, 1]], dtype=np.float32)
        projections = np.stack([ego_to_image @ l2ego] * 6)
        actual = cp.asnumpy(self.surround.camera_bev([cp.asarray(image)]*6, projections, l2ego, 64, 10))
        rows, cols = np.indices((64, 64), dtype=np.float32)
        x, y = 10-(rows+.5)*20/64, 10-(cols+.5)*20/64
        expected = cv2.remap(image, 63.5-4*y, 63.5-4*x, cv2.INTER_LINEAR)
        outside_ego = (np.abs(x) >= 2.3) | (np.abs(y) >= .95)
        np.testing.assert_allclose(actual[outside_ego], expected[outside_ego], atol=1)
        # Behind all six cameras must remain unobserved.
        projections[:, 2, :] = -projections[:, 2, :]
        blind = cp.asnumpy(self.surround.camera_bev([cp.asarray(image)]*6, projections, l2ego, 64, 10))
        np.testing.assert_array_equal(blind[outside_ego], np.tile([22, 19, 13], (outside_ego.sum(), 1)))


if __name__ == '__main__':
    unittest.main()
