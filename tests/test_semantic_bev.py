"""Metric mask projection checks with analytically located road/lane/occlusion cells."""
import pathlib
import sys
import unittest


class SemanticBevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest('CUDA GPU required')
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))
        import numpy as np
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'jetson'))
        import gpu_semantic_bev
        cls.cp, cls.np, cls.module = cp, np, gpu_semantic_bev

    def fixture(self):
        cp, np = self.cp, self.np
        # An independent orthographic camera: u=64-4y, v=64-4x, depth=10.
        p = np.array([[0, -40, 0, 640], [-40, 0, 0, 640],
                      [0, 0, 0, 10], [0, 0, 0, 1]], dtype=np.float32)
        l2ego = np.array([[0, -1, 0, 1.2], [1, 0, 0, -.8],
                         [0, 0, 1, 1.84], [0, 0, 0, 1]], dtype=np.float32)
        invisible = p.copy(); invisible[2, 3] = -10
        projections = np.stack([p@l2ego]+[invisible@l2ego]*5)
        road = np.zeros((6, 2, 128, 128), dtype=np.float32)
        road[:, 0] = 1
        road[:, 0, :, 40:88] = 0; road[:, 1, :, 40:88] = 1
        lane = np.zeros_like(road); lane[:, 0] = 1
        lane[:, 0, :, 44:47] = 0; lane[:, 1, :, 44:47] = 1
        return cp.asarray(road), cp.asarray(lane), [[128, 128, 1, 0, 0]]*6, projections, l2ego

    def render(self, boxes=None, objects=None, max_range=10):
        return self.module.semantic_bev(*self.fixture(), boxes or [[] for _ in range(6)],
                                        objects or [], {2: (66, 236, 176)},
                                        size=80, extent=10, max_range=max_range)

    def test_metric_road_lane_labels_and_lidar_pose_compensation(self):
        cp, np = self.cp, self.np
        image, labels = self.render()
        rows, cols = np.indices((80, 80))
        x, y = 10-(rows+.5)*.25, 10-(cols+.5)*.25
        u = np.floor(64-4*y).astype(int)
        expected = np.zeros((80, 80), np.uint8)
        valid = x*x+y*y <= 100
        expected[valid & (u >= 40) & (u < 88)] = 1
        expected[valid & (u >= 44) & (u < 47)] = 2
        expected[(np.abs(x) < 2.3) & (np.abs(y) < .95)] = 4
        np.testing.assert_array_equal(cp.asnumpy(labels), expected)
        for label in (0, 1, 2):
            np.testing.assert_array_equal(cp.asnumpy(image)[expected == label],
                                          np.tile(self.module.PALETTE[label], ((expected == label).sum(), 1)))

    def test_occluded_rays_are_unknown_and_objects_are_filled(self):
        cp, np = self.cp, self.np
        box = {'class_id': 2, 'box': [58, 24, 70, 40]}
        object_ = {'class_id': 2, 'center': [8, 0, 1], 'size': [2, 2, 2]}
        image, labels = self.render([[box], [], [], [], [], []], [object_])
        labels = cp.asnumpy(labels)
        # (x=6.125, y=.125) is inside the 2D occlusion but outside the footprint.
        self.assertEqual(labels[15, 39], 0)
        # (x=8.125, y=.125) is a filled object footprint, not an outline/texture.
        self.assertEqual(labels[7, 39], 3)
        np.testing.assert_array_equal(cp.asnumpy(image)[7, 39], [176, 236, 66])

    def test_far_range_and_bad_shapes_do_not_fabricate_lanes(self):
        cp = self.cp
        _, labels = self.render(max_range=5)
        self.assertEqual(int(labels[15, 39]), 0)
        args = list(self.fixture()); args[1] = args[1][:5]
        with self.assertRaises(ValueError):
            self.module.semantic_bev(*args, [[]]*6, [], {})

    def test_an_unoccluded_overlap_camera_can_supply_road(self):
        cp, np = self.cp, self.np
        args = list(self.fixture())
        args[3][1] = args[3][0]
        box = {'class_id': 2, 'box': [58, 24, 70, 40]}
        _, labels = self.module.semantic_bev(*args, [[box], [], [], [], [], []], [], {},
                                             size=80, extent=10, max_range=10)
        self.assertEqual(int(labels[15, 39]), 1)

    def test_history_follows_pose_and_does_not_accumulate_objects(self):
        cp, np = self.cp, self.np
        memory = self.module.GroundMemory(hole_width_m=0)
        original = cp.zeros((80, 80), dtype=cp.uint8)
        original[20:30, 30:40] = 1
        original[21:29, 35] = 2
        original[40:45, 50:55] = 3
        unknown = cp.zeros_like(original)
        memory.update(original, unknown, np.eye(4), 0., 10., 10.)
        # A one-metre forward movement shifts static ground down four cells.
        pose = np.eye(4); pose[0, 3] = 1
        actual = memory.update(unknown, unknown, pose, .1, 10., 10.)
        expected = np.zeros((80, 80), dtype=np.uint8)
        expected[24:34, 30:40] = cp.asnumpy(original)[20:30, 30:40]
        np.testing.assert_array_equal(cp.asnumpy(actual), expected)
        # Turn left 90 degrees at the same translated origin. Static ground
        # must rotate right on screen, rather than retaining its old pixels.
        pose[:2, :2] = [[0, -1], [1, 0]]
        actual = memory.update(unknown, unknown, pose, .2, 10., 10.)
        expected[:] = 0
        expected[30:40, 46:56] = np.rot90(cp.asnumpy(original)[20:30, 30:40], k=3)
        np.testing.assert_array_equal(cp.asnumpy(actual), expected)

    def test_inherited_observations_expire_and_negative_evidence_wins(self):
        cp, np = self.cp, self.np
        memory = self.module.GroundMemory(max_age_s=.35, lane_age_s=.15, hole_width_m=0)
        original = cp.zeros((80, 80), dtype=cp.uint8); original[20:30, 30:40] = 1; original[25, 35] = 2
        unknown = cp.zeros_like(original)
        memory.update(original, unknown, np.eye(4), 1., 10., 10.)
        visibility = unknown.copy(); visibility[20, 30] = 2
        actual = memory.update(unknown, visibility, np.eye(4), 1.1, 10., 10.)
        self.assertEqual(int(actual[20, 30]), 0)
        self.assertEqual(int(actual[25, 35]), 2)
        actual = memory.update(unknown, unknown, np.eye(4), 1.2, 10., 10.)
        self.assertEqual(int(actual[25, 35]), 1)  # Old lane color expires, observed ground remains.
        self.assertGreater(int(cp.count_nonzero(actual)), 0)
        actual = memory.update(unknown, unknown, np.eye(4), 1.4, 10., 10.)
        self.assertEqual(int(cp.count_nonzero(actual)), 0)

    def test_history_resets_on_seek_and_limits_small_gap_interpolation(self):
        cp, np = self.cp, self.np
        memory = self.module.GroundMemory(hole_width_m=.4)
        original = cp.zeros((100, 100), dtype=cp.uint8); original[20:40, 20:40] = 1
        original[28, 28] = 0; original[28, 32] = 0; original[33, 28] = 0
        visibility = cp.ones_like(original); visibility[28, 32] = 0; visibility[33, 28] = 2
        result = memory.update(original, visibility, np.eye(4), 1., 5., 5.)
        self.assertEqual(int(result[28, 28]), 1)
        self.assertEqual(int(result[28, 32]), 0)
        self.assertEqual(int(result[33, 28]), 0)  # Strong non-road evidence stays empty.
        self.assertEqual(int(result[50, 50]), 0)
        unknown = cp.zeros_like(original)
        result = memory.update(unknown, unknown, np.eye(4), .5, 5., 5.)
        self.assertEqual(int(cp.count_nonzero(result)), 0)


if __name__ == '__main__':
    unittest.main()
