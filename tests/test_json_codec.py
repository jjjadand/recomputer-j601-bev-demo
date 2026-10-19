"""Frame transport must retain numeric values and reject GPU NaN/Inf results."""
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from json_codec import dumps, loads


class JsonCodecTests(unittest.TestCase):
    def test_transport_round_trip_retains_frame_values(self):
        frame = {'points': [[-1.25, 32.7, .04, 654321]], 'label': '车道线',
                 'jpeg': 'data:image/jpeg;base64,//9j+', 'ground_truth_used': False}
        encoded = dumps(frame)
        self.assertIsInstance(encoded, bytes)
        self.assertEqual(loads(encoded), frame)

    def test_nonfinite_gpu_values_reject_the_entire_frame(self):
        for value in [math.nan, math.inf, -math.inf]:
            with self.subTest(value=value), self.assertRaises((ValueError, OverflowError)):
                dumps({'detections': [{'center': [0, value, 1]}]})
