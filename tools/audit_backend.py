"""Verify deployed fusion and accumulation against sensor ablations and CPU geometry."""
import argparse
import collections
import hashlib
import io
import json
import pathlib
import time

import numpy as np
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]


def occupied_cells(points):
    """Independent metric-cell reference; do not import backend code."""
    valid = ((np.abs(points[:, :2]) < 50).all(axis=1)
             & (points[:, 2] > -.6) & (points[:, 2] < 6))
    cells = np.floor((50-points[valid, :2])*5.12).astype(np.int64)
    return len(np.unique(cells, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://192.168.55.18:9000')
    parser.add_argument('--output', default='logs/backend-audit-20261014.json')
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    controls = {}
    report = {'passed': False, 'gateway': args.gateway, 'checks': {}, 'frames': []}
    manifest = json.loads((ROOT/'data/nuscenes-long/manifest.json').read_text())
    output = ROOT/args.output

    def packet(index):
        return (ROOT/'data/nuscenes-long'/manifest['frames'][index]['packet']).read_bytes()

    def process(body, index, reset):
        response = session.post(args.gateway+'/frame', data=body,
                                headers={'X-Frame-Index': str(index), 'X-Reset': str(int(reset))},
                                timeout=60)
        response.raise_for_status()
        return response.json()

    def digest(result, key):
        return hashlib.sha256(result['images'][key].encode()).hexdigest()

    def record(result):
        return {key: result[key] for key in ('index', 'timestamp', 'points_count',
                                            'vision_detections', 'fusion_bev', 'mapping', 'timings')} | {
            'image_sha256': {key: digest(result, key) for key in ('lidar', 'vision', 'fusion', 'map')}}

    try:
        for ui in ('http://127.0.0.1:8088', 'http://127.0.0.1:18088'):
            state = session.get(ui+'/api/status', timeout=5).json()['state']
            controls[ui] = {key: state[key] for key in ('playing', 'fps', 'index')}
            response = session.post(ui+'/api/control', json={'playing': False}, timeout=5)
            response.raise_for_status()
        time.sleep(.5)  # Finish any request already in flight before the audit.
        baseline = process(packet(0), 0, True)
        repeated = process(packet(0), 0, True)
        assert baseline['fusion_bev']['object_count'] > 0
        assert baseline['fusion_bev']['associated_lidar_points'] > 0
        assert all(digest(baseline, k) == digest(repeated, k) for k in ('lidar', 'vision', 'fusion', 'map'))
        assert digest(baseline, 'fusion') != digest(baseline, 'lidar')
        report['checks']['repeat_input_reproduces_output'] = True
        report['checks']['fusion_has_lidar_associated_objects'] = True

        with np.load(io.BytesIO(packet(0)), allow_pickle=False) as source:
            empty = {key: source[key] for key in source.files}
        empty['points'] = np.empty((0, 5), dtype=np.float32)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **empty)
        ablation = process(buffer.getvalue(), 0, True)
        assert ablation['vision_detections'] == baseline['vision_detections']
        assert ablation['fusion_bev']['object_count'] == 0
        assert ablation['fusion_bev']['associated_lidar_points'] == 0
        assert ablation['fusion_bev']['occupied_cells'] == ablation['mapping']['occupied_cells'] == 0
        assert len(ablation['pointcloud']) == 0
        assert digest(ablation, 'fusion') == digest(ablation, 'lidar')
        assert digest(ablation, 'map') != digest(baseline, 'map')
        report['checks']['remove_lidar_removes_occupancy_and_spatial_objects'] = True
        report['baseline'] = record(baseline)
        report['no_lidar'] = record(ablation)

        aligned_history = collections.deque(maxlen=25)
        origin = None
        for index in range(27):
            body = packet(index)
            result = process(body, index, index == 0)
            with np.load(io.BytesIO(body), allow_pickle=False) as source:
                xyz = source['points'][:, :3].astype(np.float64)
                homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
                ego = homogeneous @ source['lidar2ego'].astype(np.float64).T
                if origin is None:
                    origin = source['lidar2world'] @ np.linalg.inv(source['lidar2ego'])
                world_to_origin = np.linalg.inv(origin) @ source['lidar2world']
                step = max(1, len(xyz)//6000)
                aligned_history.append((homogeneous[::step] @ world_to_origin.T)[:, :3])
            expected_current = occupied_cells(ego[:, :3])
            expected_map = occupied_cells(np.concatenate(list(aligned_history)))
            actual_current = result['fusion_bev']['occupied_cells']
            actual_map = result['mapping']['occupied_cells']
            # Float32 GPU and independent float64 geometry may disagree at cell edges.
            assert abs(actual_current-expected_current) <= max(4, expected_current*.002)
            assert abs(actual_map-expected_map) <= max(4, expected_map*.002)
            assert result['mapping']['frames'] == min(index+1, 25)
            assert result['mapping']['retained_points'] == sum(len(points) for points in aligned_history)
            assert result['mapping']['newest_timestamp'] == result['timestamp']
            assert result['mapping']['oldest_timestamp'] == manifest['frames'][max(0, index-24)]['timestamp']
            row = record(result)
            row['cpu_reference_occupied_cells'] = {'current': expected_current, 'map': expected_map}
            report['frames'].append(row)
        report['checks']['occupancy_matches_independent_cpu_geometry'] = True
        report['checks']['rolling_history_caps_at_25_frames'] = True
        accumulated = result
        reset = process(packet(26), 26, True)
        assert reset['mapping']['frames'] == 1 and reset['mapping']['sensor_span_s'] == 0
        assert reset['mapping']['occupied_cells'] < accumulated['mapping']['occupied_cells']
        assert digest(reset, 'map') != digest(accumulated, 'map')
        report['checks']['reset_changes_map_from_25_frames_to_one'] = True
        report['reset'] = record(reset)
        report['passed'] = True
        print(json.dumps({'passed': True, 'checks': report['checks'],
                          'baseline_objects': baseline['fusion_bev']['object_count'],
                          'no_lidar_objects': ablation['fusion_bev']['object_count'],
                          'accumulated_cells': accumulated['mapping']['occupied_cells'],
                          'reset_cells': reset['mapping']['occupied_cells']}, indent=2))
    finally:
        output.write_text(json.dumps(report, indent=2)+'\n')
        for ui, state in controls.items():
            response = session.post(ui+'/api/control', json={
                'playing': state['playing'], 'fps': state['fps'], 'seek': state['index']}, timeout=8)
            response.raise_for_status()


if __name__ == '__main__':
    main()
