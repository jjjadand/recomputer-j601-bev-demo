"""Start the existing Ethernet/GPU backend and a persistent Jetson-local web UI."""
import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = pathlib.Path('/home/seeed/jetson-bev-demo')
UI_SERVICE = 'jetson-bev-local-ui.service'
GATEWAY = 'http://127.0.0.1:9000'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def command(*args, check=True):
    return subprocess.run(args, check=check)


def request(url, data=None, headers=None, timeout=5):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with OPENER.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors='replace')
        try:
            detail = json.loads(detail).get('error', detail)
        except ValueError:
            pass
        raise RuntimeError(f'{url}: HTTP {error.code}: {detail}') from error


def control(ui, values):
    return request(ui+'/api/control', json.dumps(values).encode(),
                   {'Content-Type': 'application/json'}, timeout=10)


def wait_for(url, predicate, seconds=30):
    deadline = time.monotonic()+seconds
    error = 'not ready'
    while time.monotonic() < deadline:
        try:
            result = request(url, timeout=3)
            if predicate(result):
                return result
            error = result.get('state', {}).get('error') or result.get('error') or 'not ready'
        except (OSError, ValueError, RuntimeError) as exc:
            error = str(exc)
        time.sleep(.5)
    raise RuntimeError(f'{url}: {error}')


def preflight(dataset, engine):
    if platform.machine() != 'aarch64' or not pathlib.Path('/etc/nv_tegra_release').is_file():
        raise RuntimeError('Run this script on the Jetson, in the seeed desktop session.')
    if ROOT != TARGET or os.getuid() == 0:
        raise RuntimeError(f'Extract the project to {TARGET} and run without sudo as seeed.')
    python = ROOT/'.venv/bin/python'
    if not python.is_file():
        raise RuntimeError('The Jetson .venv is missing. Install requirements-jetson.txt as described in README.md.')
    command(str(python), '-c',
            'import aiohttp,cupy,tensorrt,cv2,psutil,numpy; '
            'assert cupy.cuda.runtime.getDeviceCount()>0, "CUDA device unavailable"')
    for file in ('server.py', 'frontend/index.html', 'node_modules/three/build/three.module.js',
                 'node_modules/three/build/three.core.js', 'deploy/network.env'):
        if not (ROOT/file).is_file():
            raise RuntimeError(f'Missing {file}; extract the complete Jetson-local bundle.')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if not manifest.get('frames'):
        raise RuntimeError('The dataset has no frames.')
    for frame in manifest['frames']:
        packet = (dataset/frame['packet']).resolve()
        if not packet.is_relative_to(dataset) or not packet.is_file():
            raise RuntimeError(f'Missing or invalid sensor packet: {frame["packet"]}')
    if engine != 'auto' and not (ROOT/'models'/engine).is_file():
        raise RuntimeError(f'Missing models/{engine}; TensorRT engines must be built on this Jetson.')
    if engine == 'auto' and not any((ROOT/'models'/name).is_file() for name in (
        'yolo26n-thor-b6.engine', 'yolo26n-thor.engine')):
        raise RuntimeError('No TensorRT engine found in models; see the target build command in README.md.')
    if not (ROOT/'models/yolop-road-thor-b6.engine').is_file():
        raise RuntimeError('Road/lane engine missing. Build models/yolop-road-b6.onnx on the Jetson as described in README.md.')
    if not shutil.which('systemctl'):
        raise RuntimeError('systemd is required.')
    command('systemctl', '--user', 'show', '--property=SystemState', '--value')
    print(f'Dataset ready: {len(manifest["frames"])} paired frames.', flush=True)
    return python, manifest


def configure_backend(engine):
    command('sudo', '-v')
    # Start an inactive link once. Never restart the active physical link.
    for service in ('jetson-bev-net.service', 'jetson-bev-worker.service', 'jetson-bev-gateway.service'):
        source = ROOT/'deploy'/service
        destination = pathlib.Path('/etc/systemd/system')/service
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            command('sudo', 'install', '-m', '644', str(source), str(destination))
    override_dir = pathlib.Path('/etc/systemd/system/jetson-bev-worker.service.d')
    override = override_dir/'90-local-engine.conf'
    if engine == 'auto':
        if override.exists():
            command('sudo', 'rm', str(override))
    else:
        command('sudo', 'mkdir', '-p', str(override_dir))
        with tempfile.TemporaryDirectory(prefix='bev-local-') as temporary:
            file = pathlib.Path(temporary)/'engine.conf'
            file.write_text(f'[Service]\nEnvironment=BEV_VISION_ENGINE={engine}\n')
            command('sudo', 'install', '-m', '644', str(file), str(override))
    command('sudo', 'systemctl', 'daemon-reload')
    command('sudo', 'systemctl', 'start', 'jetson-bev-net.service')
    command('sudo', 'systemctl', 'restart', 'jetson-bev-worker.service', 'jetson-bev-gateway.service')
    wait_for(GATEWAY+'/status', lambda value: value.get('connected') is True)


def validate_first_frame(dataset, manifest):
    frame = manifest['frames'][0]
    packet = (dataset/frame['packet']).read_bytes()
    if hashlib.sha256(packet).hexdigest() != frame['sha256']:
        raise RuntimeError('The first sensor packet failed its SHA256 check.')
    print('Warming up the GPU and verifying a real six-camera/LiDAR frame...', flush=True)
    result = request(GATEWAY+'/frame', packet,
                     {'X-Frame-Index': str(frame['index']), 'X-Reset': '1'}, timeout=90)
    if (result.get('index') != frame['index'] or result.get('timestamp') != frame['timestamp']
            or result.get('points_count') != frame['points'] or len(result.get('cameras', [])) != 6
            or not {'lidar', 'vision', 'fusion', 'map', 'surround'} <= set(result.get('images', {}))
            or not result.get('surround', {}).get('semantic_lane_predictions')
            or not result.get('gpu_stages') or result.get('ground_truth_used') is not False):
        raise RuntimeError('The worker did not return the expected GPU pipeline result.')
    if not all(math.isfinite(value) and value >= 0 for value in result['timings'].values()):
        raise RuntimeError('Invalid inference timings.')
    print(f'GPU ready: {result["vision_model"]}, six cameras, '
          f'{result["vision_resolution"]}x{result["vision_resolution"]}; '
          f'pipeline {result["timings"]["total_ms"]:.1f} ms (warmup frame).', flush=True)


def configure_ui(python, dataset, port):
    quoted = lambda value: json.dumps(str(value).replace('%', '%%').replace('$', '$$'), ensure_ascii=False)
    service_dir = pathlib.Path.home()/'.config/systemd/user'
    service_dir.mkdir(parents=True, exist_ok=True)
    (service_dir/UI_SERVICE).write_text(
        '[Unit]\nDescription=Jetson-local BEV frontend and sensor replay\nAfter=network.target\n\n'
        '[Service]\nType=simple\nWorkingDirectory='+str(ROOT)+'\n'
        'Environment=PYTHONUNBUFFERED=1\nExecStart='+quoted(python)+' '+quoted(ROOT/'server.py')+
        ' --local --host 127.0.0.1 --port '+str(port)+' --dataset '+quoted(dataset)+'\n'
        'Restart=on-failure\nRestartSec=3\n\n[Install]\nWantedBy=default.target\n')
    command('systemctl', '--user', 'daemon-reload')
    command('systemctl', '--user', 'restart', UI_SERVICE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='start', choices=['start', 'stop', 'status'])
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check local prerequisites without starting services')
    parser.add_argument('--dataset', default=str(ROOT/'data/nuscenes-long'))
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--fps', type=float, default=20, help='Target replay FPS (actual throughput is measured)')
    parser.add_argument('--engine', default='auto', help='Engine filename in models/, or auto (worker preference)')
    parser.add_argument('--load-mbps', type=float, help='Optional additional physical TCP load, up to 9500 Mbps')
    parser.add_argument('--direction', choices=['input', 'output', 'bidir'], default='input')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.fps <= 20:
        parser.error('Port must be 1..65535 and FPS must be 1..20.')
    if args.load_mbps is not None and (not math.isfinite(args.load_mbps) or not 0 <= args.load_mbps <= 9500):
        parser.error('Load must be between 0 and 9500 Mbps.')
    if args.engine != 'auto' and not re.fullmatch(r'[A-Za-z0-9_.-]+\.engine', args.engine):
        parser.error('Use an engine filename, not a path.')
    ui = f'http://127.0.0.1:{args.port}'
    if args.action == 'stop':
        command('systemctl', '--user', 'stop', UI_SERVICE)
        print('Local UI/replay stopped. Backend services and the Ethernet link remain available.')
        return
    if args.action == 'status':
        command('systemctl', '--user', 'status', UI_SERVICE, '--no-pager', check=False)
        command('systemctl', 'is-active', 'jetson-bev-net', 'jetson-bev-worker', 'jetson-bev-gateway', check=False)
        status = request(ui+'/api/status')
        print(json.dumps({k: status['state'].get(k) for k in (
            'replay_location', 'connected', 'processed', 'actual_fps', 'gateway_url', 'error')}, indent=2))
        return
    dataset = pathlib.Path(args.dataset).resolve()
    python, manifest = preflight(dataset, args.engine)
    if args.check:
        print('Local prerequisites passed. No services were changed.')
        return
    active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', UI_SERVICE]).returncode == 0
    if not active:
        import socket
        with socket.socket() as listener:
            try:
                listener.bind(('127.0.0.1', args.port))
            except OSError as exc:
                raise RuntimeError(f'Port {args.port} is already occupied; stop its server or use --port.') from exc
    else:
        command('systemctl', '--user', 'stop', UI_SERVICE)
    configure_backend(args.engine)
    validate_first_frame(dataset, manifest)
    configure_ui(python, dataset, args.port)
    wait_for(ui+'/api/status', lambda value: value['state'].get('replay_location') == 'jetson')
    settings = {'fps': args.fps}
    if args.load_mbps is not None:
        settings.update(load_mbps=args.load_mbps, load_direction=args.direction)
    control(ui, settings)
    wait_for(ui+'/api/status', lambda value: value['state'].get('connected') is True)
    print(f'Frontend and backend ready. Open on the Jetson: {ui}/\n'
          f'UI logs: journalctl --user -u {UI_SERVICE} -f\n'
          'GPU logs: sudo journalctl -u jetson-bev-worker -f', flush=True)
    if not args.no_browser and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        browser = next((path for path in ('/snap/bin/firefox', '/snap/bin/chromium')
                        if pathlib.Path(path).exists()), None)
        launch = [browser, '--new-window', ui+'/'] if browser else ['xdg-open', ui+'/']
        if browser or shutil.which('xdg-open'):
            log_dir = ROOT/'logs'
            log_dir.mkdir(exist_ok=True)
            with (log_dir/'browser-launch.log').open('ab') as log:
                subprocess.Popen(launch, stdout=log, stderr=log, start_new_session=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Startup failed: {error}\nInspect: sudo journalctl -u jetson-bev-worker -u jetson-bev-gateway -n 60 --no-pager', file=sys.stderr)
        sys.exit(1)
