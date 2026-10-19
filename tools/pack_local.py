"""Bundle the local frontend, target backend, and paired packets for offline transfer."""
import argparse
import hashlib
import json
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/jetson-bev-local.tar.gz')
    args = parser.parse_args()
    dataset = ROOT/'data/nuscenes-long'
    manifest = json.loads((dataset/'manifest.json').read_text())
    files = [ROOT/name for name in (
        'start-local.sh', 'server.py', 'json_codec.py', 'README.md', 'requirements-jetson.txt',
        'tools/start_local.py', 'node_modules/three/build/three.module.js',
        'node_modules/three/build/three.core.js', 'node_modules/three/LICENSE',
        'deploy/network.env', 'deploy/jetson-bev-net.service',
        'deploy/jetson-bev-worker.service', 'deploy/jetson-bev-gateway.service')]
    for file in files:
        if not file.is_file():
            raise RuntimeError(f'Missing bundle prerequisite: {file}')
    files += sorted((ROOT/'frontend').glob('*'))
    files += sorted((ROOT/'jetson').glob('*.py'))+sorted((ROOT/'jetson').glob('*.sh'))
    files += sorted((ROOT/'tests').glob('*.py'))
    files += [file for file in sorted((ROOT/'models').glob('*.onnx')) if '-upstream-' not in file.name]
    files += sorted((ROOT/'models').glob('manifest*.json'))+sorted((ROOT/'models').glob('*LICENSE'))
    files += [dataset/'manifest.json']
    files += [(dataset/frame['packet']).resolve() for frame in manifest['frames']]
    files = sorted(set(file for file in files if file.is_file()))
    for frame in manifest['frames']:
        file = (dataset/frame['packet']).resolve()
        if not file.is_relative_to(dataset) or not file.is_file():
            raise RuntimeError(f'Missing or invalid paired packet: {frame["packet"]}')
        if hashlib.sha256(file.read_bytes()).hexdigest() != frame['sha256']:
            raise RuntimeError(f'Sensor packet checksum mismatch: {file.name}')
    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Venv, passwords, IRQ rollback state, logs and host-specific engines are excluded.
    with tarfile.open(output, 'w:gz', compresslevel=1) as archive:
        for file in files:
            if not file.is_relative_to(ROOT):
                raise RuntimeError(f'File outside project: {file}')
            archive.add(file, arcname=str(pathlib.Path('jetson-bev-demo')/file.relative_to(ROOT)), recursive=False)
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    output.with_suffix(output.suffix+'.sha256').write_text(f'{digest.hexdigest()}  {output.name}\n')
    print(f'{output}: {output.stat().st_size/2**20:.1f} MiB; {len(manifest["frames"])} frames; {len(files)} files.')


if __name__ == '__main__':
    main()
