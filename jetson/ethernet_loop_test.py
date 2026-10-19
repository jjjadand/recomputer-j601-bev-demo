"""Bounded, isolated physical-cable test; inference keeps its software route.

Run as root on Jetson after verifying the requested ports are cabled together.
Ports must have no assigned global addresses. State is restored on exit.
"""
import argparse
import json
import os
import pathlib
import subprocess
import time

NAMESPACE = 'bev-cable-test'
HOST_CIDR, PEER_CIDR = '10.79.0.1/30', '10.79.0.2/30'


def run(*args, check=True, timeout=15):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'{args}: {result.stderr.strip() or result.stdout.strip()}')
    return result


def read(port, field):
    return pathlib.Path('/sys/class/net', port, field).read_text().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ports', nargs=2, required=True, choices=[f'mgbe{i}_0' for i in range(4)])
    parser.add_argument('--output', default='logs/ethernet-loop-test.json')
    args = parser.parse_args()
    host, peer = args.ports
    if os.geteuid() != 0 or host == peer:
        parser.error('Run with sudo and choose two different ports')
    report = {'ports': args.ports, 'passed': False, 'tests': [], 'started_at': time.time(),
              'inference_route_changed': False}
    state = {}
    server = None
    moved = namespace_created = address_added = False
    for port in args.ports:
        addresses = json.loads(run('ip', '-j', 'addr', 'show', 'dev', port).stdout)[0]['addr_info']
        if any(address['scope'] == 'global' for address in addresses):
            raise RuntimeError(f'{port} has global addresses; choose unused dedicated ports')
        state[port] = {'up': bool(int(read(port, 'flags'), 16) & 1),
                       'managed': run('nmcli', '-g', 'GENERAL.NM-MANAGED', 'device', 'show', port).stdout.strip(),
                       'mtu': int(read(port, 'mtu'))}
    if NAMESPACE in run('ip', 'netns', 'list').stdout:
        raise RuntimeError('Test namespace already exists; inspect it before retrying')
    report['original_state'] = state
    try:
        for port in args.ports:
            run('nmcli', 'device', 'set', port, 'managed', 'no')
            run('ip', 'link', 'set', port, 'up')
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            if all(read(port, 'carrier') == '1' for port in args.ports):
                break
            time.sleep(.5)
        else:
            raise RuntimeError('No carrier on both selected ports; traffic test was not started')
        report['negotiated_mbps'] = {port: int(read(port, 'speed')) for port in args.ports}
        run('ip', 'netns', 'add', NAMESPACE); namespace_created = True
        run('ip', 'link', 'set', peer, 'netns', NAMESPACE); moved = True
        run('ip', 'netns', 'exec', NAMESPACE, 'ip', 'link', 'set', 'lo', 'up')
        run('ip', 'netns', 'exec', NAMESPACE, 'ip', 'link', 'set', peer, 'up')
        run('ip', 'addr', 'add', HOST_CIDR, 'dev', host); address_added = True
        run('ip', 'netns', 'exec', NAMESPACE, 'ip', 'addr', 'add', PEER_CIDR, 'dev', peer)
        report['route'] = run('ip', 'route', 'get', '10.79.0.2').stdout.strip()
        if f'dev {host} ' not in report['route']:
            raise RuntimeError('Peer route does not use the selected physical interface')
        report['ping'] = run('ping', '-I', host, '-c', '5', '-W', '1', '10.79.0.2').stdout
        server = subprocess.Popen(['ip', 'netns', 'exec', NAMESPACE, 'iperf3', '-s',
                                   '-B', '10.79.0.2', '-p', '5202'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(.5)
        for mbps in (100, 1000):
            for reverse in (False, True):
                command = ['iperf3', '-c', '10.79.0.2', '-B', '10.79.0.1', '-p', '5202',
                           '-u', '-b', f'{mbps}M', '-l', '1200', '-t', '3', '-J']
                if reverse:
                    command.append('-R')
                output = json.loads(run(*command, timeout=12).stdout)
                if 'error' in output:
                    raise RuntimeError(output['error'])
                received = output['end']['sum_received']
                measured = received['bits_per_second']/1e6
                test = {'requested_mbps': mbps, 'direction': 'reverse' if reverse else 'forward',
                        'received_mbps': measured, 'lost_percent': received['lost_percent'], 'raw': output}
                report['tests'].append(test)
                print(json.dumps({key: value for key, value in test.items() if key != 'raw'}), flush=True)
                if measured < .8*mbps or received['lost_percent'] > 1:
                    raise RuntimeError('Low-rate cable test throughput/loss check failed')
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if server:
            server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        rollback_errors = []
        def restore(*command):
            try:
                result = run(*command, check=False)
                if result.returncode:
                    rollback_errors.append(result.stderr.strip())
            except Exception as error:
                rollback_errors.append(str(error))
        if moved:
            restore('ip', 'netns', 'exec', NAMESPACE, 'ip', 'link', 'set', peer, 'down')
            restore('ip', 'netns', 'exec', NAMESPACE, 'ip', 'addr', 'del', PEER_CIDR, 'dev', peer)
            restore('ip', 'netns', 'exec', NAMESPACE, 'ip', 'link', 'set', peer, 'netns', '1')
        if address_added:
            restore('ip', 'addr', 'del', HOST_CIDR, 'dev', host)
        if namespace_created:
            restore('ip', 'netns', 'del', NAMESPACE)
        for port, original in state.items():
            restore('nmcli', 'device', 'set', port, 'managed', original['managed'])
            restore('ip', 'link', 'set', port, 'up' if original['up'] else 'down')
        report['rollback_errors'] = rollback_errors
        report['finished_at'] = time.time()
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+'\n')
    if rollback_errors:
        raise RuntimeError('Test completed but interface restoration needs inspection')


if __name__ == '__main__':
    main()
