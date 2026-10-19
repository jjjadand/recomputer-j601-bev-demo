"""Actual physical Ethernet counters in the caller's network namespace."""
import pathlib
import re
import subprocess
import time


def read_port(interface, name, default=''):
    try:
        return pathlib.Path(f'/sys/class/net/{interface}/{name}').read_text().strip()
    except OSError:
        return default


def counter(interface, name):
    try:
        return int(read_port(interface, 'statistics/'+name, '0'))
    except ValueError:
        return 0


class PortSampler:
    def __init__(self, interfaces=None, namespace='host'):
        self.interfaces = interfaces
        self.namespace = namespace
        self.previous = {}
        self.cached = []
        self.last = 0

    def sample(self):
        now = time.monotonic()
        if now-self.last < .5:
            return self.cached
        self.last = now
        names = self.interfaces
        if names is None:
            names = [p.name for p in sorted(pathlib.Path('/sys/class/net').iterdir())
                     if p.name not in ('usb0','usb1') and read_port(p.name, 'type') == '1'
                     and (p/'device').exists() and not (p/'wireless').exists()]
        ports = []
        for name in names:
            if not pathlib.Path('/sys/class/net', name).exists():
                continue
            kernel_rx, kernel_tx = counter(name, 'rx_bytes'), counter(name, 'tx_bytes')
            rx, tx = kernel_rx, kernel_tx
            source = 'linux_interface'
            admin_up = bool(int(read_port(name, 'flags', '0'), 16) & 1)
            # This Thor nvethernet build can report corrupt Linux tx_bytes after
            # TSO completions. MAC MMC octet counters agree with peer RX and iperf.
            try:
                # nvethernet logs an error on every stats query of a down port.
                result = subprocess.run(['ethtool','-S',name],capture_output=True,text=True,timeout=1) if admin_up else None
                hardware = {key:int(value) for key,value in re.findall(r'^\s*(\w+):\s*(\d+)\s*$',result.stdout,re.M)} if result else {}
                if 'mmc_tx_octetcount_gb' in hardware and 'mmc_rx_octetcount_gb' in hardware:
                    rx, tx = hardware['mmc_rx_octetcount_gb'], hardware['mmc_tx_octetcount_gb']
                    source = 'mac_hardware'
            except (OSError,subprocess.TimeoutExpired):
                hardware = {}
            previous = self.previous.get(name)
            elapsed = now-previous[0] if previous else 0
            carrier = read_port(name, 'carrier') == '1'
            speed_text = read_port(name, 'speed')
            speed = int(speed_text) if carrier and speed_text.isdigit() and int(speed_text)>0 else None
            port = {'interface': name, 'namespace': self.namespace, 'carrier': carrier,
                    'operstate': read_port(name, 'operstate', 'unknown'), 'speed_mbps': speed,
                    'duplex': read_port(name, 'duplex', 'unknown'), 'mac': read_port(name, 'address'),
                    'mtu': int(read_port(name, 'mtu', '0')), 'rx_bytes': rx, 'tx_bytes': tx,
                    'counter_source': source, 'kernel_rx_bytes': kernel_rx, 'kernel_tx_bytes': kernel_tx,
                    'rx_mbps': round(max(0,rx-previous[1])*8/elapsed/1e6,3) if elapsed else 0,
                    'tx_mbps': round(max(0,tx-previous[2])*8/elapsed/1e6,3) if elapsed else 0}
            for key in ('rx_packets','tx_packets','rx_dropped','tx_dropped','rx_errors','tx_errors'):
                port[key] = counter(name,key)
            port['rx_crc_errors'] = hardware.get('mmc_rx_crc_error')
            ports.append(port)
            self.previous[name] = (now,rx,tx)
        self.cached = ports
        return ports
