"""Balance only the demo NIC queue IRQs, with exact runtime rollback."""
import argparse
import json
import os
import pathlib
import re

STATE = pathlib.Path(__file__).resolve().parent.parent/'deploy/irq-affinity.json'


def irq_names(ports):
    found = {}
    for line in pathlib.Path('/proc/interrupts').read_text().splitlines():
        match = re.match(r'\s*(\d+):.*\s(\S+)\s*$',line)
        if match and any(re.fullmatch(re.escape(port)+r'\.vm[0-3]',match[2]) for port in ports):
            found[match[2]] = match[1]
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action',choices=['start','stop'])
    args = ap.parse_args()
    ports = [os.environ['BEV_LINK_IFACE'],os.environ['BEV_EDGE_IFACE']]
    names = irq_names(ports)
    if args.action == 'stop':
        if STATE.exists():
            saved=json.loads(STATE.read_text())
            for name,old in saved.items():
                if name in names:
                    pathlib.Path('/proc/irq',names[name],'smp_affinity_list').write_text(old['original'])
            STATE.unlink()
        return
    saved=json.loads(STATE.read_text()) if STATE.exists() else {}
    online=pathlib.Path('/sys/devices/system/cpu/online').read_text().strip()
    available=[]
    for part in online.split(','):
        bounds=part.split('-')
        available.extend(range(int(bounds[0]),int(bounds[-1])+1))
    cpus=available[2:] or available
    for index,(name,irq) in enumerate(sorted(names.items())):
        path=pathlib.Path('/proc/irq',irq,'smp_affinity_list')
        saved.setdefault(name,{'original':path.read_text().strip()})
        desired=str(cpus[index%len(cpus)])
        saved[name]['assigned']=desired
        saved[name]['irq']=irq
    STATE.write_text(json.dumps(saved,indent=2))
    for name,record in saved.items():
        if name in names:
            pathlib.Path('/proc/irq',names[name],'smp_affinity_list').write_text(record['assigned'])
    print(json.dumps(saved,indent=2))


if __name__=='__main__':
    main()
