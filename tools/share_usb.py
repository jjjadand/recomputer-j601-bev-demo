"""Start/stop scoped PC Internet forwarding to this Jetson's existing USB address."""
import argparse
import getpass
import json
import subprocess

from remote import connect, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop"], default="start", nargs="?")
    args = ap.parse_args()
    routes = lambda target: json.loads(subprocess.check_output(["ip", "-j", "route", "get", target]))[0]
    downstream = routes("192.168.55.18")["dev"]
    upstream = routes("1.1.1.1")["dev"]
    if upstream == downstream:
        raise RuntimeError("The PC upstream must be different from the Jetson management link")
    password = getpass.getpass("PC sudo password: ")
    def sudo(command, check=True):
        return subprocess.run(["sudo", "-S", "-p", ""]+command, input=password+"\n",
                              text=True, capture_output=True, check=check)
    rules = [
        ["iptables", "-t", "nat", "-C", "POSTROUTING", "-s", "192.168.55.18/32", "-o", upstream,
         "-m", "comment", "--comment", "jetson-bev-usb-share", "-j", "MASQUERADE"],
        ["iptables", "-C", "FORWARD", "-i", downstream, "-o", upstream, "-s", "192.168.55.18/32",
         "-m", "comment", "--comment", "jetson-bev-usb-share", "-j", "ACCEPT"],
        ["iptables", "-C", "FORWARD", "-i", upstream, "-o", downstream, "-d", "192.168.55.18/32",
         "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-m", "comment",
         "--comment", "jetson-bev-usb-share", "-j", "ACCEPT"],
    ]
    if args.action == "start":
        if subprocess.check_output(["sysctl", "-n", "net.ipv4.ip_forward"]).strip() != b"1":
            raise RuntimeError("Enable PC IP forwarding before sharing: sudo sysctl -w net.ipv4.ip_forward=1")
    for rule in rules:
        exists = sudo(rule, check=False).returncode == 0
        if args.action == "start" and not exists:
            rule[rule.index("-C")] = "-A" if "-t" in rule else "-I"
            sudo(rule)
        elif args.action == "stop" and exists:
            rule[rule.index("-C")] = "-D"
            sudo(rule)
    client, jetson_password = connect()
    try:
        if args.action == "start":
            command = "ip route replace default via 192.168.55.100 dev l4tbr0 metric 100; resolvectl dns l4tbr0 1.1.1.1; resolvectl domain l4tbr0 '~.'"
        else:
            command = "ip route del default via 192.168.55.100 dev l4tbr0 metric 100; resolvectl revert l4tbr0"
        code, out, err = run(client, command, jetson_password, sudo=True)
        if code:
            raise RuntimeError(err)
        if args.action == "start":
            code, out, err = run(client, "ping -c 1 -W 2 192.168.55.100; ping -c 1 -W 3 1.1.1.1; resolvectl query pypi.org; curl -I --max-time 10 https://pypi.org")
            print(out, err)
            if code:
                raise RuntimeError("Jetson HTTPS verification failed")
        print(f"Sharing {args.action}: {upstream} -> {downstream} -> 192.168.55.18")
    finally:
        client.close()


if __name__ == "__main__":
    main()
