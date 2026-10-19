"""Deployment helper. Passwords are prompted or read from the process environment."""
import argparse
import base64
import getpass
import hashlib
import os
import shlex
import sys

import paramiko

FINGERPRINT = "z/CTdkrvvq/Tr47vifjoQ6+N3LXozGlh6aPvNQMcKqA"


class VerifiedHost(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if actual != FINGERPRINT:
            raise RuntimeError(f"Unexpected Jetson host key for {hostname}: {actual}")


def connect(host="192.168.55.18", password=None):
    password = password or os.environ.get("JETSON_PASSWORD") or getpass.getpass("Jetson password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(VerifiedHost())
    client.connect(host, username="seeed", password=password, look_for_keys=False,
                   allow_agent=False, timeout=8)
    return client, password


def run(client, command, password=None, sudo=False, timeout=60):
    if sudo:
        command = "sudo -S -p '' /bin/bash -c " + shlex.quote(command)
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    if sudo:
        stdin.write(password + "\n")
        stdin.flush()
    stdin.channel.shutdown_write()
    out, err = stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.168.55.18")
    p.add_argument("--sudo", action="store_true")
    p.add_argument("command")
    args = p.parse_args()
    client, password = connect(args.host)
    try:
        code, out, err = run(client, args.command, password, args.sudo)
        sys.stdout.write(out)
        sys.stderr.write(err)
        return code
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
