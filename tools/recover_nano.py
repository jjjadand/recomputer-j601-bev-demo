"""Restore the verified nano engine after the medium engine disconnected the target."""
from remote import connect, run


def main():
    client, password = connect()
    try:
        code, out, err = run(client, 'systemctl --user stop jetson-bev-local-ui')
        print(out, err, end='')
        command = (
            "mkdir -p /etc/systemd/system/jetson-bev-worker.service.d && "
            "printf '[Service]\\nEnvironment=BEV_VISION_ENGINE=yolo26n-thor-b6.engine\\n' "
            "> /etc/systemd/system/jetson-bev-worker.service.d/90-local-engine.conf && "
            "systemctl daemon-reload && systemctl restart jetson-bev-worker"
        )
        code, out, err = run(client, command, password, sudo=True)
        print(out, err, end='')
        if code:
            raise RuntimeError('Failed to restore nano engine')
        code, out, err = run(client, 'systemctl --user restart jetson-bev-local-ui')
        print(out, err, end='')
        if code:
            raise RuntimeError('Failed to resume local replay')
        print('Restored YOLO26n and local replay. View through ssh jetson-bev / localhost:18088.')
    finally:
        client.close()


if __name__ == '__main__':
    main()
