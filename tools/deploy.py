"""Deploy the worker/gateway to the verified Jetson; credentials are never saved."""
import argparse
import pathlib
import shlex
from remote import connect, run

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = "/home/seeed/jetson-bev-demo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="Create/update the isolated Jetson Python environment")
    ap.add_argument("--build", action="store_true", help="Rebuild the six-camera engine on Thor")
    ap.add_argument("--model", default="models/yolo26n-b6.onnx")
    ap.add_argument("--engine", help="Target TensorRT engine path, relative to the project")
    args = ap.parse_args()
    engine = args.engine or ("models/yolo26m-thor-b6-960.engine" if "yolo26m" in args.model else "models/yolo26n-thor-b6.engine")
    for path in (args.model, engine):
        if pathlib.PurePosixPath(path).is_absolute() or '..' in pathlib.PurePosixPath(path).parts:
            ap.error("Model and engine paths must be relative to the project")
    client, password = connect()
    try:
        def execute(command, sudo=False, timeout=300):
            code, out, err = run(client, command, password, sudo=sudo, timeout=timeout)
            print(out, err, end="")
            if code:
                raise RuntimeError(f"Remote command failed with status {code}")
        execute(f"mkdir -p {TARGET}/jetson {TARGET}/models {TARGET}/deploy {TARGET}/tests")
        with client.open_sftp() as sftp:
            for folder in ("jetson", "deploy", "tests"):
                for file in (ROOT/folder).iterdir():
                    if file.is_file():
                        sftp.put(str(file), f"{TARGET}/{folder}/{file.name}")
            sftp.put(str(ROOT/"requirements-jetson.txt"), TARGET+"/requirements-jetson.txt")
            sftp.put(str(ROOT/"json_codec.py"), TARGET+"/json_codec.py")
            sftp.put(str(ROOT/args.model), TARGET+"/"+args.model)
            for name in ('yolop-road-b6.onnx', 'manifest-road.json', 'YOLOP-LICENSE'):
                sftp.put(str(ROOT/'models'/name), TARGET+'/models/'+name)
        if args.install:
            execute(f"set -e; cd {TARGET}; python3 -m venv --system-site-packages --without-pip .venv; "
                    "curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/jetson-bev-get-pip.py; "
                    ".venv/bin/python /tmp/jetson-bev-get-pip.py; .venv/bin/python -m pip install -r requirements-jetson.txt")
        if args.build:
            execute(f"cd {TARGET}; LD_LIBRARY_PATH=/usr/local/cuda/lib64 .venv/bin/python jetson/build_engine.py "
                    f"--model {shlex.quote(args.model)} --output {shlex.quote(engine)}", timeout=900)
            execute(f"cd {TARGET}; LD_LIBRARY_PATH=/usr/local/cuda/lib64 .venv/bin/python jetson/build_engine.py "
                    "--model models/yolop-road-b6.onnx --output models/yolop-road-thor-b6.engine "
                    "--aux-streams 0 --optimization-level 0", timeout=900)
        execute(f"set -e; cp {TARGET}/deploy/jetson-bev-*.service /etc/systemd/system/; "
                "systemctl daemon-reload; systemctl enable jetson-bev-net jetson-bev-worker jetson-bev-gateway; "
                "systemctl stop jetson-bev-gateway jetson-bev-worker; systemctl restart jetson-bev-net; "
                "systemctl start jetson-bev-worker jetson-bev-gateway; "
                "systemctl is-active jetson-bev-net jetson-bev-worker jetson-bev-gateway", sudo=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
