"""Prune the official MIT YOLOP ONNX to its road/lane heads and batch six cameras."""
import hashlib
import json
import pathlib

import onnx
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = 'https://raw.githubusercontent.com/hustvl/YOLOP/main/weights/yolop-640-640.onnx'
SOURCE_SHA256 = 'cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696'


def main():
    original = ROOT/'models/yolop-upstream-640.onnx'
    if not original.exists():
        with requests.get(SOURCE, stream=True, timeout=60) as response:
            response.raise_for_status()
            with original.open('wb') as stream:
                for chunk in response.iter_content(1 << 20):
                    stream.write(chunk)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError('YOLOP upstream checksum mismatch')
    target = ROOT/'models/yolop-road-b6.onnx'
    onnx.utils.extract_model(str(original), str(target), ['images'],
                             ['drive_area_seg', 'lane_line_seg'])
    model = onnx.load(target)
    for tensor in list(model.graph.input) + list(model.graph.output):
        tensor.type.tensor_type.shape.dim[0].dim_value = 6
    # Upstream shape annotations describe batch one, including internal tensors.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    manifest = {
        'model': 'YOLOP road / lane segmentation', 'license': 'MIT',
        'source': SOURCE, 'source_sha256': digest,
        'onnx_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'input': [6, 3, 640, 640], 'outputs': ['drive_area_seg', 'lane_line_seg'],
        'preprocess': 'RGB / 255; ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]; centered 114 letterbox',
        'training_data': 'BDD100K; nuScenes domain accuracy not measured',
        'engine': 'Build on target: yolop-road-thor-b6.engine, TensorRT FP16',
        'ground_truth_used': False,
    }
    (ROOT/'models/manifest-road.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'{target}: {target.stat().st_size/2**20:.1f} MiB, segmentation heads only')


if __name__ == '__main__':
    main()
