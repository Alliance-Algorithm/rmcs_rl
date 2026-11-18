#!/usr/bin/env python3
"""Check an RlBridge YAML block, ONNX contract and CPU inference without ROS."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]


def load_block(config, component):
    import yaml
    document = yaml.safe_load(Path(config).read_text())
    return document['rmcs_executor']['ros__parameters']['update_rate'], document[component]['ros__parameters']


def frame_layout(block):
    actions = len(block['actions']['order'])
    layout = []
    for name in block['observations']['order']:
        term = block['observations'][name]
        kind = term.get('type', 'scalar')
        if kind == 'scalar':
            size = 1
        elif kind == 'vector3':
            size = 3
        elif kind == 'array':
            size = term['size']
        elif kind == 'scalars':
            size = len(term['sources'])
        elif kind == 'previous_action':
            size = actions
        else:
            raise ValueError(f'observations.{name}: unsupported type {kind}')
        if type(size) is not int or size <= 0:
            raise ValueError(f'observations.{name}: source width must be positive')
        selection = term.get('select', [])
        if any(type(i) is not int or not 0 <= i < size for i in selection):
            raise ValueError(f'observations.{name}: select index outside source width {size}')
        layout.append(dict(name=name, source_size=size, size=len(selection) if selection else size))
    if not layout or not actions:
        raise ValueError('observations.order and actions.order must be nonempty')
    return layout


def check_rates(update_rate, block):
    policy = block['policy_rate']
    history = block.get('history_rate', policy)
    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
                for v in (update_rate, policy, history)) or policy < history
            or math.fmod(update_rate, policy) != 0 or math.fmod(policy, history) != 0):
        raise ValueError('update_rate, policy_rate and history_rate must divide evenly')
    return dict(update=float(update_rate), policy=float(policy), history=float(history))


def check_model(model, block, frame, history):
    import onnx
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError('model must have one input and one output')
    source, target = model.graph.input[0].type.tensor_type, model.graph.output[0].type.tensor_type
    if source.elem_type != onnx.TensorProto.FLOAT or target.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError('model input and output must be float32')
    shapes = {1: [history * frame], 2: [1, history * frame], 3: [1, history, frame]}
    expected = shapes.get(len(source.shape.dim))
    if expected is None or any(d.HasField('dim_value') and d.dim_value > 0 and d.dim_value != n
                               for d, n in zip(source.shape.dim, expected)):
        raise ValueError('model shape does not match configuration')
    action_size = len(block['actions']['order'])
    if not target.shape.dim:
        raise ValueError('model output must have at least one dimension')
    if all(d.HasField('dim_value') and d.dim_value > 0 for d in target.shape.dim):
        if math.prod(d.dim_value for d in target.shape.dim) != action_size:
            raise ValueError('model shape does not match configuration')
    return expected


def check_metadata(model, block):
    import numpy as np
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    errors = []
    if 'rmcs_observation_scales' in metadata:
        try:
            expected_scales = [float(v) for v in metadata['rmcs_observation_scales'].split(',')]
            configured = [float(block['observations'][term['name']].get('scale', 1.))
                          for term in frame_layout(block) for _ in range(term['size'])]
            if expected_scales != configured or not all(math.isfinite(v) for v in expected_scales):
                errors.append('model observation scales do not match YAML preprocessing')
        except (ValueError, TypeError):
            errors.append('invalid rmcs_observation_scales metadata')
    for key, expected in block.get('expected_metadata', {}).items():
        if not isinstance(expected, str) or metadata.get(key) != expected:
            errors.append(f'model metadata mismatch: {key}')
    size = sum(t['size'] for t in frame_layout(block)) * block['history_length']
    mean, deviation = metadata.get('rmcs_obs_mean'), metadata.get('rmcs_obs_std')
    if mean is not None or deviation is not None:
        try:
            means = [float(x) for x in mean.split(',')]
            stds = [float(x) for x in deviation.split(',')]
            if (len(means) != size or len(stds) != size or not np.isfinite(means).all()
                    or not np.isfinite(stds).all() or any(v <= 0 for v in stds)):
                raise ValueError()
        except (AttributeError, ValueError):
            errors.append('model normalization does not match configuration')
    for key in ('rmcs_obs_clip', 'rmcs_action_clip'):
        if key in metadata:
            try:
                value = float(metadata[key])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError()
            except ValueError:
                errors.append('invalid model clipping metadata: ' + key)
    if errors:
        raise ValueError('\n'.join(errors))
    return metadata


def check_inference(model, block, n=256):
    import numpy as np
    import onnxruntime as ort
    frame = sum(t['size'] for t in frame_layout(block))
    shape = check_model(model, block, frame, block['history_length'])
    metadata = check_metadata(model, block)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    clip = float(metadata.get('rmcs_obs_clip', '10'))
    mean = np.array([float(x) for x in metadata['rmcs_obs_mean'].split(',')]) if 'rmcs_obs_mean' in metadata else None
    std = np.array([float(x) for x in metadata['rmcs_obs_std'].split(',')]) if mean is not None else None
    rng = np.random.default_rng(107)
    for _ in range(n):
        observation = rng.uniform(-clip, clip, size=math.prod(shape))
        if mean is not None:
            observation = (observation - mean) / std
        if 'rmcs_obs_clip' in metadata:
            observation = np.clip(observation, -clip, clip)
        result = session.run(None, {session.get_inputs()[0].name: observation.astype(np.float32).reshape(shape)})[0]
        if result.size != len(block['actions']['order']):
            raise ValueError('model shape does not match configuration')
        if not np.isfinite(result).all():
            raise ValueError('model inference returned non-finite actions')
    return n


def generic_path(path):
    path = Path(path).resolve()
    root = next((p for p in (PACKAGE, *PACKAGE.parents) if (p / 'rmcs_ws').is_dir()), PACKAGE)
    if not path.is_relative_to(root):
        raise ValueError('Receipt paths must be inside the RMCS repository')
    return 'RMCS:' + path.relative_to(root).as_posix()


def write_receipt(path, model, config, component, block, rates, layout, metadata, samples):
    receipt = dict(schema='rmcs-policy-check-v1',
        model=dict(path=generic_path(model), sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest()),
        config=dict(path=generic_path(config), component=component,
                    block_sha256=hashlib.sha256(json.dumps(block, sort_keys=True, separators=(',', ':')).encode()).hexdigest()),
        rates=rates, shape=dict(history_length=block['history_length'], frame_size=sum(t['size'] for t in layout),
                                action_size=len(block['actions']['order'])), metadata=metadata, inference_samples=samples)
    with Path(path).open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--component', default='rl_bridge')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    try:
        import onnx
        update_rate, block = load_block(args.config, args.component)
        errors = []
        rates = layout = None
        for check in ('rates', 'layout'):
            try:
                if check == 'rates':
                    rates = check_rates(update_rate, block)
                else:
                    layout = frame_layout(block)
                    if type(block['history_length']) is not int or block['history_length'] <= 0:
                        raise ValueError('history_length must be a positive integer')
            except (KeyError, ValueError, TypeError) as error:
                errors.extend(str(error).splitlines())
        path = args.model or Path(block['rl_model_path'])
        model_path = path if path.is_absolute() else PACKAGE / path
        model = None
        if not model_path.is_file():
            errors.append('model file is missing: ' + str(model_path))
        else:
            model = onnx.load(str(model_path))
        if model is not None and layout is not None:
            for check in (lambda: check_model(model, block, sum(t['size'] for t in layout), block['history_length']),
                          lambda: check_metadata(model, block)):
                try:
                    check()
                except (KeyError, ValueError, TypeError) as error:
                    errors.extend(str(error).splitlines())
        if errors:
            raise ValueError('\n'.join(errors))
        samples = check_inference(model, block)
        if args.receipt:
            write_receipt(args.receipt, model_path, args.config, args.component, block, rates, layout,
                          check_metadata(model, block), samples)
        print(f'Policy check passed: obs={block["history_length"]}x{sum(t["size"] for t in layout)} '
              f'action={len(block["actions"]["order"])} policy={rates["policy"]:g} Hz history={rates["history"]:g} Hz; {samples} samples')
        return 0
    except Exception as error:
        for reason in str(error).splitlines():
            print('FAIL: ' + reason, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
