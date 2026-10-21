#!/usr/bin/env python3
"""Stamp/check the flat 35D deployment using its authenticated training package.

Only metadata is changed. The source package and its verification receipt remain
untouched; graph identity and batch-one inference are checked after stamping.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

import numpy as np
import onnx
import onnxruntime as ort
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_layout(bridge, contract, spec):
    names = ['left_hip_joint', 'left_knee_joint', 'right_hip_joint',
             'right_knee_joint', 'left_wheel', 'right_wheel']
    native = ['L_joint1', 'LL_joint1', 'R_joint1', 'RR_joint1', 'L_joint3', 'R_joint3']
    require(contract['policy_action_order'] == native, 'Training action order changed')
    require(bridge['joint_names'] == names, 'YAML joint order differs')
    require([bridge['default_joint_pos'][name] for name in names]
            == [spec['nominal_joint_pos'][name] for name in native], 'Nominal pose differs')
    require(contract['actor_layout'] == [
        'command_xyz3', 'height_command1_times5', 'imu_gyro3_times0.5',
        'imu_projected_gravity3', 'motor_position_delta6_wheels_zeroed',
        'motor_velocity6_times0.1', 'previous_policy_action6',
        'command_context7_no_contact_phase'], 'Unknown training observation layout')
    require((bridge['rl_obs_size'], bridge['rl_feature_size'], bridge['history_length'],
             bridge['rl_action_size']) == (contract['actor_dim'], contract['actor_frame_dim'],
                                         contract['history_length'], contract['action_dim'])
            == (35, 35, 1, 6), 'Observation/action/history dimensions differ')
    terms = [dict(token.split('=', 1) for token in shlex.split(term))
             for term in bridge['observation_terms']]
    # Verify the physical source order and normalization before recording the layout.
    base = bridge['rl_base']
    expected = [
        (0, {'path': base + '/input/command', 'take': 'x'}),
        (1, {'path': base + '/input/command', 'take': 'y'}),
        (2, {'path': base + '/input/command', 'take': 'z'}),
        (3, {'path': base + '/input/height', 'scale': '5.0'}),
        (4, {'path': base + '/input/angular_velocity', 'take': 'vec3', 'scale': '0.5'}),
        (7, {'path': base + '/input/projected_gravity', 'take': 'vec3'}),
        (10, {'type': 'joint_pos', 'joints': ','.join(names[:4]), 'relative': 'true'}),
        (14, {'type': 'joint_pos', 'joints': ','.join(names[4:]), 'zero': 'true'}),
        (16, {'type': 'joint_vel', 'joints': ','.join(names), 'scale': '0.1'}),
        *[(22 + i, {'path': base + '/effective_action/' + name})
          for i, name in enumerate(names)],
        (28, {'type': 'constant', 'value': '1,0,0,0,0,0,0'}),
    ]
    require(len(terms) == len(expected), 'Observation term count differs')
    for term, (index, fields) in zip(terms, expected):
        fields = dict(fields, index=str(index), clip='-100:100')
        require(all(term.get(k, '1' if k == 'scale' else None) == v for k, v in fields.items()),
                f'Observation slot {index} differs from training')
        require('scale' in fields or float(term.get('scale', 1)) == 1,
                f'Unexpected scale at slot {index}')
    require(bridge['joint_base_path'] == base + '/model', 'Joint source path differs')
    control = contract['v5_control']
    for i, (text, name) in enumerate(zip(bridge['action_terms'], names)):
        term = dict(token.split('=', 1) for token in shlex.split(text))
        scale = control['leg_position_scale'] if i < 4 else control['wheel_velocity_scale']
        bound = scale * (control['action_clip'] if i < 4 else control['wheel_action_clip'])
        require(int(term['index']) == i and term['output'] == base + '/target/' + name
                and float(term['scale']) == scale
                and [float(v) for v in term['clip'].split(':')] == [-bound, bound],
                f'Action slot {i} differs from training')
    require(len(bridge['action_terms']) == 6, 'Action term count differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-package', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--stamp', action='store_true', help='Write validated metadata in place')
    args = parser.parse_args()
    proof_path = args.source_package / 'verification.json'
    proof = json.loads(proof_path.read_text())
    sources = list(args.source_package.glob('*.onnx'))
    require(len(sources) == 1, 'Expected exactly one source ONNX')
    source = sources[0]
    require(digest(source) == proof['model_sha256'], 'Source ONNX differs from verification')
    plan_path = args.training_root / proof['training_plan_file']
    plan = json.loads(plan_path.read_text())
    require(digest(plan_path) == proof['training_plan_sha256'], 'Training plan digest differs')
    contract_path = Path(proof['contract'])
    require(digest(contract_path) == proof['contract_sha256'], 'Training contract digest differs')
    contract = json.loads(contract_path.read_text())
    spec_path = args.training_root / plan['asset_directory'] / 'model_spec.json'
    require(digest(spec_path) == plan['model_spec_sha256'], 'Mechanical model digest differs')
    config = yaml.safe_load(args.config.read_text())
    bridge = config['rl_bridge']['ros__parameters']
    chassis = config['wheel_leg_controller']['ros__parameters']
    validate_layout(bridge, contract, json.loads(spec_path.read_text()))
    require(bridge['leg_winding_version'] == chassis['leg_winding_version']
            == plan['leg_target_winding'] == 'paired_branch107_v1', 'Winding differs')
    require(bridge['expected_model_spec_sha256'] == plan['model_spec_sha256'], 'Model spec differs')
    rate = 1 / contract['policy_dt']
    require(bridge['policy_rate'] == proof['policy_hz'] == rate, 'Policy frequency differs')
    require(config['rmcs_executor']['ros__parameters']['update_rate']
            == chassis['pd_frequency'] == chassis['recovery_frequency']
            == proof['physics_hz'] == 1 / contract['physics_dt'], 'Control frequency differs')
    require(bridge['max_action_age'] == bridge['inference_timeout']
            == chassis['inference_timeout'] == 3 / rate, 'Three-period action age differs')
    reference = contract['flat_rmcs_command_training']['contract']
    commands = config['wheel_leg_chassis_controller']['ros__parameters']
    require(chassis['vx_max'] == commands['vx_max'] == reference['vx_max_m_s']
            and chassis['yaw_rate_max'] == commands['yaw_rate_max'] == commands['spin_yaw_rate']
            == reference['yaw_rate_max_rad_s']
            and chassis['heading_hold_yaw_rate_max'] == reference['heading_hold_yaw_rate_max_rad_s'],
            'Motion command envelope differs from training')
    source_model = onnx.load(source)
    model = onnx.load(args.model)
    require(model.graph.SerializeToString() == source_model.graph.SerializeToString(),
            'Deployment graph differs from verified source')
    layouts = {key: bridge[key] for key in ('observation_terms', 'action_terms', 'joint_names',
               'default_joint_pos', 'history_length', 'rl_feature_size')}
    metadata = dict(leg_winding_version=plan['leg_target_winding'],
                    leg_target_model_spec_sha256=plan['model_spec_sha256'],
                    rmcs_source_model_sha256=proof['model_sha256'],
                    rmcs_training_contract_sha256=proof['contract_sha256'],
                    rmcs_obs_layout=encode(bridge['observation_terms']),
                    rmcs_actions_layout=encode(bridge['action_terms']),
                    rmcs_layout_hash=hashlib.sha256(encode(layouts).encode()).hexdigest(),
                    rmcs_history_length='1', rmcs_obs_frame_size='35')
    existing = {v.key: v.value for v in model.metadata_props}
    source_metadata = {v.key: v.value for v in source_model.metadata_props}
    require(all(existing.get(k) == v for k, v in source_metadata.items())
            and set(existing) <= set(source_metadata) | set(metadata),
            'Source metadata changed or unknown preprocessing metadata was added')
    require(existing.get('policy_hz') == str(round(rate)), 'ONNX policy_hz differs')
    require(existing.get('source_checkpoint_sha256') == proof['checkpoint_sha256'],
            'ONNX checkpoint lineage differs')
    if args.stamp:
        onnx.helper.set_model_props(model, dict(existing, **metadata))
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, args.model)
    else:
        require(all(existing.get(k) == v for k, v in metadata.items()), 'Deployment metadata differs')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    original = ort.InferenceSession(str(source), sess_options=options, providers=['CPUExecutionProvider'])
    deployed = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
    values = np.random.default_rng(20261021).uniform(-2, 2, (256, 35)).astype(np.float32)
    values[:, 3] = .305 * 5
    values[:, 7:10] = [0, 0, -1]
    values[:, 28:] = [1, 0, 0, 0, 0, 0, 0]
    maximum = 0.
    for value in values:
        a = original.run(None, {'obs': value[None]})[0]
        b = deployed.run(None, {'obs': value[None]})[0]
        require(a.shape == b.shape == (1, 6) and np.isfinite(b).all(), 'Invalid inference output')
        require(np.array_equal(a, b), 'Metadata stamping changed inference')
        maximum = max(maximum, float(np.abs(a - b).max()))
    receipt = dict(schema='rmcs-flat-policy-deployment-v1', model_sha256=digest(args.model),
                   model_filename=args.model.name,
                   source_model_sha256=proof['model_sha256'], graph_unchanged=True,
                   model_spec_sha256=plan['model_spec_sha256'],
                   training_plan_sha256=digest(plan_path), training_contract_sha256=digest(contract_path),
                   source_verification_sha256=digest(proof_path),
                   layout_hash=metadata['rmcs_layout_hash'], policy_hz=rate,
                   physics_hz=proof['physics_hz'], inference_samples=256,
                   action_timeout_seconds=3 / rate,
                   vx_max_m_s=reference['vx_max_m_s'],
                   yaw_rate_max_rad_s=reference['yaw_rate_max_rad_s'],
                   heading_hold_yaw_rate_max_rad_s=reference['heading_hold_yaw_rate_max_rad_s'],
                   inference_max_abs_difference=maximum,
                   behavior_qualified=proof['behavior_qualified'],
                   hardware_deployment_ready=proof['hardware_deployment_ready'])
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
