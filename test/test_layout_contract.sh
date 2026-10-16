#!/usr/bin/env bash
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
TOOLS="$PKG/tool"
PY="${PYTHON:-python3}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmcs-rl-contract.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
FIXTURE="$WORK/x.yaml"
FIXTURE_TYPES="$WORK/x_types.yaml"
MODEL="$WORK/policy.onnx"
LOG="$WORK/rmcs_rl_layout_last.log"

PASS=0
FAIL=0
ok() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

run_expect() {
    local expect="$1" label="$2"
    shift 2
    "$@" >"$LOG" 2>&1
    local rc=$?
    if [ "$rc" = "$expect" ]; then
        ok "$label (exit $rc)"
    else
        bad "$label (期望 exit $expect，实得 $rc)"
        sed 's/^/    | /' "$LOG"
    fi
}

expect_msg() {
    if grep -q -- "$1" "$LOG"; then
        ok "消息含 $2"
    else
        bad "消息缺 $2（未找到 $(printf %q "$1")）"
        sed 's/^/    | /' "$LOG"
    fi
}

cat >"$FIXTURE" <<'YAML'
# rmcs_rl 桥部署夹具（临时验证用；由 test_layout_contract.sh 生成到 /tmp）
rl_bridge:
  ros__parameters:
    rl_obs_size: 20
    rl_action_size: 6

    # obs 顺序 = 索引顺序；index= 是防静默错位的冗余声明（与顺序冲突会 warn）
    observation_terms:
      - index=0 path=/chassis/control_height                        # 标量路径 → 1 维
      - index=1 path=/chassis/control_velocity take=x               # 取分量 → 1 维
      - index=2 path=/wheel_leg/imu/angular_velocity take=vec3 scale=0.5   # 整条向量 → 3 维
      - index=5 path=/wheel_leg/imu/quaternion transform=projected_gravity  # 重力投影 → 3 维
      - index=8 type=joint_pos joints=lf0,lf1 relative=true          # 关节相对位置 → 2 维
      - index=10 type=joint_vel joints=lf0,lf1 scale=0.1             # 关节速度 → 2 维
      - index=12 type=last_action                                    # 上一帧动作 → action_size 维
      - index=18 type=constant value=0,0.5                           # 常量 → 2 维

    action_terms:
      - index=0 output=/wheel_leg/rl/action/lf0 scale=0.5
      - index=1 output=/wheel_leg/rl/action/lf1
      - index=2 output=/wheel_leg/rl/action/l_wheel
      - index=3 output=/wheel_leg/rl/action/rf0
      - index=4 output=/wheel_leg/rl/action/rf1
      - index=5 output=/wheel_leg/rl/action/r_wheel
YAML

sed -e 's|path=/chassis/control_height |path=/chassis/control_height type=scalar |' \
    -e 's|take=x |take=x type=direction_vector |' \
    -e 's|take=vec3 |take=vec3 type=vector3 |' \
    -e 's|transform=projected_gravity |transform=projected_gravity type=quaternion |' \
    "$FIXTURE" >"$FIXTURE_TYPES"

echo "== 1. 词条语法自检（rl_layout.py）=="
run_expect 0 "rl_layout self-test" "$PY" "$HERE/test_layout.py"
expect_msg "rl_layout self-test OK" "self-test OK"

echo "== 2. 生成 → 盖章 → 校验（正例必须 PASS）=="
run_expect 0 "gen_synthetic_policy --from-config" \
    "$PY" "$HERE/gen_synthetic_policy.py" --from-config "$FIXTURE" -o "$MODEL"
run_expect 0 "stamp_layout_metadata" \
    "$PY" "$TOOLS/stamp_layout_metadata.py" --model "$MODEL" --from-config "$FIXTURE"
cat "$LOG"
SHA_BEFORE="$(sha256sum "$MODEL" | cut -d' ' -f1)"
run_expect 0 "stamp_layout_metadata（幂等重跑）" \
    "$PY" "$TOOLS/stamp_layout_metadata.py" --model "$MODEL" --from-config "$FIXTURE"
SHA_AFTER="$(sha256sum "$MODEL" | cut -d' ' -f1)"
if [ "$SHA_BEFORE" = "$SHA_AFTER" ]; then
    ok "盖章幂等（文件字节不变）"
else
    bad "盖章非幂等：$SHA_BEFORE != $SHA_AFTER"
fi
run_expect 0 "check_policy_contract（正例）" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL" --config "$FIXTURE"
cat "$LOG"
expect_msg "== SUMMARY: PASS ==" "SUMMARY PASS"

echo "== 3. 逃生口 type= 不改变布局指纹 =="
SIG_PLAIN="$("$PY" "$TOOLS/check_policy_contract.py" --print-layout --config "$FIXTURE" \
    | grep -E 'obs_signature|action_signature|layout_hash')"
SIG_TYPES="$("$PY" "$TOOLS/check_policy_contract.py" --print-layout --config "$FIXTURE_TYPES" \
    | grep -E 'obs_signature|action_signature|layout_hash')"
if [ -n "$SIG_PLAIN" ] && [ "$SIG_PLAIN" = "$SIG_TYPES" ]; then
    ok "带/不带 type= 的规范串与 layout_hash 完全一致"
else
    bad "type= 改变了布局指纹"
    diff <(echo "$SIG_PLAIN") <(echo "$SIG_TYPES") | sed 's/^/    | /'
fi
echo "$SIG_PLAIN" | sed 's/^/    /'

echo "== 4. 负例（必须 FAIL，exit 1，消息可读）=="
"$PY" - "$MODEL" "$WORK/policy_neg_a.onnx" <<'PYEOF'
import sys, onnx
src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)
keep = [p for p in model.metadata_props if p.key != "rmcs_obs_layout"]
del model.metadata_props[:]
model.metadata_props.extend(keep)
onnx.save(model, dst)
PYEOF
run_expect 1 "(a) metadata 缺 rmcs_obs_layout" \
    "$PY" "$TOOLS/check_policy_contract.py" "$WORK/policy_neg_a.onnx" --config "$FIXTURE"
expect_msg "rmcs_obs_layout" "缺键提示"

"$PY" - "$FIXTURE" "$WORK/x_swapped.yaml" <<'PYEOF'
import re, sys, yaml
src, dst = sys.argv[1], sys.argv[2]
doc = yaml.safe_load(open(src, encoding="utf-8"))
terms = doc["rl_bridge"]["ros__parameters"]["observation_terms"]
t0, t1 = terms[0], terms[1]
i0 = re.search(r"index=\S+", t0).group(0)
i1 = re.search(r"index=\S+", t1).group(0)
body0 = t0.replace(i0, "").strip()
body1 = t1.replace(i1, "").strip()
terms[0] = f"{i0} {body1}"     # 槽位不动（index 自洽），语义换位 → 签名必须变
terms[1] = f"{i1} {body0}"
doc["rl_bridge"]["ros__parameters"]["observation_terms"] = terms
yaml.safe_dump(doc, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PYEOF
run_expect 1 "(b) 真·换序（index 位置不变，语义换位）" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL" --config "$WORK/x_swapped.yaml"
expect_msg "rmcs_obs_layout" "签名不一致提示"

"$PY" - "$FIXTURE" "$WORK/x_index_clash.yaml" <<'PYEOF'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
doc = yaml.safe_load(open(src, encoding="utf-8"))
terms = doc["rl_bridge"]["ros__parameters"]["observation_terms"]
terms[0], terms[1] = terms[1], terms[0]          # 只换顺序，index= 保持原样 = 自相矛盾
yaml.safe_dump(doc, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PYEOF
run_expect 1 "(b2) 换序但不改 index= → index 断言" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL" --config "$WORK/x_index_clash.yaml"
expect_msg "index=" "index 断言提示"

"$PY" - "$FIXTURE" "$WORK/x_scale.yaml" <<'PYEOF'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
doc = yaml.safe_load(open(src, encoding="utf-8"))
terms = doc["rl_bridge"]["ros__parameters"]["observation_terms"]
terms[5] = terms[5].replace("scale=0.1", "scale=0.2")   # joint_vel 缩放
yaml.safe_dump(doc, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PYEOF
run_expect 1 "(c) 改一个 scale=" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL" --config "$WORK/x_scale.yaml"
expect_msg "policy_layout_hash" "hash 不一致提示"

"$PY" - "$MODEL" "$WORK/policy_neg_d.onnx" <<'PYEOF'
import sys, onnx
src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)
for prop in model.metadata_props:
    if prop.key == "policy_layout_hash":
        prop.value = "0xdeadbeefdeadbeef"
onnx.save(model, dst)
PYEOF
run_expect 1 "(d) policy_layout_hash 写错" \
    "$PY" "$TOOLS/check_policy_contract.py" "$WORK/policy_neg_d.onnx" --config "$FIXTURE"
expect_msg "policy_layout_hash" "hash 校验失败提示"

"$PY" - "$FIXTURE" "$WORK/x_actsize.yaml" <<'PYEOF'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
doc = yaml.safe_load(open(src, encoding="utf-8"))
doc["rl_bridge"]["ros__parameters"]["rl_action_size"] = 5   # 词条是 6 条
yaml.safe_dump(doc, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PYEOF
run_expect 1 "(e) rl_action_size != 动作词条数" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL" --config "$WORK/x_actsize.yaml"
expect_msg "rl_action_size" "尺寸不一致提示"

echo "== 5. Transformer rank-3 与 T==history 强制 =="
FIXTURE_H4="$WORK/x_h4.yaml"
FIXTURE_H5="$WORK/x_h5.yaml"
MODEL_SEQ="$WORK/policy_seq.onnx"

cat >"$FIXTURE_H4" <<'YAML'
rl_bridge:
  ros__parameters:
    history_length: 4
    rl_obs_size: 20
    rl_action_size: 6
    observation_terms:
      - index=0 path=/chassis/control_height
      - index=1 path=/chassis/control_velocity take=x
      - index=2 path=/wheel_leg/imu/angular_velocity take=vec3
    action_terms:
      - index=0 output=/wheel_leg/rl/action/lf0
      - index=1 output=/wheel_leg/rl/action/lf1
      - index=2 output=/wheel_leg/rl/action/l_wheel
      - index=3 output=/wheel_leg/rl/action/rf0
      - index=4 output=/wheel_leg/rl/action/rf1
      - index=5 output=/wheel_leg/rl/action/r_wheel
YAML
sed -e 's/history_length: 4/history_length: 5/' \
    -e 's|index=2 path=/wheel_leg/imu/angular_velocity take=vec3|index=2 type=joint_pos joints=lf0,lf1 relative=true|' \
    "$FIXTURE_H4" >"$FIXTURE_H5"

run_expect 0 "gen rank-3 transformer 夹具 (T=4, F=5)" \
    "$PY" "$HERE/gen_synthetic_policy.py" --obs 20 --act 6 \
    --model-type transformer --sequence-length 4 -o "$MODEL_SEQ"
run_expect 0 "盖章 transformer（默认写 rmcs_history_length）" \
    "$PY" "$TOOLS/stamp_layout_metadata.py" --model "$MODEL_SEQ" \
    --from-config "$FIXTURE_H4" --model-type transformer
expect_msg "rmcs_history_length" "盖章输出含序列键"
run_expect 0 "check rank-3 transformer（T==history 正例）" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL_SEQ" --config "$FIXTURE_H4"
expect_msg "model sequence dim == expected history" "T 等值检查已执行"

cp "$MODEL_SEQ" "$WORK/policy_seq_h5.onnx"
run_expect 0 "同一模型按 history=5 配置盖章（模拟窗口错配）" \
    "$PY" "$TOOLS/stamp_layout_metadata.py" --model "$WORK/policy_seq_h5.onnx" \
    --from-config "$FIXTURE_H5" --model-type transformer
run_expect 1 "(f) 模型 T=4 != history=5 → 必须 FAIL" \
    "$PY" "$TOOLS/check_policy_contract.py" "$WORK/policy_seq_h5.onnx" --config "$FIXTURE_H5"
expect_msg "model sequence dim == expected history" "T 不匹配提示"

cp "$MODEL" "$WORK/policy_mlptype.onnx"
run_expect 0 "给 rank-2 模型盖 rmcs_model_type=transformer" \
    "$PY" "$TOOLS/stamp_layout_metadata.py" --model "$WORK/policy_mlptype.onnx" \
    --from-config "$FIXTURE" --model-type transformer
run_expect 1 "(g) model_type=transformer 但输入 rank=2 → 必须 FAIL" \
    "$PY" "$TOOLS/check_policy_contract.py" "$WORK/policy_mlptype.onnx" --config "$FIXTURE"
expect_msg "input rank/model type" "rank/type 不匹配提示"

echo "== 6. 多输入（policy_server extra_inputs 常量）=="
FIXTURE_EXTRA="$WORK/x_extra.yaml"
MODEL_EXTRA="$WORK/policy_extra.onnx"
{
    cat "$FIXTURE"
    cat <<'YAML'

policy_server:
  ros__parameters:
    rl_obs_size: 20
    rl_action_size: 6
    extra_inputs:
      mask: [1, 1, 1, 1]
YAML
} >"$FIXTURE_EXTRA"

run_expect 0 "gen 多输入夹具（obs + mask:4）" \
    "$PY" "$HERE/gen_synthetic_policy.py" --obs 20 --act 6 --extra-input mask:4 -o "$MODEL_EXTRA"
run_expect 0 "盖章多输入夹具" \
    "$PY" "$TOOLS/stamp_layout_metadata.py" --model "$MODEL_EXTRA" --from-config "$FIXTURE_EXTRA"
run_expect 0 "(h) 多输入 + extra_inputs 声明匹配 → PASS" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL_EXTRA" --config "$FIXTURE_EXTRA"
expect_msg "extra input mask" "额外输入校验已执行"

run_expect 1 "(i) 模型有额外输入但 YAML 无 extra_inputs → FAIL" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL_EXTRA" --config "$FIXTURE"
expect_msg "extra_inputs match model" "缺声明提示"

FIXTURE_EXTRA_UNKNOWN="$WORK/x_extra_unknown.yaml"
{
    cat "$FIXTURE"
    cat <<'YAML'

policy_server:
  ros__parameters:
    extra_inputs:
      bogus: [1, 2, 3]
YAML
} >"$FIXTURE_EXTRA_UNKNOWN"
run_expect 1 "(j) YAML 声明了模型没有的额外输入 → FAIL" \
    "$PY" "$TOOLS/check_policy_contract.py" "$MODEL" --config "$FIXTURE_EXTRA_UNKNOWN"
expect_msg "extra_inputs match model" "多余声明提示"

run_expect 0 "typed extra_inputs（int64 + bool）自检 → PASS" \
    "$PY" "$TOOLS/check_policy_contract.py" \
    "$PKG/test/data/typed_extra_input_identity.onnx" --obs 2 --act 2
expect_msg "dtype" "typed extra_inputs 类型校验"

run_expect 0 "仅布局签名解析动态 rank-3 → PASS" \
    "$PY" "$TOOLS/check_policy_contract.py" \
    "$PKG/test/data/seq_layout_dynamic.onnx"
expect_msg "rank-3 layout history" "动态序列由布局解析"

run_expect 1 "模型 T 与布局 history 不符 → FAIL" \
    "$PY" "$TOOLS/check_policy_contract.py" \
    "$PKG/test/data/seq_layout_mismatch.onnx"
expect_msg "metadata history == layout history" "布局与序列 metadata 不一致"

echo
echo "==== 结果：PASS=$PASS FAIL=$FAIL ===="
[ "$FAIL" = 0 ] || exit 1
