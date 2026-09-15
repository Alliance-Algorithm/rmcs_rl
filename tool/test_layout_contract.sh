#!/usr/bin/env bash
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python3}"
WORK="${TMPDIR:-/tmp}"
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
run_expect 0 "rl_layout self-test" "$PY" "$HERE/rl_layout.py"
expect_msg "rl_layout self-test OK" "self-test OK"

echo "== 2. 生成 → 盖章 → 校验（正例必须 PASS）=="
run_expect 0 "gen_synthetic_policy --from-config" \
    "$PY" "$HERE/gen_synthetic_policy.py" --from-config "$FIXTURE" -o "$MODEL"
run_expect 0 "stamp_layout_metadata" \
    "$PY" "$HERE/stamp_layout_metadata.py" --model "$MODEL" --from-config "$FIXTURE"
cat "$LOG"
SHA_BEFORE="$(sha256sum "$MODEL" | cut -d' ' -f1)"
run_expect 0 "stamp_layout_metadata（幂等重跑）" \
    "$PY" "$HERE/stamp_layout_metadata.py" --model "$MODEL" --from-config "$FIXTURE"
SHA_AFTER="$(sha256sum "$MODEL" | cut -d' ' -f1)"
if [ "$SHA_BEFORE" = "$SHA_AFTER" ]; then
    ok "盖章幂等（文件字节不变）"
else
    bad "盖章非幂等：$SHA_BEFORE != $SHA_AFTER"
fi
run_expect 0 "check_policy_contract（正例）" \
    "$PY" "$HERE/check_policy_contract.py" "$MODEL" --config "$FIXTURE"
cat "$LOG"
expect_msg "== SUMMARY: PASS ==" "SUMMARY PASS"

echo "== 3. 逃生口 type= 不改变布局指纹 =="
SIG_PLAIN="$("$PY" "$HERE/check_policy_contract.py" --print-layout --config "$FIXTURE" \
    | grep -E 'obs_signature|action_signature|layout_hash')"
SIG_TYPES="$("$PY" "$HERE/check_policy_contract.py" --print-layout --config "$FIXTURE_TYPES" \
    | grep -E 'obs_signature|action_signature|layout_hash')"
if [ -n "$SIG_PLAIN" ] && [ "$SIG_PLAIN" = "$SIG_TYPES" ]; then
    ok "带/不带 type= 的规范串与 layout_hash 完全一致"
else
    bad "type= 改变了布局指纹"
    diff <(echo "$SIG_PLAIN") <(echo "$SIG_TYPES") | sed 's/^/    | /'
fi
echo "$SIG_PLAIN" | sed 's/^/    /'

echo "== 4. 与桥侧验收夹具集成（config/bridge_test.yaml，存在才跑）=="
BRIDGE_FIXTURE="$PKG/config/bridge_test.yaml"
if [ -f "$BRIDGE_FIXTURE" ]; then
    BRIDGE_MODEL="$WORK/bridge_test.onnx"
    run_expect 0 "gen --from-config（桥夹具）" \
        "$PY" "$HERE/gen_synthetic_policy.py" --from-config "$BRIDGE_FIXTURE" --node rl_bridge \
        -o "$BRIDGE_MODEL"
    run_expect 0 "stamp（桥夹具）" \
        "$PY" "$HERE/stamp_layout_metadata.py" --model "$BRIDGE_MODEL" \
        --from-config "$BRIDGE_FIXTURE" --node rl_bridge
    run_expect 0 "check（桥夹具）" \
        "$PY" "$HERE/check_policy_contract.py" "$BRIDGE_MODEL" \
        --config "$BRIDGE_FIXTURE" --node rl_bridge
    BRIDGE_HASH="$("$PY" "$HERE/check_policy_contract.py" --print-layout \
        --config "$BRIDGE_FIXTURE" --node rl_bridge 2>/dev/null \
        | grep -oE '[0-9a-f]{16}' | head -1)"
    if [ -n "$BRIDGE_HASH" ]; then
        ok "p0_acceptance.sh 解析方式取到 layout_hash=$BRIDGE_HASH"
    else
        bad "--print-layout 首行未给出可被 grep 提取的 layout_hash"
    fi
else
    echo "  [SKIP] $BRIDGE_FIXTURE 不存在（桥侧夹具未提供）"
fi

echo "== 5. 负例（必须 FAIL，exit 1，消息可读）=="
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
    "$PY" "$HERE/check_policy_contract.py" "$WORK/policy_neg_a.onnx" --config "$FIXTURE"
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
    "$PY" "$HERE/check_policy_contract.py" "$MODEL" --config "$WORK/x_swapped.yaml"
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
    "$PY" "$HERE/check_policy_contract.py" "$MODEL" --config "$WORK/x_index_clash.yaml"
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
    "$PY" "$HERE/check_policy_contract.py" "$MODEL" --config "$WORK/x_scale.yaml"
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
    "$PY" "$HERE/check_policy_contract.py" "$WORK/policy_neg_d.onnx" --config "$FIXTURE"
expect_msg "policy_layout_hash" "hash 校验失败提示"

"$PY" - "$FIXTURE" "$WORK/x_actsize.yaml" <<'PYEOF'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
doc = yaml.safe_load(open(src, encoding="utf-8"))
doc["rl_bridge"]["ros__parameters"]["rl_action_size"] = 5   # 词条是 6 条
yaml.safe_dump(doc, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PYEOF
run_expect 1 "(e) rl_action_size != 动作词条数" \
    "$PY" "$HERE/check_policy_contract.py" "$MODEL" --config "$WORK/x_actsize.yaml"
expect_msg "rl_action_size" "尺寸不一致提示"

echo
echo "==== 结果：PASS=$PASS FAIL=$FAIL ===="
[ "$FAIL" = 0 ] || exit 1
