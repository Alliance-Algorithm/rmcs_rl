#!/usr/bin/env bash
set -o pipefail   # 注意：不能加 -u —— ROS setup.bash 会引用未定义变量并直接终止非交互 shell

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${TOOL_DIR}/.." && pwd)"
WS_DIR="$(cd "${PKG_DIR}/../.." && pwd)"
CONFIG="${PKG_DIR}/config/bridge_test.yaml"
TMP="/tmp/rmcs_rl_accept"
LOGS="${TMP}/logs"

PASS=0
FAIL=0
EXECUTOR_PID=""
SERVER_PID=""

pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL + 1)); }
info() { printf '\033[36m==>\033[0m %s\n' "$1"; }

cleanup() {
    [[ -n "${SERVER_PID}" ]] && kill -9 "${SERVER_PID}" 2>/dev/null
    [[ -n "${EXECUTOR_PID}" ]] && kill -9 "${EXECUTOR_PID}" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

wait_for_log() {
    local file="$1" pattern="$2" timeout="${3:-15}"
    local attempts=$(( timeout * 5 ))
    for (( i = 0; i < attempts; ++i )); do
        grep -qE "${pattern}" "${file}" 2>/dev/null && return 0
        sleep 0.2
    done
    return 1
}

start_executor() {
    local config="$1" log="$2"
    rm -f "${log}"
    "${WS_DIR}/install/lib/rmcs_executor/rmcs_executor" --ros-args --params-file "${config}" \
        >"${log}" 2>&1 &
    EXECUTOR_PID=$!
    disown "${EXECUTOR_PID}" 2>/dev/null || true
    wait_for_log "${log}" "layout_hash=" 25 || return 1
    return 0
}

start_server() {
    local params="$1" log="$2"
    rm -f "${log}"
    "${WS_DIR}/install/lib/rmcs_rl/policy_server" --ros-args --params-file "${params}" \
        >"${log}" 2>&1 &
    SERVER_PID=$!
    disown "${SERVER_PID}" 2>/dev/null || true
    wait_for_log "${log}" "waiting for obs|startup failed" 25 || return 1
    return 0
}

stop_executor() { [[ -n "${EXECUTOR_PID}" ]] && kill -9 "${EXECUTOR_PID}" 2>/dev/null; sleep 1; EXECUTOR_PID=""; }
stop_server() { [[ -n "${SERVER_PID}" ]] && kill -9 "${SERVER_PID}" 2>/dev/null; sleep 0.3; SERVER_PID=""; }

probe() {
    python3 "${TOOL_DIR}/p0_probe.py" --rl-base /test/rl --timeout "${PROBE_TIMEOUT:-3.0}" \
        --float-topic /test/rl/valid --float-topic /test/rl/healthy \
        --float-topic /test/rl/action_age --float-topic /test/rl/action/j0 2>/dev/null
}

await_valid() {
    local expected="$1" timeout="${2:-10}" attempts
    attempts=$(( timeout * 2 ))
    for (( i = 0; i < attempts; ++i )); do
        JSON="$(probe)"
        if [[ "$(json_get "received.get('/test/rl/valid')" "${JSON}")" == "${expected}" ]]; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

json_get() {
    python3 -c "
import json, sys
data = json.loads(sys.argv[2])
received = data.get('received', {})
try:
    value = eval(sys.argv[1], {'data': data, 'received': received, 'obs': data.get('obs', [])})
except Exception:
    value = None
print(value)
" "$1" "$2"
}

layout_hash_from_config() {
    python3 "${TOOL_DIR}/check_policy_contract.py" --print-layout --config "$1" \
        --node rl_bridge 2>/dev/null | grep -oE '[0-9a-f]{16}' | head -1
}

bridge_hash_from_log() { grep -oE 'layout_hash=[0-9a-f]{16}' "$1" | head -1 | cut -d= -f2; }

mkdir -p "${LOGS}"
rm -f "${LOGS}"/*
source "${WS_DIR}/install/setup.bash" >/dev/null 2>&1 || {
    echo "install/setup.bash not found: build the workspace first (colcon build)"
    exit 1
}

info "0. 生成合成策略 + 盖章 metadata"
python3 "${TOOL_DIR}/gen_synthetic_policy.py" --from-config "${CONFIG}" --node rl_bridge \
    -o "${TMP}/bridge_test.onnx" >"${LOGS}/gen.log" 2>&1 || {
    echo "gen_synthetic_policy.py failed:"; cat "${LOGS}/gen.log"; exit 1
}
python3 "${TOOL_DIR}/stamp_layout_metadata.py" --model "${TMP}/bridge_test.onnx" \
    --from-config "${CONFIG}" --node rl_bridge >"${LOGS}/stamp.log" 2>&1 || {
    echo "stamp_layout_metadata.py failed:"; cat "${LOGS}/stamp.log"; exit 1
}
python3 "${TOOL_DIR}/check_policy_contract.py" "${TMP}/bridge_test.onnx" --config "${CONFIG}" \
    --node rl_bridge >"${LOGS}/contract.log" 2>&1
if [[ $? -eq 0 ]]; then pass "合同校验（模型 metadata ↔ YAML 词条）通过"; else
    fail "合同校验未通过："; cat "${LOGS}/contract.log"; fi

EXPECTED_HASH="$(layout_hash_from_config "${CONFIG}")"
[[ -n "${EXPECTED_HASH}" ]] && pass "YAML 侧 layout_hash=${EXPECTED_HASH}" \
    || fail "无法从 YAML 计算 layout_hash"

info "1. 启动 executor（合成观测源 + RlBridge）与 policy_server"
start_executor "${CONFIG}" "${LOGS}/executor_1.log" || fail "executor 未在 25s 内报告 layout_hash"
BRIDGE_HASH="$(bridge_hash_from_log "${LOGS}/executor_1.log")"
if [[ -n "${BRIDGE_HASH}" && "${BRIDGE_HASH}" == "${EXPECTED_HASH}" ]]; then
    pass "桥侧 layout_hash 与 YAML 侧一致（${BRIDGE_HASH}）"
else
    fail "桥侧 layout_hash='${BRIDGE_HASH}' 与 YAML 侧 '${EXPECTED_HASH}' 不一致"
fi
if grep -q "policy loaded" "${LOGS}/executor_1.log" 2>/dev/null; then
    fail "桥进程不应加载策略模型（ONNX 属于 policy_server）"
else
    pass "桥进程不加载 ONNX（控制进程无 ORT 依赖）"
fi

cat >"${TMP}/policy_server.yaml" <<EOF
policy_server:
  ros__parameters:
    rl_base: "/test/rl"
    rl_model_path: "${TMP}/bridge_test.onnx"
    publish_status: true
    status_rate: 2.0
EOF
start_server "${TMP}/policy_server.yaml" "${LOGS}/server_1.log" || fail "policy_server 未就绪"
if grep -q "layout_hash   : ${EXPECTED_HASH}" "${LOGS}/server_1.log"; then
    pass "策略进程由模型 metadata 算出同一 layout_hash"
else
    fail "策略进程 layout_hash 与桥不一致（见 ${LOGS}/server_1.log）"
fi

info "2. obs → action 链路"
await_valid 1.0 15 || true
OBS_LEN="$(json_get "data.get('obs_len')" "${JSON}")"
VALID="$(json_get "received.get('/test/rl/valid')" "${JSON}")"
AGE="$(json_get "received.get('/test/rl/action_age')" "${JSON}")"
J0="$(json_get "received.get('/test/rl/action/j0')" "${JSON}")"
HEIGHT_OBS="$(json_get "obs[2]" "${JSON}")"
ANGVEL_OBS="$(json_get "obs[3]" "${JSON}")"
[[ "${OBS_LEN}" == "17" ]] && pass "obs 维度 = 17" || fail "obs 维度 = ${OBS_LEN}（期望 17）"
[[ "${VALID}" == "1.0" ]] && pass "valid = 1" || fail "valid = ${VALID}（期望 1）"
python3 -c "
import math, sys
age = float('${AGE}')
sys.exit(0 if math.isfinite(age) and age <= 0.04 else 1)" \
    && pass "action_age=${AGE} s ≤ max_action_age" || fail "action_age=${AGE} 超出上限"
python3 -c "
import math, sys
value = float('${J0}')
sys.exit(0 if math.isfinite(value) else 1)" \
    && pass "动作接口写出有限值（j0=${J0}）" || fail "动作未写出有限值（j0=${J0}）"
python3 -c "
import sys
value = float('${HEIGHT_OBS}')
sys.exit(0 if 0.19 <= value <= 0.31 else 1)" \
    && pass "obs[2] 高度指令落在 [0.20,0.30]（实测 ${HEIGHT_OBS}）" \
    || fail "obs[2]=${HEIGHT_OBS} 不在高度指令范围内"
python3 -c "
import sys
value = float('${ANGVEL_OBS}')
sys.exit(0 if abs(value) <= 0.25 else 1)" \
    && pass "obs[3] = 角速度×0.5（实测 ${ANGVEL_OBS}）" \
    || fail "obs[3]=${ANGVEL_OBS} 超出 ×0.5 后的量程"

info "3. 杀掉策略进程 → 失效语义（1 s 内 valid=0 且动作 NaN）"
stop_server
await_valid 0.0 8 || true
VALID="$(json_get "received.get('/test/rl/valid')" "${JSON}")"
HEALTHY="$(json_get "received.get('/test/rl/healthy')" "${JSON}")"
J0="$(json_get "received.get('/test/rl/action/j0')" "${JSON}")"
[[ "${VALID}" == "0.0" ]] && pass "valid 已回落到 0" || fail "valid=${VALID}（期望 0）"
[[ "${HEALTHY}" == "0.0" ]] && pass "healthy 已回落到 0" || fail "healthy=${HEALTHY}（期望 0）"
python3 -c "
import math, sys
sys.exit(0 if math.isnan(float('${J0}')) else 1)" \
    && pass "动作接口写出 NaN（不保持旧动作）" || fail "动作=${J0}（期望 NaN）"

info "4. metadata 缺失 → 策略进程启动即失败"
python3 "${TOOL_DIR}/gen_synthetic_policy.py" --from-config "${CONFIG}" --node rl_bridge \
    -o "${TMP}/no_metadata.onnx" >/dev/null 2>&1
sed "s|${TMP}/bridge_test.onnx|${TMP}/no_metadata.onnx|" "${TMP}/policy_server.yaml" \
    >"${TMP}/policy_server_no_meta.yaml"
start_server "${TMP}/policy_server_no_meta.yaml" "${LOGS}/server_nometa.log"
if grep -q "rmcs_obs_layout" "${LOGS}/server_nometa.log"; then
    pass "缺少 rmcs_obs_layout 时报错清晰"
else
    fail "缺少 metadata 未给出可读报错（见 ${LOGS}/server_nometa.log）"
fi
stop_server

info "5. 词条改一个 scale → 指纹不符 → 策略进程拒答、桥 valid=0"
sed 's|path=/test/command/height|path=/test/command/height scale=5.0|' "${CONFIG}" \
    >"${TMP}/bridge_test_mismatch.yaml"
stop_executor
start_executor "${TMP}/bridge_test_mismatch.yaml" "${LOGS}/executor_2.log" \
    || fail "executor（错配配置）未就绪"
MISMATCH_HASH="$(bridge_hash_from_log "${LOGS}/executor_2.log")"
[[ "${MISMATCH_HASH}" != "${EXPECTED_HASH}" ]] \
    && pass "改动 scale 后指纹变化（${MISMATCH_HASH}）" || fail "改动 scale 后指纹未变化"
start_server "${TMP}/policy_server.yaml" "${LOGS}/server_mismatch.log"
sleep 1.5
if grep -q "layout_hash mismatch" "${LOGS}/server_mismatch.log"; then
    pass "策略进程检测到指纹不符并拒答"
else
    fail "策略进程未检测到指纹不符（见 ${LOGS}/server_mismatch.log）"
fi
await_valid 0.0 8 || true
VALID="$(json_get "received.get('/test/rl/valid')" "${JSON}")"
[[ "${VALID}" == "0.0" ]] && pass "桥 valid=0（不会用错合同的动作驱动电机）" \
    || fail "桥 valid=${VALID}（期望 0）"
stop_server
stop_executor

info "6. 指纹去类型化：同一路径换 C++ 类型（DirectionVector → Vector3d）"
sed 's|velocity_interface_type: "direction_vector"|velocity_interface_type: "vector3"|' "${CONFIG}" \
    >"${TMP}/bridge_test_vector3.yaml"
start_executor "${TMP}/bridge_test_vector3.yaml" "${LOGS}/executor_3.log" \
    || fail "executor（vector3 变体）未就绪"
VECTOR3_HASH="$(bridge_hash_from_log "${LOGS}/executor_3.log")"
[[ "${VECTOR3_HASH}" == "${EXPECTED_HASH}" ]] \
    && pass "换接口类型后 layout_hash 不变（${VECTOR3_HASH}）" \
    || fail "换接口类型后 layout_hash 变了：${VECTOR3_HASH} != ${EXPECTED_HASH}"
start_server "${TMP}/policy_server.yaml" "${LOGS}/server_3.log"
await_valid 1.0 15 || true
VALID="$(json_get "received.get('/test/rl/valid')" "${JSON}")"
[[ "${VALID}" == "1.0" ]] && pass "vector3 变体下链路仍然 valid=1" \
    || fail "vector3 变体下 valid=${VALID}（期望 1）"

info "7. 启动级校验：词条错位 / 尺寸不符 → 启动即失败（不进入运行期）"
stop_executor
sed 's|index=3  path=/test/imu/angular_velocity|index=4  path=/test/imu/angular_velocity|' \
    "${CONFIG}" >"${TMP}/bridge_test_badindex.yaml"
timeout 12 "${WS_DIR}/install/lib/rmcs_executor/rmcs_executor" --ros-args \
    --params-file "${TMP}/bridge_test_badindex.yaml" >"${LOGS}/executor_badindex.log" 2>&1
grep -q "declares index=4 but the running offset is 3" "${LOGS}/executor_badindex.log" \
    && pass "显式 index= 与推导偏移不符 → 启动失败并指明位置" \
    || fail "错位词条未报错（见 ${LOGS}/executor_badindex.log）"

sed 's|rl_obs_size: 17|rl_obs_size: 18|' "${CONFIG}" >"${TMP}/bridge_test_badsize.yaml"
timeout 12 "${WS_DIR}/install/lib/rmcs_executor/rmcs_executor" --ros-args \
    --params-file "${TMP}/bridge_test_badsize.yaml" >"${LOGS}/executor_badsize.log" 2>&1
grep -q "rl_obs_size=18 but observation_terms sum to 17" "${LOGS}/executor_badsize.log" \
    && pass "rl_obs_size 与词条维度之和不符 → 启动失败" \
    || fail "尺寸不符未报错（见 ${LOGS}/executor_badsize.log）"

python3 - "${CONFIG}" "${TMP}/bridge_test_dupaction.yaml" <<'FIXTURE'
import sys
# 让 #3 也指向 #0 的接口：接口重名必须在启动期被拒绝
text = open(sys.argv[1], encoding="utf-8").read()
open(sys.argv[2], "w", encoding="utf-8").write(
    text.replace("output=/test/rl/action/j3", "output=/test/rl/action/j0"))
FIXTURE
timeout 12 "${WS_DIR}/install/lib/rmcs_executor/rmcs_executor" --ros-args \
    --params-file "${TMP}/bridge_test_dupaction.yaml" >"${LOGS}/executor_dupaction.log" 2>&1
grep -qE "two action terms map to the same output interface" "${LOGS}/executor_dupaction.log" \
    && pass "两个动作槽位指向同一接口 → 启动失败" \
    || fail "动作接口重复未报错（见 ${LOGS}/executor_dupaction.log）"

echo
info "结果：PASS=${PASS} FAIL=${FAIL}（日志目录 ${LOGS}）"
[[ "${FAIL}" -eq 0 ]]
