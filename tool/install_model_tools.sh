#!/usr/bin/env bash
# 安装 ONNX 迭代工具链（仅开发/部署用）：
#   在容器里建一个固定 venv，并装 onnx + pyyaml，供 stamp_model.sh / check_policy_contract 使用。
#   运行期 policy_server 只需要 libonnxruntime.so（由 install_rl_deps.sh 负责），与本工具隔离。
#
# 用法:
#   bash install_model_tools.sh
# 可选环境变量:
#   RMCS_TOOLS_VENV   工具 venv 目录（默认 /opt/rmcs-tools/venv）
#   RMCS_ONNX_PYTHON  若已指向一个带 onnx+yaml 的 python，则直接跳过安装
set -euo pipefail

VENV_DIR="${RMCS_TOOLS_VENV:-/opt/rmcs-tools/venv}"

log() { printf '[install_model_tools] %s\n' "$*"; }
die() { printf '[install_model_tools] ERROR: %s\n' "$*" >&2; exit 1; }

run_privileged() {
    if command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        "$@"
    fi
}

# 已有可用的 onnx python → 跳过
if [[ -n "${RMCS_ONNX_PYTHON:-}" ]] \
    && "$RMCS_ONNX_PYTHON" -c "import onnx, yaml" >/dev/null 2>&1; then
    log "RMCS_ONNX_PYTHON 已可用，跳过安装: $RMCS_ONNX_PYTHON"
    exit 0
fi

command -v python3 >/dev/null 2>&1 || die "找不到 python3"

# 建 venv；若 python3-venv 缺失则先安装它
if ! { [[ -x "$VENV_DIR/bin/python" ]] \
        && "$VENV_DIR/bin/python" -c "import onnx, yaml" >/dev/null 2>&1; }; then
    if ! run_privileged python3 -m venv "$VENV_DIR" >/dev/null 2>&1 \
        || ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        log "python3 venv 不可用，安装 python3-venv ..."
        run_privileged apt-get update
        run_privileged apt-get install -y python3-venv
        run_privileged rm -rf "$VENV_DIR"
        run_privileged python3 -m venv "$VENV_DIR" || die "创建 venv 失败: $VENV_DIR"
    fi

    log "安装 onnx + pyyaml 到 $VENV_DIR ..."
    run_privileged "$VENV_DIR/bin/python" -m pip install --upgrade pip
    run_privileged "$VENV_DIR/bin/python" -m pip install --upgrade onnx pyyaml
fi

"$VENV_DIR/bin/python" -c "import onnx, yaml; print('[install_model_tools] onnx', onnx.__version__)" \
    || die "onnx 安装校验失败"

cat <<EOF

[install_model_tools] 完成。
  工具 venv : $VENV_DIR
  python    : $VENV_DIR/bin/python

stamp_model.sh 会自动探测该 venv；如需手动指定：
  export RMCS_ONNX_PYTHON=$VENV_DIR/bin/python
EOF
