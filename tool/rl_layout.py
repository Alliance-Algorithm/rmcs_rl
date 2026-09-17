#!/usr/bin/env python3
"""rmcs_rl 布局单一真源：词条语法 → 规范串(v2) → layout_hash / model_id。

C++ 桥（rl_bridge）与策略进程各自实现同一份规范（doc/bridge-design.md §6.2/§6.3），
Python 侧的真源就是本模块：stamp_layout_metadata.py / check_policy_contract.py /
gen_synthetic_policy.py 全部从这里取语法与 hash，避免同一份契约靠人写两遍（drift）。

观测词条（YAML observation_terms，每个元素是一个 "key=value ..." 字符串）：
  path / take / transform / type / index / scale / clip / default / name
  关节词条另有 joints / relative / zero；last_action 有 indices；constant 有 value。

dim 与 id 只由 take / transform / 词条类型决定，**绝不**由 C++ 接口类型决定：

  | 词条形式                                          | dim        | id                      |
  | path=P（无 take/transform）                       | 1          | P                       |
  | path=P take=x（y/z 同理）                         | 1          | vec3c:P:x               |
  | path=P take=vec3（vec/all/vector 为别名）         | 3          | vec3:P                  |
  | path=P transform=projected_gravity（take=gravity）| 3          | gravity:P               |
  | type=joint_pos joints=a,b relative/zero/abs       | len(joints)| joint_pos:rel/zero/abs:a+b |
  | type=joint_vel joints=a,b                         | len(joints)| joint_vel:abs:a+b       |
  | type=joint_torque joints=a,b                      | len(joints)| joint_torque:abs:a+b    |
  | type=last_action [indices=0,1,2 或 0..5]          | count      | last_action:0+1+2       |
  | type=constant value=1,0,0.5                       | count      | constant:1+0+0.5        |
  | 任意词条 + name=ID                                | 不变       | ID（原样覆盖）           |

  type=scalar|vector3|quaternion|direction_vector（别名 double/bool/int/size/size_t/vec3，
  与 rl_bridge.cpp 的 Binding 表一致）只是「钉住 C++ 接口类型」的逃生口
  （自省失败时才写），**不改变 dim 与 id** —— 这正是 v2 去类型化的目的：同一个控制量
  在仿真里是 Vector3d、真机上是 DirectionVector，策略不该因此换 fingerprint。
  type=quaternion 必须配 transform=projected_gravity。

规范串（v2）：
  obs_signature    := "v2" ( "|" entry )*      entry := <id> [ "*" <scale> ] "@" <dim>
  action_signature := "v2" ( "|" entry )*      entry := "#" <index> ":" <name> [ "*" <scale> ]
  <scale> 仅在 != 1.0 时追加，格式 '%.6g'；clip/default 等部署侧参数不进指纹。

  layout_hash = FNV1a64((obs_sig + "||" + act_sig + "||" + str(obs_size) + "x" + str(act_size)))
  model_id    = FNV1a64(模型文件全字节)

命令行（自检，不依赖 onnx/yaml）：
  python3 rl_layout.py     # 断言上方语法表 + FNV 稳定性，打印 "rl_layout self-test OK"
"""
from typing import Dict, List, Optional, Sequence, Tuple


FNV1A64_OFFSET_BASIS = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF

SIGNATURE_VERSION = "v2"
DEFAULT_NODE = "rl_bridge"

OBS_KEYS = frozenset({
    "path", "take", "transform", "type", "index", "scale", "clip", "default",
    "name", "joints", "relative", "zero", "indices", "value",
})
ACTION_KEYS = frozenset({"index", "output", "name", "scale", "clip"})

SCALAR_INTERFACE_TYPES = ("scalar", "double", "bool", "int", "size", "size_t")
VECTOR_INTERFACE_TYPES = ("vector3", "vec3", "direction_vector")
INTERFACE_TYPES = SCALAR_INTERFACE_TYPES + VECTOR_INTERFACE_TYPES + ("quaternion",)
TERM_TYPES = ("joint_pos", "joint_vel", "joint_torque", "last_action", "constant")
JOINT_TYPES = ("joint_pos", "joint_vel", "joint_torque")

TAKE_COMPONENTS = ("x", "y", "z")
TAKE_VEC3_ALIASES = ("vec3", "vec", "all", "vector")
GRAVITY_TRANSFORMS = ("projected_gravity", "gravity")

V1_ID_MARKERS = ("vector3_component:", "direction_vector_component:")

_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


class LayoutError(ValueError):
    """词条 / 配置不合法。消息面向用户，可直接打印。"""



def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise LayoutError("需要 PyYAML 才能读部署 YAML（请用容器内 python3 运行）") from exc
    return yaml


def _import_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise LayoutError("需要 onnx 才能读/写模型 metadata（请用容器内 python3 运行）") from exc
    return onnx


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64 位哈希（非密码学）。"""
    h = FNV1A64_OFFSET_BASIS
    for byte in data:
        h ^= byte
        h = (h * FNV1A64_PRIME) & MASK64
    return h


def hex16(value: int) -> str:
    """uint64 → 16 位小写十六进制（带 0x 前缀，零填充）。"""
    return "0x%016x" % (int(value) & MASK64)


def format_scale(scale: float) -> str:
    """规范串里的 scale 文本：'%.6g'（1.0 不写）。"""
    return "%.6g" % float(scale)


def _parse_float(text: str, key: str, spec: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise LayoutError(f"{key} 不是合法浮点数（{key}={text}）：{spec!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise LayoutError(f"{key} 必须是有限数（{key}={text}）：{spec!r}")
    return value


def _parse_bool(text: str, key: str, spec: str) -> bool:
    low = text.strip().lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    raise LayoutError(f"{key} 必须是 true/false（{key}={text}）：{spec!r}")


def _parse_float_list(text: str, key: str, spec: str) -> List[float]:
    items = [t.strip() for t in text.split(",")]
    if not items or any(t == "" for t in items):
        raise LayoutError(f"{key} 列表含空项（{key}={text}）：{spec!r}")
    return [_parse_float(t, key, spec) for t in items]


def _split_tokens(spec: str) -> Dict[str, str]:
    """把一条词条串拆成 key=value（重复键 / 缺等号即报错）。"""
    if spec is None:
        raise LayoutError("词条为空（None）")
    if not isinstance(spec, str):
        raise LayoutError(f"词条必须是字符串（YAML 词条 {spec!r}）")
    tokens = spec.split()
    if not tokens:
        raise LayoutError("空词条（只有空白字符）")
    kv: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            raise LayoutError(f"非法 token {token!r}（应为 key=value）：{spec!r}")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise LayoutError(f"token 缺键名 {token!r}：{spec!r}")
        if key in kv:
            raise LayoutError(f"重复键 {key}=：{spec!r}")
        kv[key] = value
    return kv


def _check_keys(kv: Dict[str, str], allowed, spec: str) -> None:
    unknown = sorted(set(kv) - set(allowed))
    if unknown:
        raise LayoutError(
            f"未知键 {unknown}（可用：{sorted(allowed)}）：{spec!r}"
        )


def _parse_joints(text: str, spec: str) -> List[str]:
    joints = [j.strip() for j in text.split(",")]
    if not joints or any(j == "" for j in joints):
        raise LayoutError(f"joints 列表为空或含空项（joints={text}）：{spec!r}")
    return joints


def _parse_indices(value: str, count: int, spec: str) -> List[int]:
    """indices=0,1,2 或 indices=0..5（闭区间）。count = 允许的上界（action_size）。"""
    indices: List[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw == "":
            raise LayoutError(f"indices 列表含空项（indices={value}）：{spec!r}")
        if ".." in raw:
            lo_text, _, hi_text = raw.partition("..")
            try:
                lo, hi = int(lo_text), int(hi_text)
            except ValueError:
                raise LayoutError(f"indices 区间不是整数（indices={value}）：{spec!r}")
            if hi < lo:
                raise LayoutError(f"indices 区间倒置（{raw}）：{spec!r}")
            indices.extend(range(lo, hi + 1))
        else:
            try:
                indices.append(int(raw))
            except ValueError:
                raise LayoutError(f"indices 含非整数项（{raw}）：{spec!r}")
    if not indices:
        raise LayoutError(f"indices 为空：{spec!r}")
    if len(set(indices)) != len(indices):
        raise LayoutError(f"indices 有重复项（indices={value}）：{spec!r}")
    for index in indices:
        if index < 0 or index >= count:
            raise LayoutError(
                f"indices 越界 {index}（合法范围 0..{count - 1}）：{spec!r}"
            )
    return indices


def _parse_clip(text: str, spec: str):
    """clip=0.5（对称限幅）或 clip=min:max（桥侧 parse_clip_ 要求的形式）。

    clip 不进 layout 指纹（部署侧参数），两种写法都接受，避免桥合法 YAML 在工具链误报。
    """
    if ":" in text:
        lo_text, _, hi_text = text.partition(":")
        low = _parse_float(lo_text, "clip min", spec)
        high = _parse_float(hi_text, "clip max", spec)
        if low > high:
            raise LayoutError(f"clip 区间倒置（clip={text}）：{spec!r}")
        return (low, high)
    value = _parse_float(text, "clip", spec)
    if value <= 0.0:
        raise LayoutError(f"clip 必须为正数（clip={text}）：{spec!r}")
    return value


def _scale_clip_default(kv: Dict[str, str], spec: str) -> Tuple[float, object, Optional[float]]:
    scale = 1.0
    clip = None
    default = None
    if "scale" in kv:
        scale = _parse_float(kv["scale"], "scale", spec)
        if scale == 0.0:
            raise LayoutError(f"scale 不能为 0（物理/单位变换必须有量纲）：{spec!r}")
    if "clip" in kv:
        clip = _parse_clip(kv["clip"], spec)
    if "default" in kv:
        default = _parse_float(kv["default"], "default", spec)
    return scale, clip, default


def _obs_term(kind: str, term_id: str, dim: int, spec: str, kv: Dict[str, str],
              path=None, take=None, joints=None, relative=False, zero=False,
              indices=None, constants=None) -> Dict:
    """组装 parse_obs_term 的返回值 —— 键集合固定，任何一侧新增都改这里。"""
    scale, clip, default = _scale_clip_default(kv, spec)
    name = kv.get("name")
    if name is not None:
        if name == "":
            raise LayoutError(f"name= 不能为空：{spec!r}")
        term_id = name
    return {
        "kind": kind,
        "path": path,
        "take": take,
        "joints": joints,
        "relative": relative,
        "zero": zero,
        "indices": indices,
        "constants": constants,
        "scale": scale,
        "clip": clip,
        "default": default,
        "name": name,
        "id": term_id,
        "dim": int(dim),
    }



def parse_obs_term(spec: str, action_size: int) -> Dict:
    """解析一条观测词条，返回 {kind,path,take,joints,relative,zero,indices,
    constants,scale,clip,default,name,id,dim}。

    action_size 仅用于 type=last_action 的缺省 indices 与越界检查。
    """
    kv = _split_tokens(spec)
    _check_keys(kv, OBS_KEYS, spec)

    term_type = kv.get("type")
    term_type = term_type.strip().lower() if term_type is not None else None
    if term_type is not None and term_type not in INTERFACE_TYPES + TERM_TYPES:
        raise LayoutError(
            f"未知 type={term_type}（可用：{'/'.join(INTERFACE_TYPES + TERM_TYPES)}）：{spec!r}"
        )

    if term_type in TERM_TYPES:
        for key in ("path", "take", "transform"):
            if key in kv:
                raise LayoutError(f"type={term_type} 不接受 {key}=（关节/常量词条无接口路径）：{spec!r}")
        if term_type in JOINT_TYPES:
            if "indices" in kv or "value" in kv:
                raise LayoutError(f"type={term_type} 不接受 indices=/value=：{spec!r}")
            if "joints" not in kv:
                raise LayoutError(f"type={term_type} 缺少 joints=（训练 DOF 序的关节名列表）：{spec!r}")
            joints = _parse_joints(kv["joints"], spec)
            relative = zero = False
            if term_type == "joint_pos":
                if "relative" in kv:
                    relative = _parse_bool(kv["relative"], "relative", spec)
                if "zero" in kv:
                    zero = _parse_bool(kv["zero"], "zero", spec)
                if relative and zero:
                    raise LayoutError(
                        f"relative=true 与 zero=true 互斥（二者都是「减参考位」的不同参考）：{spec!r}"
                    )
                mode = "rel" if relative else ("zero" if zero else "abs")
            else:
                if "relative" in kv or "zero" in kv:
                    raise LayoutError(f"type={term_type} 不支持 relative=/zero=（恒为绝对量）：{spec!r}")
                mode = "abs"
            term_id = f"{term_type}:{mode}:{'+'.join(joints)}"
            return _obs_term(term_type, term_id, len(joints), spec, kv,
                             joints=joints, relative=relative, zero=zero)

        if term_type == "last_action":
            for key in ("joints", "value"):
                if key in kv:
                    raise LayoutError(f"type=last_action 不接受 {key}=：{spec!r}")
            if "indices" in kv:
                indices = _parse_indices(kv["indices"], action_size, spec)
            else:
                if action_size <= 0:
                    raise LayoutError(
                        f"type=last_action 无法推导维度（action_size={action_size}），"
                        f"请显式写 indices=：{spec!r}"
                    )
                indices = list(range(action_size))
            term_id = "last_action:" + "+".join(str(i) for i in indices)
            return _obs_term("last_action", term_id, len(indices), spec, kv, indices=indices)

        for key in ("joints", "indices"):
            if key in kv:
                raise LayoutError(f"type=constant 不接受 {key}=：{spec!r}")
        if "value" not in kv:
            raise LayoutError(f"type=constant 缺少 value=（逗号分隔常数，如 value=1,0,0.5）：{spec!r}")
        constants = _parse_float_list(kv["value"], "value", spec)
        term_id = "constant:" + "+".join(format_scale(v) for v in constants)
        return _obs_term("constant", term_id, len(constants), spec, kv, constants=constants)

    if "path" not in kv:
        raise LayoutError(
            f"观测词条缺少 path=（或写 type=joint_*/last_action/constant）：{spec!r}"
        )
    path = kv["path"]
    if path == "":
        raise LayoutError(f"path= 不能为空：{spec!r}")
    for key in ("joints", "relative", "zero", "indices", "value"):
        if key in kv:
            raise LayoutError(f"路径词条不接受 {key}=（该键属于关节/last_action/constant 词条）：{spec!r}")

    take = kv.get("take")
    take = take.strip().lower() if take is not None else None
    transform = kv.get("transform")
    transform = transform.strip().lower() if transform is not None else None

    if transform is not None and take is not None:
        raise LayoutError(f"take= 与 transform= 互斥（取值语义只能有一种）：{spec!r}")

    if transform is not None:
        if transform not in GRAVITY_TRANSFORMS:
            raise LayoutError(
                f"未知 transform={transform}（可用：{'/'.join(GRAVITY_TRANSFORMS)}）：{spec!r}"
            )
        if term_type in SCALAR_INTERFACE_TYPES + VECTOR_INTERFACE_TYPES:
            raise LayoutError(
                f"transform=projected_gravity 需要四元数接口（type=quaternion 或省 type= 由自省决定），"
                f"与 type={term_type} 矛盾：{spec!r}"
            )
        return _obs_term("gravity", f"gravity:{path}", 3, spec, kv, path=path)

    if term_type == "quaternion":
        raise LayoutError(
            f"type=quaternion 必须配 transform=projected_gravity（四元数不给策略，只给重力投影）：{spec!r}"
        )

    if take is not None:
        if take in GRAVITY_TRANSFORMS:
            if term_type in SCALAR_INTERFACE_TYPES + VECTOR_INTERFACE_TYPES:
                raise LayoutError(
                    f"take=gravity 需要四元数接口（type=quaternion 或省 type= 由自省决定），"
                    f"与 type={term_type} 矛盾：{spec!r}"
                )
            return _obs_term("gravity", f"gravity:{path}", 3, spec, kv, path=path, take=take)
        if term_type == "quaternion":
            raise LayoutError(
                f"type=quaternion 只支持 transform=projected_gravity / take=gravity：{spec!r}"
            )
        if take in TAKE_COMPONENTS:
            return _obs_term("component", f"vec3c:{path}:{take}", 1, spec, kv, path=path, take=take)
        if take in TAKE_VEC3_ALIASES:
            return _obs_term("vec3", f"vec3:{path}", 3, spec, kv, path=path, take=take)
        raise LayoutError(
            f"未知 take={take}"
            f"（可用：{'/'.join(TAKE_COMPONENTS + TAKE_VEC3_ALIASES + ('gravity',))}）：{spec!r}"
        )

    return _obs_term("scalar", path, 1, spec, kv, path=path)


def obs_term_interface(spec: str) -> Optional[str]:
    """逃生口 type= 的声明值（scalar/vector3/quaternion/direction_vector）；没写返回 None。

    单独提供：parse_obs_term 的返回键集合是冻结契约，不塞入额外键。
    """
    kv = _split_tokens(spec)
    term_type = kv.get("type")
    if term_type is None:
        return None
    term_type = term_type.strip().lower()
    return term_type if term_type in INTERFACE_TYPES else None


def _obs_entry(term: Dict) -> str:
    """<id> [ "*" <scale> ] "@" <dim>。"""
    entry = str(term["id"])
    if float(term["scale"]) != 1.0:
        entry += "*" + format_scale(term["scale"])
    return entry + "@" + str(term["dim"])


def obs_signature(terms: Sequence[str], action_size: int) -> str:
    """canonical v2 观测布局串："v2" ( "|" <id>[*scale]@dim )*。"""
    entries = [_obs_entry(parse_obs_term(spec, action_size)) for spec in terms]
    return "|".join([SIGNATURE_VERSION] + entries)


def obs_signature_dim(signature: str) -> int:
    """规范串里所有 entry 的 dim 之和（用于「metadata 自洽」检查）。"""
    total = 0
    for entry in _signature_entries(signature):
        _, _, dim_text = entry.rpartition("@")
        try:
            total += int(dim_text)
        except ValueError:
            raise LayoutError(f"规范串 entry 缺少 @dim（{entry!r}）：{signature!r}")
    return total



def parse_action_term(spec: str) -> Dict:
    """解析一条动作词条，返回 {index,output,name,scale,clip,id}。

    index 必填（0 基 actor 输出槽），output 必填（输出接口路径），
    name 缺省 = output；每个动作词条 dim 恒为 1。
    """
    kv = _split_tokens(spec)
    _check_keys(kv, ACTION_KEYS, spec)

    if "index" not in kv:
        raise LayoutError(f"动作词条缺少 index=（0 基 actor 输出槽）：{spec!r}")
    try:
        index = int(kv["index"])
    except ValueError:
        raise LayoutError(f"index 不是整数（index={kv['index']}）：{spec!r}")
    if index < 0:
        raise LayoutError(f"index 不能为负（index={index}）：{spec!r}")

    if "output" not in kv:
        raise LayoutError(f"动作词条缺少 output=（输出接口路径）：{spec!r}")
    output = kv["output"]
    if output == "":
        raise LayoutError(f"output= 不能为空：{spec!r}")

    name = kv.get("name", output)
    if name == "":
        raise LayoutError(f"name= 不能为空：{spec!r}")

    scale, clip, _ = _scale_clip_default(kv, spec)
    return {
        "index": index,
        "output": output,
        "name": name,
        "scale": scale,
        "clip": clip,
        "id": f"#{index}:{name}" + ("*" + format_scale(scale) if scale != 1.0 else ""),
    }


def action_signature(terms: Sequence[str]) -> str:
    """canonical v2 动作布局串："v2" ( "|" "#" <index> ":" <name> [ "*" <scale> ] )*。

    entry 按 index 升序排列：index 是显式槽位，YAML 里的书写顺序不参与语义
    （观测词条相反：观测串的 entry 顺序**就是** obs 索引顺序）。
    顺带校验 index 恰为 0..M-1 的完整置换。
    """
    parsed = [parse_action_term(spec) for spec in terms]
    if not parsed:
        raise LayoutError("action_terms 为空（至少一条动作词条）")
    parsed.sort(key=lambda term: term["index"])
    indices = [term["index"] for term in parsed]
    if indices != list(range(len(parsed))):
        raise LayoutError(
            f"动作 index 必须是 0..{len(parsed) - 1} 的完整置换（无空洞/无重复），实际 {indices}"
        )
    return "|".join([SIGNATURE_VERSION] + [term["id"] for term in parsed])



def _signature_entries(signature: str) -> List[str]:
    text = (signature or "").strip()
    if text in ("", SIGNATURE_VERSION):
        return []
    if text.startswith(SIGNATURE_VERSION + "|"):
        text = text[len(SIGNATURE_VERSION) + 1:]
    return [entry for entry in text.split("|") if entry != ""]


def parse_signature(signature: str) -> Dict:
    """拆开规范串：{"version": "v2"|"v1", "entries": [...]}。

    没有 "v2" 前缀的一律算 v1（含显式 v1 前缀、类型派生前缀 id、
    旧动作条目 "<joint>:<mode>[*scale][@default]"）—— 便于校验器给出
    「模型带 v1 metadata，需重盖章」而不是一个困惑的 mismatch。
    """
    text = (signature or "").strip()
    version = "v2" if (text == SIGNATURE_VERSION or text.startswith(SIGNATURE_VERSION + "|")) else "v1"
    return {"version": version, "entries": _signature_entries(text)}


def is_v1_signature(signature: str) -> bool:
    """True = 旧规范串（类型前缀会跨实现漂移，需 stamp_layout_metadata.py 重盖章）。"""
    return parse_signature(signature)["version"] == "v1"



def layout_hash(obs_sig: str, act_sig: str, obs_size: int, act_size: int) -> int:
    """FNV1a64(obs_sig + "||" + act_sig + "||" + "<obs_size>x<act_size>")。"""
    payload = "%s||%s||%dx%d" % (obs_sig, act_sig, int(obs_size), int(act_size))
    return fnv1a64(payload.encode("utf-8"))


def layout_payload(obs_sig: str, act_sig: str, obs_size: int, act_size: int) -> str:
    """layout_hash 的原文（排障时打印，能一眼看出哪一段不一致）。"""
    return "%s||%s||%dx%d" % (obs_sig, act_sig, int(obs_size), int(act_size))


def model_id(path) -> int:
    """FNV1a64(模型文件全字节) —— 模型身份指纹（非密码学）。"""
    with open(str(path), "rb") as handle:
        return fnv1a64(handle.read())


def model_id_hex(path) -> str:
    return hex16(model_id(path))



def load_config(config_path, node: str = DEFAULT_NODE):
    """读部署 YAML → (obs_terms, act_terms, obs_size, act_size)。

    YAML 形如：
        rl_bridge:
          ros__parameters:
            rl_obs_size: 20          # 可省（缺省由词条推导，写了就必须一致）
            rl_action_size: 6
            observation_terms: [ "path=/x", ... ]
            action_terms:      [ "index=0 output=/a", ... ]
    也接受直接给出 ros__parameters 内容（无节点包裹）的简写。
    """
    yaml = _import_yaml()
    try:
        with open(str(config_path), "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except OSError as exc:
        raise LayoutError(f"无法读取配置 {config_path}: {exc}")
    except yaml.YAMLError as exc:
        raise LayoutError(f"YAML 解析失败 {config_path}: {exc}")

    params = _select_params(document, node, config_path)
    obs_terms = _terms_list(params, "observation_terms", config_path)
    act_terms = _terms_list(params, "action_terms", config_path)

    declared_act = _declared_size(params, "rl_action_size", config_path)
    act_size = len(act_terms) if declared_act is None else declared_act
    if declared_act is not None and declared_act != len(act_terms):
        raise LayoutError(
            f"rl_action_size={declared_act} 与 action_terms 条数 {len(act_terms)} 不一致"
            f"（{config_path} 节点 {node}）"
        )
    action_signature(act_terms)
    _check_action_index_order(act_terms, config_path, node)

    parsed_obs = [parse_obs_term(spec, act_size) for spec in obs_terms]
    dim_sum = sum(term["dim"] for term in parsed_obs)
    declared_obs = _declared_size(params, "rl_obs_size", config_path)
    obs_size = dim_sum if declared_obs is None else declared_obs
    if declared_obs is not None and declared_obs != dim_sum:
        raise LayoutError(
            f"rl_obs_size={declared_obs} 与观测词条维度之和 {dim_sum} 不一致"
            f"（{config_path} 节点 {node}）"
        )
    _check_obs_index_order(obs_terms, parsed_obs, config_path, node)
    return obs_terms, act_terms, obs_size, act_size


def _select_params(document, node: str, config_path) -> Dict:
    if not isinstance(document, dict) or not document:
        raise LayoutError(f"配置 {config_path} 为空或不是 YAML 映射")
    section = document.get(node)
    if isinstance(section, dict):
        params = section.get("ros__parameters", section)
        if isinstance(params, dict):
            return params
    for candidate in ("ros__parameters",):
        if isinstance(document.get(candidate), dict):
            return document[candidate]
    if "observation_terms" in document:
        return document
    sections = sorted(
        key for key, value in document.items()
        if isinstance(value, dict) and ("ros__parameters" in value or "observation_terms" in value)
    )
    raise LayoutError(
        f"配置 {config_path} 里找不到节点 '{node}' 的 ros__parameters"
        f"（可用节点：{sections or '无'}；用 --node 指定）"
    )


def _terms_list(params: Dict, key: str, config_path) -> List[str]:
    if key not in params:
        raise LayoutError(f"配置 {config_path} 缺少 {key}（ros__parameters 下）")
    terms = params[key]
    if not isinstance(terms, (list, tuple)) or not terms:
        raise LayoutError(f"配置 {config_path} 的 {key} 必须是非空列表")
    out = []
    for spec in terms:
        if not isinstance(spec, str):
            raise LayoutError(f"配置 {config_path} 的 {key} 含非字符串词条 {spec!r}")
        out.append(spec)
    return out


def _declared_size(params: Dict, key: str, config_path) -> Optional[int]:
    if key not in params or params[key] is None:
        return None
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LayoutError(
            f"配置 {config_path} 的 {key}={value!r} 不是整数（C++ 侧只接受 int，浮点/字符串会启动失败）"
        )
    return value


def _check_obs_index_order(obs_terms, parsed_obs, config_path, node) -> None:
    """index= 只是「防静默错位」的冗余声明：与顺序冲突时警告（不失败）。

    观测串的 entry 顺序**就是** obs 索引顺序（顺序变了 layout_hash 就变），
    因此这里不做重排，只提示 YAML 自相矛盾。
    """
    import sys
    declared = []
    for spec in obs_terms:
        kv = _split_tokens(spec)
        if "index" in kv:
            try:
                declared.append(int(kv["index"]))
            except ValueError:
                raise LayoutError(f"观测词条 index 不是整数（index={kv['index']}）：{spec!r}")
    if not declared:
        return
    if len(declared) != len(obs_terms):
        return
    if len(set(declared)) != len(declared):
        raise LayoutError(f"观测词条 index= 有重复（{sorted(declared)}），{config_path} 节点 {node}")
    position, cursor = [], 0
    for term in parsed_obs:
        position.append(cursor)
        cursor += term["dim"]
    if declared != position:
        raise LayoutError(
            f"观测词条 index= 与词条顺序推导出的槽位不一致：声明 {declared} vs 顺序 {position}"
            f"（{config_path} 节点 {node}）；index= 是断言而非重排，桥 rl_bridge.cpp 会启动失败"
        )


def _check_action_index_order(act_terms, config_path, node) -> None:
    """动作词条的书写顺序不等于 index 顺序时警告（不失败）。

    规范串按 index 升序规范化，所以书写顺序不影响 layout_hash；但桥 rl_bridge.cpp
    要求「第 i 条词条的 index == i」（否则启动失败），因此这里必须提示。
    """
    import sys
    declared = [parse_action_term(spec)["index"] for spec in act_terms]
    if declared != list(range(len(declared))):
        raise LayoutError(
            f"action_terms 的书写顺序必须与 index 一致：实际 {declared}，期望 "
            f"{list(range(len(declared)))}（{config_path} 节点 {node}）"
        )



def read_metadata(model_path) -> Dict[str, str]:
    """ONNX metadata_props → dict（缺省空 dict）。"""
    onnx = _import_onnx()
    model = onnx.load(str(model_path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def write_metadata(model_path, updates: Dict[str, object]) -> None:
    """写入/覆盖 metadata_props（保留其它键，值一律转字符串；幂等）。

    已全部一致时不重写文件 —— 保证重复盖章得到字节相同的模型（model_id 不变）。
    """
    onnx = _import_onnx()
    path = str(model_path)
    model = onnx.load(path, load_external_data=False)
    current = {prop.key: prop.value for prop in model.metadata_props}
    merged = dict(current)
    merged.update({str(key): str(value) for key, value in updates.items()})
    if merged == current:
        return
    del model.metadata_props[:]
    for key, value in merged.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, path)


def stamp_metadata(model_path, obs_sig: str, act_sig: str, obs_size: int, act_size: int,
                   policy_version: Optional[str] = None,
                   obs_mean=None, obs_std=None,
                   obs_clip=None, action_clip=None) -> Dict[str, str]:
    """按规范串盖章：返回实际写入的 metadata（含 policy_layout_hash）。"""
    updates = {
        "rmcs_obs_layout": obs_sig,
        "rmcs_actions_layout": act_sig,
        "policy_layout_hash": hex16(layout_hash(obs_sig, act_sig, obs_size, act_size)),
    }
    if policy_version:
        updates["policy_version"] = str(policy_version)
    if obs_mean is not None:
        updates["rmcs_obs_mean"] = _join_floats(obs_mean)
    if obs_std is not None:
        updates["rmcs_obs_std"] = _join_floats(obs_std)
    if obs_clip is not None:
        updates["rmcs_obs_clip"] = format_scale(obs_clip)
    if action_clip is not None:
        updates["rmcs_action_clip"] = format_scale(action_clip)
    write_metadata(model_path, updates)
    return updates


def _join_floats(values) -> str:
    if isinstance(values, str):
        values = [v for v in values.replace(" ", "").split(",") if v != ""]
    return ",".join(format_scale(float(v)) for v in values)



def _self_test() -> None:
    """断言语法表（含「同签名与 C++ 接口类型无关」）+ FNV 稳定性。"""
    for spec in ("path=/chassis/control_height",
                 "path=/chassis/control_height type=scalar"):
        term = parse_obs_term(spec, 6)
        assert term["kind"] == "scalar", term
        assert term["dim"] == 1 and term["id"] == "/chassis/control_height", term
        assert term["scale"] == 1.0 and term["clip"] is None, term

    base = parse_obs_term("path=/v take=x", 6)
    assert (base["dim"], base["id"], base["kind"]) == (1, "vec3c:/v:x", "component"), base
    for interface in ("vector3", "direction_vector", "scalar"):
        other = parse_obs_term(f"path=/v take=x type={interface}", 6)
        assert (other["dim"], other["id"]) == (1, "vec3c:/v:x"), other
    assert obs_signature(["path=/v take=x"], 6) == obs_signature(["path=/v take=x type=direction_vector"], 6)
    assert obs_signature(["path=/v take=x"], 6) == "v2|vec3c:/v:x@1"
    for axis in ("x", "y", "z"):
        assert parse_obs_term(f"path=/v take={axis}", 6)["id"] == f"vec3c:/v:{axis}"

    for alias in ("vec3", "vec", "all", "vector"):
        term = parse_obs_term(f"path=/v take={alias}", 6)
        assert (term["dim"], term["id"], term["kind"]) == (3, "vec3:/v", "vec3"), term
    assert parse_obs_term("path=/v take=vec3 scale=0.5", 6)["scale"] == 0.5

    gravity = parse_obs_term("path=/imu/quaternion transform=projected_gravity", 6)
    assert (gravity["dim"], gravity["id"], gravity["kind"]) == (3, "gravity:/imu/quaternion", "gravity")
    assert parse_obs_term("path=/imu/quaternion take=gravity", 6)["id"] == "gravity:/imu/quaternion"
    assert obs_signature(["path=/imu/quaternion transform=projected_gravity type=quaternion"], 6) \
        == obs_signature(["path=/imu/quaternion transform=projected_gravity"], 6)
    _expect_error(lambda: parse_obs_term("path=/imu/quaternion type=quaternion", 6), "quaternion")

    joint_pos = parse_obs_term("type=joint_pos joints=a,b relative=true", 6)
    assert (joint_pos["dim"], joint_pos["id"], joint_pos["relative"]) == (2, "joint_pos:rel:a+b", True)
    assert parse_obs_term("type=joint_pos joints=a,b", 6)["id"] == "joint_pos:abs:a+b"
    assert parse_obs_term("type=joint_pos joints=a,b zero=true", 6)["id"] == "joint_pos:zero:a+b"
    assert parse_obs_term("type=joint_pos joints=a,b relative=false", 6)["id"] == "joint_pos:abs:a+b"
    assert joint_pos["joints"] == ["a", "b"] and joint_pos["kind"] == "joint_pos"
    assert parse_obs_term("type=joint_vel joints=a,b", 6)["id"] == "joint_vel:abs:a+b"
    assert parse_obs_term("type=joint_torque joints=a,b scale=0.25", 6)["id"] == "joint_torque:abs:a+b"
    assert parse_obs_term("type=joint_torque joints=a,b scale=0.25", 6)["scale"] == 0.25
    _expect_error(lambda: parse_obs_term("type=joint_pos joints=", 6), "joints")

    last = parse_obs_term("type=last_action", 6)
    assert (last["dim"], last["id"]) == (6, "last_action:0+1+2+3+4+5"), last
    assert last["indices"] == [0, 1, 2, 3, 4, 5]
    assert parse_obs_term("type=last_action indices=0,1,2", 6)["id"] == "last_action:0+1+2"
    range_term = parse_obs_term("type=last_action indices=0..5", 6)
    assert (range_term["dim"], range_term["id"]) == (6, "last_action:0+1+2+3+4+5"), range_term
    _expect_error(lambda: parse_obs_term("type=last_action indices=0..9", 6), "越界")

    const = parse_obs_term("type=constant value=1,0,0.5", 6)
    assert (const["dim"], const["id"], const["constants"]) == (3, "constant:1+0+0.5", [1.0, 0.0, 0.5]), const
    assert parse_obs_term("type=constant value=1.0,0.0000001", 6)["id"] == "constant:1+1e-07"

    named = parse_obs_term("path=/v take=vec3 name=my_obs", 6)
    assert (named["id"], named["dim"]) == ("my_obs", 3), named

    for alias in ("double", "bool", "int", "size", "size_t", "vec3"):
        assert parse_obs_term(f"path=/v take=x type={alias}", 6)["id"] == "vec3c:/v:x", alias
    assert parse_obs_term("path=/v type=bool", 6)["id"] == "/v"
    assert parse_obs_term("path=/v clip=0.5", 6)["clip"] == 0.5
    assert parse_obs_term("path=/v clip=-1:1", 6)["clip"] == (-1.0, 1.0)
    _expect_error(lambda: parse_obs_term("path=/v clip=1:-1", 6), "clip")

    assert obs_signature(["path=/v scale=2 clip=10 index=3 default=0.1"], 6) == "v2|/v*2@1"
    assert obs_signature(["path=/v"], 6) == "v2|/v@1"

    action = parse_action_term("index=0 output=/rl/action/lf0")
    assert (action["index"], action["output"], action["name"], action["id"]) \
        == (0, "/rl/action/lf0", "/rl/action/lf0", "#0:/rl/action/lf0"), action
    assert parse_action_term("index=1 output=/o name=hip scale=0.5")["id"] == "#1:hip*0.5"
    assert action_signature(["index=0 output=/a", "index=1 output=/b scale=0.5"]) == "v2|#0:/a|#1:/b*0.5"
    assert action_signature(["index=1 output=/b", "index=0 output=/a"]) == "v2|#0:/a|#1:/b"
    _expect_error(lambda: action_signature(["index=0 output=/a", "index=2 output=/c"]), "置换")
    _expect_error(lambda: action_signature(["index=0 output=/a", "index=0 output=/b"]), "置换")

    _expect_error(lambda: parse_obs_term("path=/v scale=0", 6), "scale")
    _expect_error(lambda: parse_obs_term("path=/v bogus=1", 6), "未知键")
    _expect_error(lambda: parse_obs_term("take=x", 6), "path")
    _expect_error(lambda: parse_obs_term("path=/v take=w", 6), "take")
    _expect_error(lambda: parse_obs_term("path=/v transform=foo", 6), "transform")
    _expect_error(lambda: parse_action_term("output=/a"), "index")
    _expect_error(lambda: parse_action_term("index=0"), "output")

    assert fnv1a64(b"") == 0xCBF29CE484222325
    assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert fnv1a64(b"abc") == 0xE71FA2190541574B
    assert fnv1a64(b"foobar") == 0x85944171F73967E8
    fixed = layout_hash("v2|/chassis/control_height@1|vec3c:/chassis/control_velocity:x@1",
                        "v2|#0:/a", 20, 6)
    assert fixed == 0xC841398B19839E56, hex16(fixed)
    assert hex16(0x1A2B3C4D5E6F7788) == "0x1a2b3c4d5e6f7788"
    assert hex16(layout_hash("v2", "v2", 0, 0)) == hex16(fnv1a64(b"v2||v2||0x0"))

    assert parse_signature("v2|/a@1")["version"] == "v2"
    assert parse_signature("v2")["version"] == "v2"
    assert parse_signature("v1|/a@1")["version"] == "v1"
    assert is_v1_signature("vector3_component:/v:x@1")
    assert is_v1_signature("direction_vector_component:/v:x@1")
    assert is_v1_signature("") and is_v1_signature(None)
    assert not is_v1_signature("v2|vec3c:/v:x@1")
    assert obs_signature_dim("v2|/a@1|vec3:/v@3") == 4

    print("rl_layout self-test OK")


def _expect_error(fn, needle: str) -> None:
    try:
        fn()
    except LayoutError as exc:
        assert needle in str(exc), f"错误消息 {exc!r} 未包含 {needle!r}"
        return
    raise AssertionError(f"应当抛 LayoutError（{needle}）但没有")


if __name__ == "__main__":
    _self_test()
