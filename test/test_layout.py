#!/usr/bin/env python3
"""Regression fixtures for the Python/C++ layout contract."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import rl_layout as layout


def _self_test() -> None:
    """断言语法表（含「同签名与 C++ 接口类型无关」）+ FNV 稳定性。"""
    for spec in ("path=/chassis/control_height",
                 "path=/chassis/control_height type=scalar"):
        term = layout.parse_obs_term(spec, 6)
        assert term["kind"] == "scalar", term
        assert term["dim"] == 1 and term["id"] == "/chassis/control_height", term
        assert term["scale"] == 1.0 and term["clip"] is None, term

    base = layout.parse_obs_term("path=/v take=x", 6)
    assert (base["dim"], base["id"], base["kind"]) == (1, "vec3c:/v:x", "component"), base
    for interface in ("vector3", "direction_vector", "scalar"):
        other = layout.parse_obs_term(f"path=/v take=x type={interface}", 6)
        assert (other["dim"], other["id"]) == (1, "vec3c:/v:x"), other
    assert layout.obs_signature(["path=/v take=x"], 6) == layout.obs_signature(["path=/v take=x type=direction_vector"], 6)
    assert layout.obs_signature(["path=/v take=x"], 6) == "v3-history=1|vec3c:/v:x@1"
    for axis in ("x", "y", "z"):
        assert layout.parse_obs_term(f"path=/v take={axis}", 6)["id"] == f"vec3c:/v:{axis}"

    for alias in ("vec3", "vec", "all", "vector"):
        term = layout.parse_obs_term(f"path=/v take={alias}", 6)
        assert (term["dim"], term["id"], term["kind"]) == (3, "vec3:/v", "vec3"), term
    assert layout.parse_obs_term("path=/v take=vec3 scale=0.5", 6)["scale"] == 0.5

    gravity = layout.parse_obs_term("path=/imu/quaternion transform=projected_gravity", 6)
    assert (gravity["dim"], gravity["id"], gravity["kind"]) == (3, "gravity:/imu/quaternion", "gravity")
    assert layout.parse_obs_term("path=/imu/quaternion take=gravity", 6)["id"] == "gravity:/imu/quaternion"
    assert layout.obs_signature(["path=/imu/quaternion transform=projected_gravity type=quaternion"], 6) \
        == layout.obs_signature(["path=/imu/quaternion transform=projected_gravity"], 6)
    _expect_error(lambda: layout.parse_obs_term("path=/imu/quaternion type=quaternion", 6), "quaternion")

    joint_pos = layout.parse_obs_term("type=joint_pos joints=a,b relative=true", 6)
    assert (joint_pos["dim"], joint_pos["id"], joint_pos["relative"]) == (2, "joint_pos:rel:a+b", True)
    assert layout.parse_obs_term("type=joint_pos joints=a,b", 6)["id"] == "joint_pos:abs:a+b"
    assert layout.parse_obs_term("type=joint_pos joints=a,b zero=true", 6)["id"] == "joint_pos:zero:a+b"
    assert layout.parse_obs_term("type=joint_pos joints=a,b relative=false", 6)["id"] == "joint_pos:abs:a+b"
    assert joint_pos["joints"] == ["a", "b"] and joint_pos["kind"] == "joint_pos"
    assert layout.parse_obs_term("type=joint_vel joints=a,b", 6)["id"] == "joint_vel:abs:a+b"
    assert layout.parse_obs_term("type=joint_torque joints=a,b scale=0.25", 6)["id"] == "joint_torque:abs:a+b"
    assert layout.parse_obs_term("type=joint_torque joints=a,b scale=0.25", 6)["scale"] == 0.25
    _expect_error(lambda: layout.parse_obs_term("type=joint_pos joints=", 6), "joints")

    last = layout.parse_obs_term("type=last_action", 6)
    assert (last["dim"], last["id"]) == (6, "last_action:0+1+2+3+4+5"), last
    assert last["indices"] == [0, 1, 2, 3, 4, 5]
    assert layout.parse_obs_term("type=last_action indices=0,1,2", 6)["id"] == "last_action:0+1+2"
    range_term = layout.parse_obs_term("type=last_action indices=0..5", 6)
    assert (range_term["dim"], range_term["id"]) == (6, "last_action:0+1+2+3+4+5"), range_term
    _expect_error(lambda: layout.parse_obs_term("type=last_action indices=0..9", 6), "越界")

    const = layout.parse_obs_term("type=constant value=1,0,0.5", 6)
    assert (const["dim"], const["id"], const["constants"]) == (3, "constant:1+0+0.5", [1.0, 0.0, 0.5]), const
    assert layout.parse_obs_term("type=constant value=1.0,0.0000001", 6)["id"] == "constant:1+1e-07"

    named = layout.parse_obs_term("path=/v take=vec3 name=my_obs", 6)
    assert (named["id"], named["dim"]) == ("my_obs", 3), named

    for alias in ("double", "bool", "int", "size", "size_t", "vec3"):
        assert layout.parse_obs_term(f"path=/v take=x type={alias}", 6)["id"] == "vec3c:/v:x", alias
    assert layout.parse_obs_term("path=/v type=bool", 6)["id"] == "/v"
    assert layout.parse_obs_term("path=/v clip=0.5", 6)["clip"] == 0.5
    assert layout.parse_obs_term("path=/v clip=-1:1", 6)["clip"] == (-1.0, 1.0)
    _expect_error(lambda: layout.parse_obs_term("path=/v clip=1:-1", 6), "clip")

    assert layout.obs_signature(["path=/v scale=2 clip=10 index=3 default=0.1"], 6) == "v3-history=1|/v*2@1"
    assert layout.obs_signature(["path=/v"], 6) == "v3-history=1|/v@1"

    action = layout.parse_action_term("index=0 output=/rl/action/lf0")
    assert (action["index"], action["output"], action["name"], action["id"]) \
        == (0, "/rl/action/lf0", "/rl/action/lf0", "#0:/rl/action/lf0"), action
    assert layout.parse_action_term("index=1 output=/o name=hip scale=0.5")["id"] == "#1:hip*0.5"
    assert layout.action_signature(["index=0 output=/a", "index=1 output=/b scale=0.5"]) == "v2|#0:/a|#1:/b*0.5"
    assert layout.action_signature(["index=1 output=/b", "index=0 output=/a"]) == "v2|#0:/a|#1:/b"
    _expect_error(lambda: layout.action_signature(["index=0 output=/a", "index=2 output=/c"]), "置换")
    _expect_error(lambda: layout.action_signature(["index=0 output=/a", "index=0 output=/b"]), "置换")

    _expect_error(lambda: layout.parse_obs_term("path=/v scale=0", 6), "scale")
    _expect_error(lambda: layout.parse_obs_term("path=/v bogus=1", 6), "未知键")
    _expect_error(lambda: layout.parse_obs_term("take=x", 6), "path")
    _expect_error(lambda: layout.parse_obs_term("path=/v take=w", 6), "take")
    _expect_error(lambda: layout.parse_obs_term("path=/v transform=foo", 6), "transform")
    _expect_error(lambda: layout.parse_action_term("output=/a"), "index")
    _expect_error(lambda: layout.parse_action_term("index=0"), "output")

    assert layout.fnv1a64(b"") == 0xCBF29CE484222325
    assert layout.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert layout.fnv1a64(b"abc") == 0xE71FA2190541574B
    assert layout.fnv1a64(b"foobar") == 0x85944171F73967E8
    fixed = layout.layout_hash("v2|/chassis/control_height@1|vec3c:/chassis/control_velocity:x@1",
                        "v2|#0:/a", 20, 6)
    assert fixed == 0xC841398B19839E56, layout.hex16(fixed)
    assert layout.hex16(0x1A2B3C4D5E6F7788) == "0x1a2b3c4d5e6f7788"
    assert layout.hex16(layout.layout_hash("v2", "v2", 0, 0)) == layout.hex16(layout.fnv1a64(b"v2||v2||0x0"))

    assert layout.parse_signature("v2|/a@1")["version"] == "v2"
    assert layout.parse_signature("v2")["version"] == "v2"
    assert layout.parse_signature("v1|/a@1")["version"] == "v1"
    assert layout.is_v1_signature("vector3_component:/v:x@1")
    assert layout.is_v1_signature("direction_vector_component:/v:x@1")
    assert layout.is_v1_signature("") and layout.is_v1_signature(None)
    assert not layout.is_v1_signature("v2|vec3c:/v:x@1")
    assert layout.obs_signature_dim("v2|/a@1|vec3:/v@3") == 4

    print("rl_layout self-test OK")


def _expect_error(fn, needle: str) -> None:
    try:
        fn()
    except layout.LayoutError as exc:
        assert needle in str(exc), f"错误消息 {exc!r} 未包含 {needle!r}"
        return
    raise AssertionError(f"应当抛 layout.LayoutError（{needle}）但没有")


if __name__ == "__main__":
    _self_test()
