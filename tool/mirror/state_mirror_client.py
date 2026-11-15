"""Read-only Foxglove transport, exact-stamp frames and JSONL replay for StateMirror."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import struct
import threading
import time

SCHEMA = "rmcs_state_mirror_v2"
RECORD_SCHEMA = "rmcs_state_mirror_record_v2"
SUBPROTOCOLS = ("foxglove.sdk.v1", "foxglove.websocket.v1")
MESSAGE_TYPES = {
    "sensor_msgs/msg/JointState", "sensor_msgs/msg/Imu",
    "geometry_msgs/msg/Vector3Stamped", "diagnostic_msgs/msg/DiagnosticArray",
}


def finite(values, count=None):
    if (count is not None and len(values) != count) or not all(math.isfinite(v) for v in values):
        raise ValueError("incomplete or non-finite numeric data")


def vector(value):
    return (float(value.x), float(value.y), float(value.z))


@dataclass(frozen=True)
class Frame:
    stamp_ns: int
    joints: dict = field(default_factory=dict)
    imu: dict | None = None
    vectors: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    frame_id: str = ""

    def validate(self):
        if type(self.stamp_ns) is not int or self.stamp_ns <= 0:
            raise ValueError("zero/negative sample timestamp")
        for name, joint in self.joints.items():
            if not isinstance(name, str) or len(joint) != 3:
                raise ValueError("invalid joint name or fields")
            finite(joint[:2], 2)
            if joint[2] is not None:
                finite((joint[2],))
        for value in self.vectors.values():
            finite(value, 3)
        if self.imu is not None:
            q = self.imu["quaternion_wxyz"]
            finite(q, 4)
            finite(self.imu["gyro"], 3)
            if abs(sum(v * v for v in q) - 1.) > .01:
                raise ValueError("IMU quaternion norm squared is not near one")
            if self.imu.get("age_s") is not None:
                finite((self.imu["age_s"],))
        if self.values.get("schema", SCHEMA) != SCHEMA:
            raise ValueError("unsupported mirror schema")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.values.items()):
            raise ValueError("diagnostics must be string key/value pairs")
        return self

    @classmethod
    def from_dict(cls, row):
        imu = row.get("imu")
        if imu is not None:
            imu = dict(imu, quaternion_wxyz=tuple(imu["quaternion_wxyz"]), gyro=tuple(imu["gyro"]))
        return cls(row["stamp_ns"], {k: tuple(v) for k, v in row.get("joints", {}).items()},
                   imu, {k: tuple(v) for k, v in row.get("vectors", {}).items()},
                   dict(row.get("values", {})), row.get("frame_id", "")).validate()


class FrameAssembler:
    """Use the diagnostics manifest for this stamp; never combine adjacent cycles."""

    def __init__(self, topic_prefix, frame_id=None):
        self.prefix = topic_prefix.rstrip("/")
        self.frame_id = frame_id
        self.pending = {}
        self.last_stamp = 0

    def accept(self, topic, message):
        if not topic.startswith(self.prefix + "/"):
            return None
        stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        if stamp <= self.last_stamp:
            return None
        bucket = self.pending.setdefault(stamp, {})
        bucket[topic] = message
        while len(self.pending) > 32:
            del self.pending[min(self.pending)]
        diagnostic_topic = self.prefix + "/diagnostics"
        if diagnostic_topic not in bucket:
            return None
        statuses = [s for s in bucket[diagnostic_topic].status if s.name == self.prefix]
        if len(statuses) != 1:
            raise ValueError("missing/duplicate mirror diagnostics")
        entries = statuses[0].values
        values = {v.key: v.value for v in entries}
        if len(values) != len(entries) or values.get("schema") != SCHEMA:
            raise ValueError("duplicate diagnostics keys or unsupported mirror schema")
        names = values.get("mirror/topics", "").split(",")
        topics = set(names)
        if len(topics) != len(names) or diagnostic_topic not in topics or any(
                not t.startswith(self.prefix + "/") for t in topics):
            raise ValueError("invalid frame topic manifest")
        if not topics.issubset(bucket):
            return None
        frames = {bucket[t].header.frame_id for t in topics}
        if len(frames) != 1 or self.frame_id is not None and frames != {self.frame_id}:
            raise ValueError("inconsistent coordinate frames")
        joints, imu, vectors = {}, None, {}
        for name in topics:
            message = bucket[name]
            kind = name.removeprefix(self.prefix + "/")
            if kind == "joint_states":
                if (len(set(message.name)) != len(message.name)
                        or len(message.position) != len(message.name)
                        or len(message.velocity) != len(message.name)):
                    raise ValueError("duplicate joints or incomplete position/velocity")
                for joint, position, velocity in zip(message.name, message.position, message.velocity):
                    age = values.get("age_s/" + joint)
                    joints[joint] = (float(position), float(velocity), None if age is None else float(age))
            elif kind == "imu_body":
                q = message.orientation
                age = values.get("age_s/imu")
                imu = dict(quaternion_wxyz=(float(q.w), float(q.x), float(q.y), float(q.z)),
                           gyro=vector(message.angular_velocity), age_s=None if age is None else float(age))
                if message.orientation_covariance[0] < 0:
                    raise ValueError("IMU orientation is unavailable")
            elif kind != "diagnostics":
                vectors[kind] = vector(message.vector)
        result = Frame(stamp, joints, imu, vectors, values, frames.pop()).validate()
        self.last_stamp = stamp
        self.pending = {key: value for key, value in self.pending.items() if key > stamp}
        return result


def convert_v1(row):
    """The legacy recording stored unnamed, fixed-order arrays; name them once here."""
    names = ("left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint",
             "left_wheel", "right_wheel")
    gimbal = ("gimbal_yaw_joint", "gimbal_pitch_joint")
    joints, values = {}, {"schema": SCHEMA, "record/source_schema": "wheel_leg_telemetry_v1",
                           "coordinate": "rmcs_hardware_output_shaft_v1"}
    positions, velocities = row["positions"], row["velocities"]
    finite(positions, 6)
    finite(velocities, 6)
    ages = row.get("sensor_age_s", ())
    for i, name in enumerate(names):
        joints[name] = (positions[i], velocities[i], ages[i] if len(ages) == 7 else None)
    gp, gv = row.get("gimbal_positions", ()), row.get("gimbal_velocities", ())
    if gp or gv:
        finite(gp, 2)
        finite(gv, 2)
        joints.update({name: (gp[i], gv[i], None) for i, name in enumerate(gimbal)})

    def append(key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                append(key + "/" + k, v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                append(key + "/" + str(i), v)
        elif value is not None:
            values[key] = str(value).lower() if isinstance(value, bool) else str(value)

    for key, value in row.items():
        if key not in {"stamp_ns", "positions", "velocities", "gimbal_positions", "gimbal_velocities",
                       "quaternion_wxyz", "gyro_body", "gravity_rl", "gyro_rl", "kind", "record_time_s"}:
            append(key, value)
    for i, value in enumerate(row.get("rmcs_model_positions", ())):
        values["state/position/" + str(i)] = str(value)
    for i, value in enumerate(row.get("model_offsets", ())):
        values["model_offset/" + names[i]] = str(value)
    imu = dict(quaternion_wxyz=row["quaternion_wxyz"], gyro=row["gyro_body"],
               age_s=ages[6] if len(ages) == 7 else None)
    vectors = {"rl_projected_gravity": row["gravity_rl"], "rl_angular_velocity": row["gyro_rl"]}
    return Frame(row["stamp_ns"], joints, imu, vectors, values, "chassis_body").validate()


class LatestFrame:
    def __init__(self, record=None, source="foxglove", metadata=None):
        self.lock = threading.RLock()
        self.listeners = []
        self.frame = None
        self.received = 0.
        self.status = "waiting for telemetry"
        self.started = time.monotonic()
        self.record = open(record, "x") if record else None
        self.source, self.count = source, 0
        if self.record:
            self.record.write(json.dumps(dict(kind="metadata", schema=RECORD_SCHEMA,
                                              metadata=metadata or {}), allow_nan=False) + "\n")
            self.record.flush()

    def put(self, frame, source=None):
        frame.validate()
        with self.lock:
            self.frame, self.received = frame, time.monotonic()
            self.status = "receiving"
            if source is not None:
                self.source = source
            self.count += 1
            if self.record:
                self.record.write(json.dumps(asdict(frame) | dict(kind="frame",
                    record_time_s=self.received - self.started, source=self.source), allow_nan=False) + "\n")
                self.record.flush()
            for listener in self.listeners:
                listener("frame", frame)

    def _row(self, kind, value, dispatch=True):
        with self.lock:
            if self.record:
                self.record.write(json.dumps(value | dict(kind=kind,
                    record_time_s=time.monotonic() - self.started), allow_nan=False) + "\n")
                self.record.flush()
            if dispatch:
                for listener in self.listeners:
                    listener(kind, value)

    def event(self, name, *, dispatch=True, **data):
        self._row("event", dict(name=name, **data), dispatch=dispatch)

    def set_status(self, status):
        with self.lock:
            self.status = status

    def get(self):
        with self.lock:
            return self.frame, time.monotonic() - self.received, self.status

    def close(self):
        with self.lock:
            if self.record:
                self.record.close()
                self.record = None


def read_rows(path, expected_metadata=None):
    with open(path) as stream:
        first = next(stream, None)
        if first is None:
            raise ValueError("empty mirror recording")
        header = json.loads(first)
        legacy = header.get("schema") == "wheel_leg_mirror_record_v1"
        if header.get("kind") != "metadata" or not legacy and header.get("schema") != RECORD_SCHEMA:
            raise ValueError("not a supported mirror recording")
        for key, value in (expected_metadata or {}).items():
            if header.get("metadata", {}).get(key) != value:
                raise ValueError(f"recording {key} differs from selected assets/mapping")
        previous = -1.
        for line in stream:
            row = json.loads(line)
            elapsed = row["record_time_s"]
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0 or elapsed < previous:
                raise ValueError("recording time must be finite, nonnegative and nondecreasing")
            previous = elapsed
            if legacy and row.get("kind", "snapshot") == "snapshot":
                row = row | asdict(convert_v1(row)) | {"kind": "frame"}
            if row.get("kind") == "frame":
                Frame.from_dict(row)
            yield elapsed, row


def read_record(path, expected_metadata=None):
    for elapsed, row in read_rows(path, expected_metadata):
        if row.get("kind") == "frame":
            yield elapsed, Frame.from_dict(row)


class ReplayReader(threading.Thread):
    def __init__(self, path, latest, speed=1., loop=False, metadata=None):
        super().__init__(daemon=True)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("replay speed must be positive")
        self.path, self.latest, self.speed, self.loop = path, latest, speed, loop
        self.metadata = metadata
        self.stopped = threading.Event()

    def run(self):
        try:
            while not self.stopped.is_set():
                started, count = time.monotonic(), 0
                self.latest.event("replay_begin")
                for elapsed, row in read_rows(self.path, self.metadata):
                    kind = row.get("kind")
                    if kind not in ("frame", "event"):
                        continue
                    if self.stopped.wait(max(0., started + elapsed / self.speed - time.monotonic())):
                        return
                    if kind == "event":
                        self.latest._row("event", {k: v for k, v in row.items() if k not in ("kind", "record_time_s")})
                    else:
                        self.latest.put(Frame.from_dict(row), source="replay")
                        count += 1
                if not count:
                    raise ValueError("recording contains no frames")
                if not self.loop:
                    self.latest.set_status("replay complete")
                    return
        except Exception as error:
            self.latest.set_status(f"replay failed: {error}")

    def close(self):
        self.stopped.set()
        self.join(timeout=5)


class FoxgloveReader(threading.Thread):
    """Only emits subscribe operations; has no publish/service path."""

    def __init__(self, url, latest, topic_prefix, frame_id=None):
        super().__init__(daemon=True)
        self.url, self.latest = url, latest
        self.prefix, self.frame_id = topic_prefix.rstrip("/"), frame_id
        self.epoch = 0
        self.stopped = threading.Event()

    def run(self):
        try:
            import asyncio
            from rosbags.typesys import Stores, get_typestore
            self.typestore = get_typestore(Stores.ROS2_JAZZY)
            asyncio.run(self.receive())
        except Exception as error:
            self.latest.set_status(f"receiver failed: {error}")

    async def receive(self):
        import asyncio
        import websockets
        while not self.stopped.is_set():
            try:
                async with websockets.connect(self.url, subprotocols=list(SUBPROTOCOLS),
                                              open_timeout=3, close_timeout=1, max_size=4 * 1024 * 1024,
                                              max_queue=16) as socket:
                    if socket.subprotocol not in SUBPROTOCOLS:
                        raise ValueError("server did not negotiate a supported Foxglove subprotocol")
                    self.epoch += 1
                    self.latest.event("source_epoch", epoch=self.epoch)
                    assembler, subscriptions = FrameAssembler(self.prefix, self.frame_id), {}
                    self.latest.set_status("connected; waiting for mirror diagnostics")
                    while not self.stopped.is_set():
                        try:
                            packet = await asyncio.wait_for(socket.recv(), timeout=.5)
                        except asyncio.TimeoutError:
                            continue
                        if isinstance(packet, str):
                            event = json.loads(packet)
                            if event.get("op") == "serverInfo":
                                # The bridge may be foxglove_bridge or the Foxglove
                                # SDK.  Keep the negotiated transport visible in a
                                # recording so a reachable-but-empty ROS graph is
                                # distinguishable from a decode failure.
                                self.latest.event(
                                    "server_info",
                                    server_name=event.get("name", ""),
                                    capabilities=event.get("capabilities", []),
                                    supported_encodings=event.get("supportedEncodings", []),
                                    metadata=event.get("metadata", {}),
                                    dispatch=False,
                                )
                            elif event.get("op") == "advertise":
                                requests = []
                                rejected = []
                                for channel in event["channels"]:
                                    topic = channel["topic"]
                                    if not topic.startswith(self.prefix + "/"):
                                        continue
                                    if channel["encoding"] != "cdr" or channel["schemaName"] not in MESSAGE_TYPES:
                                        rejected.append(
                                            f"{topic} ({channel.get('encoding')}/{channel.get('schemaName')})"
                                        )
                                        continue
                                    ident = channel["id"]
                                    channel_info = (topic, channel["schemaName"])
                                    if subscriptions.get(ident) != channel_info:
                                        subscriptions[ident] = channel_info
                                        requests.append({"id": ident, "channelId": ident})
                                if requests:
                                    await socket.send(json.dumps({"op": "subscribe", "subscriptions": requests}))
                                details = [f"subscribed to {len(subscriptions)} mirror channel(s)"]
                                if rejected:
                                    details.append("rejected: " + ", ".join(rejected))
                                self.latest.set_status("; ".join(details))
                            elif event.get("op") == "unadvertise":
                                for ident in event["channelIds"]:
                                    subscriptions.pop(ident, None)
                                assembler = FrameAssembler(self.prefix, self.frame_id)
                                self.latest.set_status("telemetry publisher disappeared")
                            elif event.get("op") == "status" and event.get("level", 0) >= 2:
                                self.latest.set_status(event.get("message", "Foxglove error"))
                            continue
                        if len(packet) < 13 or packet[0] != 1:
                            continue
                        ident, _ = struct.unpack_from("<IQ", packet, 1)
                        channel = subscriptions.get(ident)
                        if channel is None:
                            continue
                        topic, schema = channel
                        try:
                            message = self.typestore.deserialize_cdr(packet[13:], schema)
                            snapshot = assembler.accept(topic, message)
                            if snapshot:
                                self.latest.put(snapshot)
                        except (ValueError, KeyError, IndexError, TypeError) as error:
                            self.latest.set_status(f"rejected sample: {error}")
            except Exception as error:
                self.latest.set_status(f"disconnected: {error}")
                for _ in range(10):
                    if self.stopped.is_set():
                        break
                    await asyncio.sleep(.1)

    def close(self):
        self.stopped.set()
        self.join(timeout=5)

