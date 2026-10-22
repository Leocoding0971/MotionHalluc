"""Load full 22-joint arrays and execute validated numerical queries locally."""

import numpy as np

from ppv import kinematics as kin
from ppv.queries import LOCALIZERS, validate_call
from ppv.results import file_hash


def load_motion(path):
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        if hasattr(array, "close"):
            array.close()
        raise ValueError(f"Expected a .npy array: {path}")
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1:] != (22, 3):
        raise ValueError(f"Motion must have shape (T, 22, 3), T > 0; got {array.shape}: {path}")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"Motion must contain finite real coordinates: {path}")
    metadata = {"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": file_hash(path)}
    # Preserve floating point precision for parity with the original implementation.
    # Convert integer arrays before midpoint construction to avoid integer overflow.
    if array.dtype.kind in "iu":
        array = array.astype(np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        converted, _ = kin.augment_mocap_to_21(array)
    if not np.isfinite(converted).all():
        raise ValueError(f"Joint conversion produced nonfinite coordinates: {path}")
    return converted, metadata


def check_correspondence(video_count, motion_count, tolerance):
    difference = abs(video_count - motion_count)
    if difference > tolerance:
        raise ValueError(f"Video/motion frame count difference {difference} exceeds {tolerance}: "
                         f"video={video_count}, motion={motion_count}")
    return {"video_frames": video_count, "motion_frames": motion_count, "difference": difference,
            "max_frame_difference": tolerance}


def reject_degenerate(array, function, args, frame=None):
    data = array if frame is None else array[frame:frame + 1]
    if function == "compute_three_joint_angle":
        a, b, c = [data[:, kin.JOINTS_21[name]] for name in args["joint_triplet"]]
        vectors = [a - b, c - b]
    elif function == "get_joint_angle_to_world_direction":
        vectors = [data[:, kin.JOINTS_21[args["target_joint"]]] -
                   data[:, kin.JOINTS_21[args["source_joint"]]]]
    else:
        vectors = []
    if any((np.linalg.norm(vector, axis=1) < 1e-8).any() for vector in vectors):
        raise ValueError(f"Zero-length vector in {function}; cannot compute an angle")


def describe(call, value):
    function, args = call["function"], call["args"]
    if function == "get_joint_axis_offset":
        target, reference, axis = args["target_joint"], args["reference_joint"], args["axis"]
        positive, negative, axis_text = {
            "UD": ("higher than", "lower than", "vertical axis"),
            "LR": ("to the left of", "to the right of", "left-right axis"),
            "FB": ("in front of", "behind", "front-back axis"),
        }[axis]
        if abs(value) < 1e-6:
            return f"{target} is aligned with {reference} on {axis_text}."
        return f"{target} is {abs(value):.1f} cm {positive if value > 0 else negative} {reference} on {axis_text}."
    if function == "compute_three_joint_angle":
        a, b, c = args["joint_triplet"]
        return f"Angle at {b} ({a}-{b}-{c}) is {value:.1f} degrees."
    if function == "get_pelvis_y_coordinate":
        return f"Pelvis height is {value:.1f} cm."
    world = {"straight_up": "world upward (+Y) direction", "straight_down": "world downward (-Y) direction",
             "straight_forward": "world forward (+Z) direction"}[args["world_direction"]]
    return f"Angle between {args['source_joint']}->{args['target_joint']} and {world} is {value:.1f} degrees."


def execute(array, qf, qk):
    stage_id = qf["stage_id"]
    if type(stage_id) is not int or stage_id < 0:
        raise ValueError(f"Invalid selected stage: {stage_id}")
    if stage_id == 0:
        if qf["call"] is not None:
            raise ValueError("Stage 0 must not contain a localization call")
        frame = 0
    else:
        localizer = validate_call(qf["call"], localizer=True)
        reject_degenerate(array, LOCALIZERS[localizer["function"]], localizer["args"])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            frame = getattr(kin, localizer["function"])(array, **localizer["args"])
    if not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(array):
        raise ValueError(f"Localizer returned an invalid motion index: {frame}")
    measurements = []
    for call in qk:
        validate_call(call)
        reject_degenerate(array, call["function"], call["args"], frame=frame)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value = float(getattr(kin, call["function"])(array, frame_index=int(frame), **call["args"]))
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite measurement from {call['function']}")
        measurements.append({"call": call, "value": value,
                             "unit": "degrees" if "angle" in call["function"] else "cm",
                             "description": describe(call, value)})
    return {"selected_frame": int(frame), "total_frame": len(array),
            "measurements": measurements, "report": "\n".join(m["description"] for m in measurements)}
