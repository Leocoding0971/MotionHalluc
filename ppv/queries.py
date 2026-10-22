"""Strict schemas for text-selected calls. No model-supplied code is evaluated."""

import json

from ppv.kinematics import JOINTS_21


STAGE_ZERO_DESCRIPTION = "Preparation state at the start of the motion; use the first motion frame (index 0)."


def available_stages(inputs):
    return {"0": STAGE_ZERO_DESCRIPTION,
            **{str(i): description for i, description in enumerate(inputs["stage_descriptions"], 1)}}


MEASUREMENTS = {
    "compute_three_joint_angle": {"joint_triplet"},
    "get_joint_axis_offset": {"target_joint", "reference_joint", "axis"},
    "get_pelvis_y_coordinate": set(),
    "get_joint_angle_to_world_direction": {"target_joint", "source_joint", "world_direction"},
}
LOCALIZERS = {
    "locate_three_joint_angle_extreme_frame": "compute_three_joint_angle",
    "locate_joint_axis_offset_extreme_frame": "get_joint_axis_offset",
    "locate_pelvis_y_extreme_frame": "get_pelvis_y_coordinate",
    "locate_joint_angle_to_world_direction_extreme_frame": "get_joint_angle_to_world_direction",
}


def exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")


def parse_json(text):
    # Accept a single JSON code fence, but not prose or duplicate/NaN fields.
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0] not in ("```", "```json"):
            raise ValueError("Expected a JSON code fence")
        text = "\n".join(lines[1:-1])

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def validate_call(call, localizer=False):
    exact_keys(call, {"function", "args"}, "Query")
    function, args = call["function"], call["args"]
    registry = LOCALIZERS if localizer else MEASUREMENTS
    if not isinstance(function, str) or function not in registry:
        raise ValueError(f"Unsupported {'localizer' if localizer else 'measurement'}: {function}")
    base = LOCALIZERS[function] if localizer else function
    expected = MEASUREMENTS[base] | ({"extreme_type"} if localizer else set())
    exact_keys(args, expected, f"{function}.args")
    for name, value in args.items():
        if name == "joint_triplet":
            if not isinstance(value, list) or len(value) != 3 or any(
                    not isinstance(j, str) or j not in JOINTS_21 for j in value) or len(set(value)) != 3:
                raise ValueError("joint_triplet must be three distinct allowed joints")
        elif name.endswith("_joint"):
            if not isinstance(value, str) or value not in JOINTS_21:
                raise ValueError(f"Invalid {name}: {value}")
        else:
            enums = {"axis": ("UD", "LR", "FB"), "extreme_type": ("min", "max"),
                     "world_direction": ("straight_up", "straight_forward", "straight_down")}
            if not isinstance(value, str) or value not in enums[name]:
                raise ValueError(f"Invalid {name}: {value}; allowed: {enums[name]}")
    for pair in (("target_joint", "reference_joint"), ("target_joint", "source_joint")):
        if all(key in args for key in pair) and args[pair[0]] == args[pair[1]]:
            raise ValueError(f"{pair[0]} and {pair[1]} must be distinct")
    return call


def validate_qf(document, inputs):
    action = inputs["action_name"]
    exact_keys(document, {action}, "Qf output")
    stages = document[action]
    if not isinstance(stages, dict) or len(stages) != 1:
        raise ValueError("Qf must select exactly one stage")
    stage = next(iter(stages))
    allowed = available_stages(inputs)
    if stage not in allowed:
        raise ValueError(f"Stage must be one of {list(allowed)}; got {stage}")
    if stage == "0":
        if stages[stage] is not None:
            raise ValueError("Stage 0 must be null: it uses the first motion frame without a localizer")
        call = None
    else:
        call = validate_call(stages[stage], localizer=True)
    return {"stage_id": int(stage), "stage_description": allowed[stage], "call": call}


def validate_qk(document, inputs):
    action, correction = inputs["action_name"], inputs["correction"]
    exact_keys(document, {action}, "Qk output")
    exact_keys(document[action], {correction}, "Qk corrections")
    entry = document[action][correction]
    exact_keys(entry, {"calls"}, "Qk correction")
    calls = entry["calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("Qk calls must be a nonempty list")
    return [validate_call(call) for call in calls]


def validate_verdict(document):
    exact_keys(document, {"reasoning", "result"}, "Verification output")
    if not isinstance(document["reasoning"], str) or not document["reasoning"].strip():
        raise ValueError("reasoning must be a nonempty string")
    if not isinstance(document["result"], str) or document["result"].lower().strip() not in ("yes", "no"):
        raise ValueError("result must be yes or no")
    return {"reasoning": document["reasoning"].strip(), "result": document["result"].lower().strip()}
