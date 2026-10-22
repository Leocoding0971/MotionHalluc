"""Numerical functions copied from method/Motion_CT/check_atom.py.

The pipeline performs strict data and degeneracy checks before these functions.
"""

import numpy as np
from typing import List

JOINTS_21 = {'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'L_Knee': 3, 'R_Knee': 4, 'L_Ankle': 5, 'R_Ankle': 6, 'L_Foot': 7, 'R_Foot': 8, 'Neck': 9, 'L_Collar': 10, 'R_Collar': 11, 'Head': 12, 'L_Shoulder': 13, 'R_Shoulder': 14, 'L_Elbow': 15, 'R_Elbow': 16, 'L_Wrist': 17, 'R_Wrist': 18, 'Spine': 19, 'Chest': 20}

def augment_mocap_to_21(
    mocap: np.ndarray,           # shape: (T, 22, 3)  原始22关节数据
    original_names: list = None
) -> tuple[np.ndarray, list]:
    """
    将原始22关节 MoCap 转换为优化后的21关节版本
    返回: (T, 21, 3), 新关节名称列表
    """
    if mocap.shape[1] != 22:
        raise ValueError(f"Expected mocap with 22 joints, but got {mocap.shape[1]} joints.")

    if original_names is None:
        original_names = [
            "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", 
            "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", 
            "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", 
            "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
        ]
    
    idx = {name: original_names.index(name) for name in original_names}
    
    # 计算新关节
    spine = mocap[:, idx["Spine3"]].copy()                    # 使用 Spine3 作为单一 Spine
    chest = (mocap[:, idx["L_Collar"]] + mocap[:, idx["R_Collar"]]) / 2.0   # Chest = 左右锁骨中点
    
    # 构造新 MoCap：保留非Spine的原始关节 + Spine + Chest
    kept_indices = [i for i, name in enumerate(original_names) 
                   if name not in ["Spine1", "Spine2", "Spine3"]]
    
    new_mocap = np.concatenate([
        mocap[:, kept_indices, :],   # 保留的19个原始关节
        spine[:, None, :],           # Spine (第19位)
        chest[:, None, :]            # Chest (第20位)
    ], axis=1)
    
    new_names = [original_names[i] for i in kept_indices] + ["Spine", "Chest"]
    
    return new_mocap, new_names

def compute_three_joint_angle(motion_data, joint_triplet: List[str], frame_index: int) -> float:
    """
    计算三个关节在中间关节处形成的夹角（单位：度）。

    Args:
        motion_data: 3D 关节位置数据，形状通常为 (T, J, 3)。
        joint_triplet: 长度必须为 3 的关节名列表，格式为
            [start_joint, center_joint, end_joint]。
        frame_index: 要查询的帧索引。

    Returns:
        float: 夹角大小（0-180 度）。

    说明:
        1. 取出 start、center、end 三个关节坐标。
        2. 构造向量 center->start 与 center->end。
        3. 通过点积反余弦计算夹角。
    """
    assert len(joint_triplet) == 3, "joint_triplet must contain exactly 3 joints"
    
    # Get joint indices
    start_joint, center_joint, end_joint = joint_triplet
    assert start_joint != center_joint != end_joint, "Joints must be distinct"
    try:
        start_idx = JOINTS_21[start_joint]
        center_idx = JOINTS_21[center_joint]
        end_idx = JOINTS_21[end_joint]
    except KeyError as e:
        raise ValueError(f"Invalid joint_name: {e.args[0]}. Please check JOINTS_21 mapping.")
    
    # Get 3D coordinates at the specified frame
    pos_A = motion_data[frame_index, start_idx, :]
    pos_B = motion_data[frame_index, center_idx, :]
    pos_C = motion_data[frame_index, end_idx, :]
    
    # Create vectors BA and BC
    vec_BA = pos_A - pos_B
    vec_BC = pos_C - pos_B
    
    # Normalize vectors
    norm_BA = np.linalg.norm(vec_BA)
    norm_BC = np.linalg.norm(vec_BC)
    
    # Handle zero-length vectors
    if norm_BA < 1e-8 or norm_BC < 1e-8:
        return 0.0
    
    vec_BA_unit = vec_BA / norm_BA
    vec_BC_unit = vec_BC / norm_BC
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(vec_BA_unit, vec_BC_unit), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)

def get_joint_axis_offset(motion_data, target_joint: str, reference_joint: str, frame_index: int, axis: str) -> float:
    """
    计算 target_joint 相对 reference_joint 在指定轴向上的位移。

    Args:
        motion_data: 3D 关节位置数据，形状通常为 (T, J, 3)。
        target_joint: 目标关节名（被测量关节）。
        reference_joint: 参考关节名（作为相对基准）。
        frame_index: 要查询的帧索引。
        axis: 轴向标识，支持 "UD"(Y)、"LR"(X)、"FB"(Z)。

    Returns:
        float: target_joint - reference_joint 在对应轴上的坐标差。
    """
    try:
        target_idx = JOINTS_21[target_joint]
        reference_idx = JOINTS_21[reference_joint]
    except KeyError as e:
        raise ValueError(f"Invalid joint_name: {e.args[0]}. Please check JOINTS_21 mapping.")
    
    pos_A = motion_data[frame_index, target_idx, :]
    pos_B = motion_data[frame_index, reference_idx, :]

    if axis == "UD":
        return (pos_A[1] - pos_B[1]) * 100.0  # 转换为厘米
    elif axis == "LR":
        return (pos_A[0] - pos_B[0]) * 100.0  # 转换为厘米
    elif axis == "FB":
        return (pos_A[2] - pos_B[2]) * 100.0  # 转换为厘米
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be one of 'UD', 'LR', 'FB'.")

def get_pelvis_y_coordinate(motion_data, frame_index: int) -> float:
    """
    获取指定帧中 Pelvis 关节的 Y 轴坐标。

    Args:
        motion_data: 3D 关节位置数据，形状通常为 (T, J, 3)。
        frame_index: 要查询的帧索引。

    Returns:
        float: Pelvis 的 Y 坐标值。
    """
    idx_pelvis = JOINTS_21["Pelvis"]
    pos_pelvis = motion_data[frame_index, idx_pelvis, :]
    return float(pos_pelvis[1]) * 100.0

def get_joint_angle_to_world_direction(motion_data, target_joint: str, source_joint: str, frame_index: int, world_direction: str) -> float:
    """
    计算骨向量 source_joint -> target_joint 与世界坐标固定方向的夹角（单位：度）。

    Args:
        motion_data: 3D 关节位置数据，形状通常为 (T, J, 3)。
        target_joint: 目标关节名（骨向量终点）。
        source_joint: 源关节名（骨向量起点）。
        frame_index: 要查询的帧索引。
        world_direction: 世界坐标中的参考方向，支持
            "straight_up"、"straight_forward"、"straight_down"。

    Returns:
        float: 夹角大小（0-180 度）。
    """
    try:
        target_idx = JOINTS_21[target_joint]
        source_idx = JOINTS_21[source_joint]
    except KeyError as e:
        raise ValueError(f"Invalid joint_name: {e.args[0]}. Please check JOINTS_21 mapping.")
    
    # Get 3D coordinates at the specified frame
    pos_A = motion_data[frame_index, target_idx, :]
    pos_B = motion_data[frame_index, source_idx, :]

    if world_direction == "straight_up":
        pos_C = pos_B + np.array([0, 1, 0])  # Up direction in world space
    elif world_direction == "straight_forward":
        pos_C = pos_B + np.array([0, 0, 1])  # Forward direction in world space
    elif world_direction == "straight_down":
        pos_C = pos_B + np.array([0, -1, 0]) # Down direction in world space
    else:
        raise ValueError(f"Invalid world_direction: {world_direction}. Must be one of 'straight_up', 'straight_forward', 'straight_down'.")
    
    # Create vectors BA and BC
    vec_BA = pos_A - pos_B
    vec_BC = pos_C - pos_B
    
    
    # Normalize vectors
    norm_BA = np.linalg.norm(vec_BA)
    norm_BC = np.linalg.norm(vec_BC)
    
    # Handle zero-length vectors
    if norm_BA < 1e-8 or norm_BC < 1e-8:
        return 0.0
    
    vec_BA_unit = vec_BA / norm_BA
    vec_BC_unit = vec_BC / norm_BC
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(vec_BA_unit, vec_BC_unit), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)

def _ensure_motion_has_21_joints(motion_data: np.ndarray) -> np.ndarray:
    """将输入统一为 (T, 21, 3)，便于后续按 JOINTS_21 索引。"""
    if not isinstance(motion_data, np.ndarray):
        raise ValueError(f"motion_data must be np.ndarray, got {type(motion_data)}.")

    if motion_data.ndim != 3 or motion_data.shape[-1] != 3:
        raise ValueError(f"motion_data must have shape (T, J, 3), got {motion_data.shape}.")

    if motion_data.shape[1] == 21:
        return motion_data
    if motion_data.shape[1] == 22:
        motion_21, _ = augment_mocap_to_21(motion_data)
        return motion_21

    raise ValueError(f"motion_data joint number must be 21 or 22, got {motion_data.shape[1]}.")

def _select_extreme_frame(values: List[float], extreme_type: str) -> int:
    """根据 extreme_type 返回极值对应的帧索引（0-based）。"""
    if extreme_type not in {"max", "min"}:
        raise ValueError(f"Invalid extreme_type: {extreme_type}. Must be 'max' or 'min'.")
    if len(values) == 0:
        raise ValueError("values must not be empty.")

    metric = np.asarray(values, dtype=np.float64)
    if extreme_type == "max":
        return int(np.argmax(metric))
    return int(np.argmin(metric))

def locate_three_joint_angle_extreme_frame(
    motion_data: np.ndarray,
    joint_triplet: List[str],
    extreme_type: str,
) -> int:
    motion_21 = _ensure_motion_has_21_joints(motion_data)
    frame_metrics = [
        compute_three_joint_angle(motion_21, joint_triplet, f)
        for f in range(motion_21.shape[0])
    ]
    return _select_extreme_frame(frame_metrics, extreme_type)

def locate_joint_axis_offset_extreme_frame(
    motion_data: np.ndarray,
    target_joint: str,
    reference_joint: str,
    axis: str,
    extreme_type: str,
) -> int:
    motion_21 = _ensure_motion_has_21_joints(motion_data)
    frame_metrics = [
        get_joint_axis_offset(motion_21, target_joint, reference_joint, f, axis)
        for f in range(motion_21.shape[0])
    ]
    return _select_extreme_frame(frame_metrics, extreme_type)

def locate_pelvis_y_extreme_frame(
    motion_data: np.ndarray,
    extreme_type: str,
) -> int:
    motion_21 = _ensure_motion_has_21_joints(motion_data)
    frame_metrics = [
        get_pelvis_y_coordinate(motion_21, f)
        for f in range(motion_21.shape[0])
    ]
    return _select_extreme_frame(frame_metrics, extreme_type)

def locate_joint_angle_to_world_direction_extreme_frame(
    motion_data: np.ndarray,
    target_joint: str,
    source_joint: str,
    world_direction: str,
    extreme_type: str,
) -> int:
    motion_21 = _ensure_motion_has_21_joints(motion_data)
    frame_metrics = [
        get_joint_angle_to_world_direction(motion_21, target_joint, source_joint, f, world_direction)
        for f in range(motion_21.shape[0])
    ]
    return _select_extreme_frame(frame_metrics, extreme_type)
