import numpy as np

def cosine_similarity_vector(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    벡터 간 코사인 유사도를 계산하여 반환합니다.
    cos_sim = (vec1 · vec2) / (||vec1|| * ||vec2||)
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def cosine_similarity_scalar(angle1: float, angle2: float) -> float:
    """
    두 스칼라 각도(angle1, angle2)에 대해 “각도 차이의 코사인”을 반환합니다.
    즉,
      diff_rad = (angle1 - angle2) * π/180
      cos_sim = cos(diff_rad)
    통상 두 각도가 동일하면 1.0, 90° 차이 나면 0.0, 180° 차이 나면 -1.0이 됩니다.
    """
    diff_rad = (angle1 - angle2) * np.pi / 180.0
    return np.cos(diff_rad)

def assign_grade_from_similarity(sim: float) -> str:
    """
    코사인 유사도(sim)에 따라 등급을 반환합니다.
    예시 기준:
      - sim ≥ 0.98 : "A"
      - 0.95 ≤ sim < 0.98 : "B"
      - 0.90 ≤ sim < 0.95 : "C"
      - 0.80 ≤ sim < 0.90 : "D"
      - sim < 0.80 : "F"
    필요하다면 임계값을 자유롭게 조정하세요.
    """
    if sim >= 0.98:
        return "A"
    elif sim >= 0.95:
        return "B"
    elif sim >= 0.90:
        return "C"
    elif sim >= 0.80:
        return "D"
    else:
        return "F"

def grade_per_joint_and_overall(
    user_angles: dict,
    reference_angles: dict,
    joint_order: list
) -> dict:
    """
    사용자 각도(user_angles)와 기준 각도(reference_angles)를 받아,
    1) joint_order 순서대로 전체 벡터의 코사인 유사도 및 등급을 구함
    2) 각 관절별 (스칼라) 코사인 유사도를 구하고, 그에 따른 등급을 매김
    
    Args:
      user_angles      : {"left_elbow": 102.50, "right_elbow": 127.22, ..., "right_hip": 74.43}
      reference_angles : {"left_elbow":  67.05, "right_elbow":  76.12, ..., "right_hip": 61.52}
      joint_order      : ["left_elbow", "right_elbow", "left_knee", ..., "right_hip"]
    
    Returns:
      {
        "overall": {
            "cosine_similarity": <float>,
            "grade": "<A/B/C/D/F>"
        },
        "per_joint": {
            "left_elbow":   {"cosine": <float>, "grade": "<A/B/C/D/F>"},
            "right_elbow":  {"cosine": <float>, "grade": "<A/B/C/D/F>"},
            ...
        }
      }
    """
    # 1) 전체 벡터 생성
    user_vec = np.array([user_angles[j] for j in joint_order], dtype=np.float32)
    ref_vec  = np.array([reference_angles[j] for j in joint_order], dtype=np.float32)
    
    # (옵션) 벡터 정규화: 각도 그대로 사용하여도 되지만, 필요 시 180으로 나누는 등 정규화 가능
    # user_vec_norm = user_vec / 180.0
    # ref_vec_norm  = ref_vec  / 180.0
    # 여기서는 원본 벡터 그대로 사용
    cos_overall = cosine_similarity_vector(user_vec, ref_vec)
    grade_overall = assign_grade_from_similarity(cos_overall)
    
    # 2) 각 관절별 스칼라 코사인 유사도 계산 & 등급
    per_joint_results = {}
    for joint in joint_order:
        ua = user_angles[joint]
        ra = reference_angles[joint]
        cos_j = cosine_similarity_scalar(ua, ra)
        grade_j = assign_grade_from_similarity(cos_j)
        per_joint_results[joint] = {
            "cosine": float(cos_j),
            "grade":  grade_j
        }
    
    return {
        "overall": {
            "cosine_similarity": float(cos_overall),
            "grade": grade_overall
        },
        "per_joint": per_joint_results
    }


if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────
    # 예시: 사용자 각도와 기준 각도를 하드코딩
    # ─────────────────────────────────────────────────────────────────────────
    user_angles = {
        "left_elbow":   102.50,
        "right_elbow":  127.22,
        "left_knee":     79.38,
        "right_knee":    75.22,
        "left_shoulder": 73.46,
        "right_shoulder":76.34,
        "left_hip":      78.02,
        "right_hip":     74.43
    }

    reference_angles = {
        "left_elbow":   67.05,
        "right_elbow":  76.12,
        "left_knee":    75.47,
        "right_knee":   75.44,
        "left_shoulder":95.10,
        "right_shoulder":93.27,
        "left_hip":     58.77,
        "right_hip":    61.52
    }

    joint_order = [
        "left_elbow", "right_elbow",
        "left_knee",  "right_knee",
        "left_shoulder", "right_shoulder",
        "left_hip",   "right_hip"
    ]

    results = grade_per_joint_and_overall(user_angles, reference_angles, joint_order)

    # ─────────────────────────────────────────────────────────────────────────
    # 출력 예시
    # ─────────────────────────────────────────────────────────────────────────
    print("==== 전체 결과 ====")
    print(f"코사인 유사도 (Overall) : {results['overall']['cosine_similarity']:.4f}")
    print(f"전체 등급            : {results['overall']['grade']}\n")

    print("==== 관절별 결과 ====")
    for joint in joint_order:
        cj = results['per_joint'][joint]["cosine"]
        gj = results['per_joint'][joint]["grade"]
        print(f"{joint:14s} → cos={cj:+.4f}, grade={gj}")
