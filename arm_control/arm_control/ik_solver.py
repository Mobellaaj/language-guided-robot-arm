import math

# Metres: these values match the current robot description.
UPPER_ARM = 0.25
FOREARM = 0.20
SHOULDER_OFFSET = 0.11
SHOULDER_HEIGHT = 0.17
WRIST_TO_TOOL = 0.05

# Order: base, shoulder, elbow, wrist.
JOINT_LIMITS = (
    (-math.pi, math.pi),
    (-math.pi / 2, math.pi / 2),
    (-3 * math.pi / 4, 3 * math.pi / 4),
    (-math.pi / 4, math.pi / 4),
)


def forward_kinematics(angles):
    base, shoulder, elbow, wrist = angles
    pitch = shoulder + elbow + wrist

    radius = (
        SHOULDER_OFFSET
        + UPPER_ARM * math.sin(shoulder)
        + FOREARM * math.sin(shoulder + elbow)
        + WRIST_TO_TOOL * math.sin(pitch)
    )

    return (
        radius * math.cos(base),
        radius * math.sin(base),
        SHOULDER_HEIGHT
        + UPPER_ARM * math.cos(shoulder)
        + FOREARM * math.cos(shoulder + elbow)
        + WRIST_TO_TOOL * math.cos(pitch),
    )


def solve_downward_ik(x, y, z):
    """Return valid joint-angle candidates for tool0 pointing downward."""
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError('Target coordinates must be finite')

    base = math.atan2(y, x)

    # Wrist position relative to the shoulder.
    r = math.hypot(x, y) - SHOULDER_OFFSET
    h = z + WRIST_TO_TOOL - SHOULDER_HEIGHT

    cos_elbow = (
        r*r + h*h - UPPER_ARM**2 - FOREARM**2
    ) / (2 * UPPER_ARM * FOREARM)

    if cos_elbow < -1 - 1e-9 or cos_elbow > 1 + 1e-9:
        raise ValueError('Target is outside the two-link reach')

    # Clamp tiny floating-point overshoots before acos.
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_magnitude = math.acos(cos_elbow)

    solutions = []

    for elbow in (elbow_magnitude, -elbow_magnitude):
        shoulder = math.atan2(r, h) - math.atan2(
            FOREARM * math.sin(elbow),
            UPPER_ARM + FOREARM * math.cos(elbow),
        )

        wrist = math.remainder(
            math.pi - shoulder - elbow, 2 * math.pi
        )

        angles = (base, shoulder, elbow, wrist)

        within_limits = all(
            lower - 1e-9 <= angle <= upper + 1e-9
            for angle, (lower, upper) in zip(angles, JOINT_LIMITS)
        )

        if not within_limits:
            continue

        reconstructed = forward_kinematics(angles)
        if math.dist(reconstructed, (x, y, z)) <= 1e-6:
            solutions.append(angles)

    if not solutions:
        raise ValueError('No downward solution within the joint limits')

    return solutions