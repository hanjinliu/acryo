from types import SimpleNamespace


class Mole(SimpleNamespace):
    """Feature header names for Molecules."""

    pf = "molecules-pf"
    isotype = "molecules-isotype"
    zncc = "molecules-zncc"
    pcc = "molecules-pcc"
    interval = "molecules-interval"


class Align(SimpleNamespace):
    zShift = "shift-z"
    yShift = "shift-y"
    xShift = "shift-x"
    zRotvec = "rotvec-z"
    yRotvec = "rotvec-y"
    xRotvec = "rotvec-x"


class EulerAxes(SimpleNamespace):
    """Sequence of Euler angles."""

    xyz = "xyz"
    yzx = "yzx"
    zxy = "zxy"
    xzy = "xzy"
    yxz = "yxz"
    zyx = "zyx"
    xyx = "xyx"
    xzx = "xzx"
    yxy = "yxy"
    yzy = "yzy"
    zxz = "zxz"
    zyz = "zyz"
    XYZ = "XYZ"
    YZX = "YZX"
    ZXY = "ZXY"
    XZY = "XZY"
    YXZ = "YXZ"
    ZYX = "ZYX"
    XYX = "XYX"
    XZX = "XZX"
    YXY = "YXY"
    YZY = "YZY"
    ZXZ = "ZXZ"
    ZYZ = "ZYZ"
