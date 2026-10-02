"""Minimal BVH reader + forward kinematics (numpy / scipy, no GPU)."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_bvh(source, max_frames=None):
    """
    source: path to a .bvh file, or the file's contents as bytes.
    max_frames: only read the first N frames (fast peek at a pose).
    Returns dict with
      names[J], parents[J] (-1 root), offsets[J,3] (file units), fps,
      trans[T,J,3], has_trans[J]  per-joint translation channels (file units);
          a joint with position channels is placed by them instead of its OFFSET
          (LAFAN1: the root; SOMA: Root and Hips),
      pos[T,3]  translation of the first joint that has position channels,
      rot[T,J,3,3]  local rotation matrices.
    Euler channels are read per joint, so any channel order works. A BVH rotation
    with channels (Zrot Yrot Xrot) is Rz @ Ry @ Rx, i.e. scipy intrinsic 'ZYX'.
    """
    if isinstance(source, str):
        with open(source) as f:
            source = f.read()
    else:
        source = source.decode()
    toks = source.split()
    assert toks[0] == "HIERARCHY"
    names, parents, offsets, chans = [], [], [], []
    stack, cur, i = [], -1, 1
    while toks[i] != "MOTION":
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            names.append(toks[i + 1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
            chans.append(None)
            cur = len(names) - 1
            i += 2
        elif t == "End":  # End Site { OFFSET x y z }
            i += 8
        elif t == "{":
            stack.append(cur)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            offsets[cur] = [float(x) for x in toks[i + 1 : i + 4]]
            i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1])
            chans[cur] = toks[i + 2 : i + 2 + n]
            i += 2 + n
        else:
            raise ValueError(f"unexpected token {t!r} in BVH header")

    # MOTION block
    assert toks[i] == "MOTION" and toks[i + 1] == "Frames:"
    T = int(toks[i + 2])
    assert toks[i + 3] == "Frame" and toks[i + 4] == "Time:"
    dt = float(toks[i + 5])
    n_chan = sum(len(c) for c in chans)
    if max_frames is not None:
        T = min(T, max_frames)
    data = np.asarray(toks[i + 6 : i + 6 + T * n_chan], dtype=np.float64).reshape(T, n_chan)

    J = len(names)
    rot = np.tile(np.eye(3), (T, J, 1, 1))
    trans = np.zeros((T, J, 3))
    has_trans = np.zeros(J, dtype=bool)
    col = 0
    for j in range(J):
        ch = chans[j]
        vals = data[:, col : col + len(ch)]
        col += len(ch)
        p_idx = [k for k, c in enumerate(ch) if c.endswith("position")]
        r_idx = [k for k, c in enumerate(ch) if c.endswith("rotation")]
        if p_idx:  # translation, reordered to X, Y, Z
            trans[:, j] = np.stack([vals[:, [k for k in p_idx if ch[k][0] == a][0]] for a in "XYZ"], axis=1)
            has_trans[j] = True
        if r_idx:
            order = "".join(ch[k][0] for k in r_idx)
            rot[:, j] = R.from_euler(order, vals[:, r_idx], degrees=True).as_matrix()

    first = int(np.argmax(has_trans)) if has_trans.any() else 0
    return {
        "names": names,
        "parents": parents,
        "offsets": np.asarray(offsets, dtype=np.float64),
        "fps": 1.0 / dt,
        "trans": trans,
        "has_trans": has_trans,
        "pos": trans[:, first],
        "rot": rot,
    }


def bvh_fk(parents, offsets, trans, has_trans, local_rot):
    """
    Global rotations [T,J,3,3] and positions [T,J,3] (same units as offsets/trans).
    BVH: child position = parent position + parent global rotation @ child translation,
    where the translation is the OFFSET, or the per-frame channels if the joint has any.
    """
    T, J = local_rot.shape[:2]
    G = np.empty((T, J, 3, 3))
    P = np.empty((T, J, 3))
    for j in range(J):
        p = parents[j]
        t = trans[:, j] if has_trans[j] else np.broadcast_to(offsets[j], (T, 3))
        if p < 0:
            G[:, j] = local_rot[:, j]
            P[:, j] = t
        else:
            G[:, j] = G[:, p] @ local_rot[:, j]
            P[:, j] = P[:, p] + np.einsum("tij,tj->ti", G[:, p], t)
    return G, P
