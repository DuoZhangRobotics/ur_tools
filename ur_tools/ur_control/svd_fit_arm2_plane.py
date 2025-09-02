import os
import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Compute rigid transform (R, t) that best aligns P to Q using SVD.

    Args:
        P: (N,3) source points
        Q: (N,3) target points

    Returns:
        R: (3,3) rotation matrix
        t: (3,) translation vector
    """
    assert P.shape == Q.shape and P.shape[1] == 3
    # Centroids
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    # Center
    P0 = P - cP
    Q0 = Q - cQ
    # Covariance
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure a proper rotation (determinant +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cQ - R @ cP
    return R, t


def as_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main():
    xs = np.array([-0.1, 0, 0.1])
    ys = np.array([-0.4, -0.5, -0.6])
    xs, ys = np.meshgrid(xs, ys)
    z=0
    # make xs, ys, z a (N, 3) array
    P = np.column_stack((xs.flatten(), ys.flatten(), z * np.ones_like(xs.flatten())))
    here = os.path.dirname(__file__)
    pts_path = os.path.join(here, 'arm2_ee_positions.txt')
    Q = np.loadtxt(pts_path, dtype=float)
    if Q.ndim != 2 or Q.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points in {pts_path}, got shape {Q.shape}")

    R, t = kabsch(P, Q)
    T = as_homogeneous(R, t)

    # Report
    print('Input points (first 3):')
    print(P[:3])
    print('Target points (first 3):')
    print(Q[:3])

    print('\nRotation R:')
    print(R)
    print('\nTranslation t:')
    print(t)
    print('\nHomogeneous transform T:')
    print(T)

    # RMS alignment error
    P_aligned = (R @ P.T).T + t
    err = np.linalg.norm(P_aligned - Q, axis=1)
    print(f"\nRMS error: {np.sqrt(np.mean(err**2)):.6f} m")
    # write this transformation matrix to base2base.txt
    base2base_path = os.path.join(here, 'base2base.txt')
    np.savetxt(base2base_path, T)

if __name__ == '__main__':
    main()
