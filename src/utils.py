import numpy as np
from scipy.linalg import expm


def homogenize(x):
    if x.ndim == 1:
        return np.append(x, 1)
    elif x.ndim == 2:
        return np.vstack((x, np.ones((1, x.shape[1]))))
    else:
        raise ValueError('Invalid dimension')

def skew_symmetric(v):
    if len(v) == 3:
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    elif len(v) == 6:
        return np.array([[0, -v[5], v[4], v[0]],
                         [v[5], 0, -v[3], v[1]],
                         [-v[4], v[3], 0, v[2]],
                         [0, 0, 0, 0]])
    else:
        raise ValueError('Invalid vector length')

def pi(q):
    if q.ndim == 1:
        return q / q[2]
    elif q.ndim == 2:
        return q / q[2, :]
    else:
        raise ValueError('Invalid dimension')

def dpi_dq(q):
    return 1/q[2] * np.array([[1, 0, -q[0]/q[2], 0],
                              [0, 1, -q[1]/q[2], 0],
                              [0, 0, 0, 0],
                              [0, 0, -q[3]/q[2], 1]])

def circle_dot(s):
    result = np.zeros((4, 6))
    result[:3, 3:] = -skew_symmetric(s[:3])
    result[:3, :3] = np.eye(3)
    
    return result

def SE3_exp(v):
    return expm(skew_symmetric(v))

def SE3_Ad(T):
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = T[:3, :3]
    Ad[3:, 3:] = T[:3, :3]
    Ad[3:, :3] = skew_symmetric(T[:3, 3]) @ T[:3, :3]
    return Ad

def triangulate_landmarks(observations, cam_poses, bTo=np.eye(4)):
    num_frames = len(observations)
    assert num_frames >= 2, 'not enough observations'

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for j in range(num_frames):
        cam_pose = cam_poses[j] @ bTo
        R_cam = cam_pose[:3, :3]
        t_cam = cam_pose[:3, 3]

        feat_norm = observations[j]
        if len(feat_norm) == 2:
            bj = R_cam @ homogenize(feat_norm)
        elif len(feat_norm) == 3:
            bj = R_cam @ feat_norm
        else:
            raise ValueError('Invalid feature length')
        
        Nj = skew_symmetric(bj)
        A += Nj.T @ Nj
        b += Nj.T @ Nj @ t_cam

    p_f = np.linalg.solve(A, b)
    return p_f

def proj_err(x, observations, cam_poses, bTo):
    residuals = []
    for i in range(len(cam_poses)):
        T_cam = cam_poses[i] @ bTo
        x_cam = (np.linalg.inv(T_cam) @ homogenize(x))[:3]
        p_proj = (x_cam / x_cam[2])[:2]
        
        residuals.append(observations[i] - p_proj)
        
    return np.array(residuals).reshape(-1)

def proj_err_stereo(x, observations, cam_poses, rel_pose, bTo):
    residuals = []
    for i in range(len(cam_poses)):
        T_cam0 = cam_poses[i] @ bTo
        x_cam0 = (np.linalg.inv(T_cam0) @ homogenize(x))[:3]
        p_proj0 = (x_cam0 / x_cam0[2])[:2]

        T_cam1 = T_cam0 @ rel_pose
        x_cam1 = (np.linalg.inv(T_cam1) @ homogenize(x))[:3]
        p_proj1 = (x_cam1 / x_cam1[2])[:2]

        residuals.append(observations[i][:2] - p_proj0)
        residuals.append(observations[i][2:] - p_proj1)
    
    return np.array(residuals).reshape(-1)


def read_poses(npz_file):
    arr = np.load(npz_file)
    num_robots = len(arr)
    poses = []
    for i in range(num_robots):
        poses.append(arr['arr_{}'.format(i)])
    return poses
