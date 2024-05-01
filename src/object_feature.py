import numpy as np
import scipy.optimize as optim
from src.utils import homogenize, triangulate_landmarks, proj_err, proj_err_stereo


class ObjFeature:
    def __init__(self, id, Km, position=np.zeros(3), is_initialized=False) -> None:
        self.id = id
        self.Km = Km
        self.position = position
        self.is_initialized = is_initialized
        self.init_frame_id = -1
        self.observations = []

        self.reproj_error_thre = 4

    def __repr__(self):
        return 'id: {}, position: {}, is_initialized: {}'.format(self.id, self.position, self.is_initialized)

    def add_observation(self, frame_id, pt_norm):
        self.observations.append((frame_id, pt_norm))

    def initialize_position(self, cam_ids, cam_poses):
        observations_init, cam_poses_init = [], []
        for i in range(len(self.observations)):
            frame_id, pt_norm = self.observations[i]
            
            if frame_id not in cam_ids:
                continue
            
            observations_init.append(pt_norm)
            cam_pose_idx = np.where(cam_ids == frame_id)[0][0]
            cam_poses_init.append(cam_poses[:, 4*cam_pose_idx:4*cam_pose_idx+4])

        if len(observations_init) < 2:
            return False
        
        p_triang = triangulate_landmarks(observations_init, cam_poses_init)
        # tic = time.time()
        optim_result = optim.least_squares(proj_err, p_triang, loss='linear', method='lm', args=(observations_init, cam_poses_init))
        # toc = time.time()
        # print('optimization time:', toc-tic)
        if not optim_result.success:
            return False

        position_tmp = optim_result.x
        max_reproj_error = 0
        for i in range(len(cam_poses_init)):
            p_cam_homo = np.linalg.inv(cam_poses_init[i]) @ homogenize(position_tmp)
            
            # check if the point is behind the camera
            if p_cam_homo[2] < 0:
                # print('invalid position initialization: point behind the camera')
                return False
            
            # check the reprojection error
            observed_uv = (self.Km @ homogenize(observations_init[i]))[:2]
            # print('observed_uv:', observed_uv)
            projected_uv = (self.Km @ (p_cam_homo[:3] / p_cam_homo[2]))[:2]
            # print('projected_uv:', projected_uv)
            reproj_error = np.linalg.norm(observed_uv - projected_uv)
            if reproj_error > max_reproj_error:
                max_reproj_error = reproj_error
            if max_reproj_error > self.reproj_error_thre:
                # print('invalid position initialization: large reprojection error', max_reproj_error, 'observed_uv:', observed_uv, 'projected_uv:', projected_uv)
                return False

        self.position = position_tmp
        self.is_initialized = True
        return True

class StereoObjFeature:
    def __init__(self, id, Km, oTb, position=np.zeros(3), is_initialized=False):
        self.id = id
        self.Km = Km
        self.oTb = oTb
        self.bTo = np.linalg.inv(oTb)
        self.position = position
        self.is_initialized = is_initialized
        self.init_frame_id = -1
        self.observations = []

        self.reproj_error_thre = 1

    def __repr__(self):
        return 'id: {}, position: {}, is_initialized: {}'.format(self.id, self.position, self.is_initialized)

    def add_observation(self, frame_id, pt_norm):
        self.observations.append((frame_id, pt_norm))

    def initialize_position(self, cam_ids, cam_poses, fsu_b, rel_cam_pose, reproj_error_thre=None):
        # rel_cam_pose: o0To1
        observations_init, cam_poses_init = [], []
        for i in range(len(self.observations)):
            frame_id, pt_norm = self.observations[i]
            
            if frame_id not in cam_ids:
                continue
            
            observations_init.append(pt_norm)
            cam_pose_idx = np.where(cam_ids == frame_id)[0][0]
            cam_poses_init.append(cam_poses[:, 4*cam_pose_idx:4*cam_pose_idx+4])

        if len(observations_init) < 3:
            return False
        
        if reproj_error_thre is not None:
            thre = reproj_error_thre
        else:
            thre = self.reproj_error_thre

        # p_triang = triangulate_landmarks(observations_init, cam_poses_init)
        # use the first stereo observation to initialize the landmark
        left_obs = observations_init[0][:2]
        right_obs = observations_init[0][2:]
        left_cam_pose = cam_poses_init[0]
        # p_norm = np.linalg.inv(self.Km) @ homogenize(left_obs)
        # print('left_obs:', left_obs)
        # print('right_obs:', right_obs)
        # print('fsu_b:', fsu_b)
        z = fsu_b / (left_obs[0] - right_obs[0]) / self.Km[0, 0]
        p_cam0 = z * homogenize(left_obs)
        p_triang = (left_cam_pose @ self.bTo @ homogenize(p_cam0))[:3]
        # print('p_triang:', p_triang)
        # tic = time.time()
        optim_result = optim.least_squares(proj_err_stereo, p_triang, loss='linear', method='lm', args=(observations_init, cam_poses_init, rel_cam_pose, self.bTo))
        if not optim_result.success:
            return False
        position_tmp = optim_result.x
        if np.linalg.norm(position_tmp - p_triang) > 3:
            # print('invalid position initialization: large distance between optimization and triangulation:', np.linalg.norm(position_tmp - p_triang))
            return False
        
        # print('distance between optimization and triangulation:', np.linalg.norm(position_tmp - p_triang))
        # position_tmp = p_triang
        # toc = time.time()
        # print('optimization time:', toc-tic)
        
        max_reproj_error_cam0, max_reproj_error_cam1 = 0, 0
        for i in range(len(cam_poses_init)):
            cam0_pose_init = cam_poses_init[i] @ self.bTo
            p_cam0_homo = np.linalg.inv(cam0_pose_init) @ homogenize(position_tmp)
            p_cam1_homo = np.linalg.inv(cam0_pose_init @ rel_cam_pose) @ homogenize(position_tmp)
            # check if the point is behind the camera
            if p_cam0_homo[2] < 0:
                # print('invalid position initialization: point behind the camera')
                return False
            if p_cam1_homo[2] < 0:
                # print('invalid position initialization: point behind the camera')
                return False
            
            # check the reprojection error
            observed_uv_cam0 = (self.Km @ homogenize(observations_init[i][:2]))[:2]
            # print('observed_uv:', observed_uv)
            projected_uv_cam0 = (self.Km @ (p_cam0_homo[:3] / p_cam0_homo[2]))[:2]
            # print('projected_uv:', projected_uv)
            reproj_error_cam0 = np.linalg.norm(observed_uv_cam0 - projected_uv_cam0)
            max_reproj_error_cam0 = max(reproj_error_cam0, max_reproj_error_cam0)
            
            observed_uv_cam1 = (self.Km @ homogenize(observations_init[i][2:]))[:2]
            projected_uv_cam1 = (self.Km @ (p_cam1_homo[:3] / p_cam1_homo[2]))[:2]
            reproj_error_cam1 = np.linalg.norm(observed_uv_cam1 - projected_uv_cam1)
            max_reproj_error_cam1 = max(reproj_error_cam1, max_reproj_error_cam1)

            if max_reproj_error_cam0 > thre or max_reproj_error_cam1 > thre:
                # print('invalid position initialization: large reprojection error', max_reproj_error, \
                #       'observed_uv:', observed_uv_cam0, 'projected_uv:', projected_uv_cam0)
                return False

        self.position = position_tmp
        self.is_initialized = True
        return True
