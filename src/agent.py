import cv2
import numpy as np
from src.utils import SE3_Ad, SE3_exp, dpi_dq, homogenize, circle_dot
from scipy.linalg import null_space
from scipy.stats import chi2
from src.object_feature import ObjFeature, StereoObjFeature
from src.geometric_feature import GeoFeature, StereoGeoFeature


class State:
    def __init__(self, max_num_cam=20, cam_pose_init=None):
        self.max_num_cam = max_num_cam
        # pose mean and landmark mean
        self.cam_ids = np.array([0])
        if cam_pose_init is None:
            self.cam_poses = np.eye(4)
        else:
            self.cam_poses = cam_pose_init
        self.object_ids = np.zeros(0)
        self.objects = np.zeros(0)
        self.object_init_frames = np.zeros(0)
        # state covariance
        self.cov = np.eye(6) * 0.001

    def num_cameras(self):
        return len(self.cam_ids)
    
    def num_objects(self):
        return len(self.object_ids)
    
    def augment(self, rel_pose, cam_id, W):
        assert self.num_cameras() < self.max_num_cam, 'maximum number of cameras reached'
        assert cam_id == self.cam_ids[-1] + 1, 'invalid camera id'
        
        n = self.num_cameras()
        m = self.num_objects()
        rel_pose_inv = np.linalg.inv(rel_pose)

        # pose mean expansion
        self.cam_ids = np.append(self.cam_ids, cam_id)
        self.cam_poses = np.hstack((self.cam_poses, self.cam_poses[:, -4:] @ rel_pose))
        # covariance expansion
        A = np.zeros((6*n+6+3*m, 6*n+3*m))
        A[:6*n, :6*n] = np.eye(6*n)
        A[6*n:6*n+6, 6*n-6:6*n] = SE3_Ad(rel_pose_inv)
        A[6*n+6:, 6*n:] = np.eye(3*m)

        W_aug = np.zeros((6*n+6+3*m, 6*n+6+3*m))
        W_aug[6*n:6*n+6, 6*n:6*n+6] = W

        self.cov = A @ self.cov @ A.T + W_aug

    def propagate(self, rel_pose, cam_id, W):
        # propagate by removing the oldest camera pose and add the latest one
        assert self.num_cameras() == self.max_num_cam, 'incorrect number of cameras'
        assert cam_id == self.cam_ids[-1] + 1, 'invalid camera id'

        n = self.num_cameras()
        m = self.num_objects()
        
        rel_pose_inv = np.linalg.inv(rel_pose)
        adjoint_of_inv = SE3_Ad(rel_pose_inv)

        # pose mean propagation
        self.cam_ids += 1
        F = np.zeros((4*n, 4*n))
        F[4:, :4*n-4] = np.eye(4*n-4)
        F[-4:, -4:] = rel_pose
        self.cam_poses = self.cam_poses @ F
        # covariance propagation
        A = np.zeros((6*n+3*m, 6*n+3*m))
        for i in range(n-1):
            # A[6*i:6*i+6, 6*i:6*i+6] = adjoint_of_inv
            A[6*i:6*i+6, 6*i+6:6*i+12] = np.eye(6)
        A[6*n-6:6*n, 6*n-6:6*n] = adjoint_of_inv
        A[6*n:, 6*n:] = np.eye(3*m)
        
        W_aug = np.zeros((6*n+3*m, 6*n+3*m))
        W_aug[6*n-6:6*n, 6*n-6:6*n] = W

        self.cov = A @ self.cov @ A.T + W_aug


class DSLAM:
    def __init__(self, robot_id, params, cam_pose_init=None):
        
        self.robot_id = robot_id
        self.oTb = params['opt_T_b']
        self.bTo = np.linalg.inv(self.oTb)
        self.Km = params['K_mono']
        self.Km_inv = np.linalg.inv(self.Km)
        self.max_num_cam = params['max_num_cam']
        self.common_mu_thre = params['common_mu_thre']
        self.o0To1 = params['o0To1']
        self.fsu_b = params['fsu_b']
        self.stereo_obj = params['stereo_obj']
        self.stereo_msckf = params['stereo_msckf']
        self.msckf_feat_reproj_thre = params['msckf_feat_reproj_thre']
        self.geo_motion_check = params['geo_motion_check']
        self.obj_init_reproj_thre = params['obj_init_reproj_thre']
        self.obj_init_dist_min_thre = params['obj_init_dist_min_thre']
        self.obj_init_dist_max_thre = params['obj_init_dist_max_thre']

        self.obj_update_reproj_thre = params['obj_update_reproj_thre']
        self.update_with_qr = params['update_with_qr']

        if self.stereo_obj or self.stereo_msckf:
            assert self.o0To1 is not None and self.fsu_b is not None, \
                'stereo relative camera pose and fsu*b not provided'
            self.o1To0 = np.linalg.inv(self.o0To1)

        self.motion_noise = params['motion_noise']
        self.geo_obs_noise = params['geo_obs_noise']
        self.obj_obs_noise = params['obj_obs_noise']

        self.pose_fuse_update_thre = float(params['pose_fuse_update_thre'])
        self.obj_fuse_update_thre = params['obj_fuse_update_thre']

        self.rel_poses = []
        self.geo_feats_dict = {}
        self.uninit_objs_dict = {}
        self.state = State(self.max_num_cam, cam_pose_init)
        self.P = np.zeros((3, 4))
        self.P[:3, :3] = np.eye(3)

        self.msckf_vis_fnum_thre = params['msckf_vis_fnum_thre']
        self.obj_init_fnum_thre = params['obj_init_fnum_thre']
        self.timestep = 0

        self.W = np.eye(6) * self.motion_noise # motion noise
        self.V_geo = np.eye(2) * self.geo_obs_noise # geometric observation noise
        self.V_obj = np.eye(2) * self.obj_obs_noise

        # initialize the chi squared test table with confidence level 0.95
        self.chi_squared_table = {}
        for i in range(200):
            self.chi_squared_table[i] = chi2.ppf(0.05, i+1)
        
        self.num_update_chance = 0
        self.num_update = 0
        self.num_msckf_update = 0
        self.num_fuse = 0

    def predict(self, rel_pose, cam_id):
        if len(self.rel_poses) < self.max_num_cam:
            self.rel_poses.append(rel_pose)
        else:
            self.rel_poses.pop(0)
            self.rel_poses.append(rel_pose)

        if self.state.num_cameras() < self.max_num_cam:
            self.state.augment(self.rel_poses[-1], cam_id, self.W)
        else:
            self.state.propagate(self.rel_poses[-1], cam_id, self.W)
        self.timestep += 1
    
    def current_cam_pose(self):
        return self.state.cam_poses[:, -4:]
    
    def msckf_feature_jacobian(self, feat_id):
        n, m = self.state.num_cameras(), self.state.num_objects()

        geo_feat = self.geo_feats_dict[feat_id]
        valid_cam_ids = []
        valid_observations = []
        for cam_id in self.state.cam_ids:
            for frame_id, pt_norm in geo_feat.observations:
                if frame_id == cam_id:
                    valid_cam_ids.append(cam_id)
                    valid_observations.append(pt_norm)
                    break
        assert len(valid_cam_ids) == len(valid_observations)
        if len(valid_cam_ids) < self.msckf_vis_fnum_thre:
            return None, None
        
        jacobian_row_size = 4 * len(valid_cam_ids) if self.stereo_msckf else 2 * len(valid_cam_ids)
        H_xj = np.zeros((jacobian_row_size, 6*n+3*m))
        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_cntr = 0
        for i in range(len(valid_cam_ids)):
            cam_id, obs = valid_cam_ids[i], valid_observations[i]
            cam_state_idx = np.where(self.state.cam_ids == cam_id)[0][0]
            cam0_pose = self.state.cam_poses[:, 4*cam_state_idx:4*cam_state_idx+4]
            cam0_pose_inv = np.linalg.inv(cam0_pose)
            landm_cam_homo = cam0_pose_inv @ homogenize(geo_feat.position)

            H_fi_cam0 = (dpi_dq(self.oTb @ landm_cam_homo) @ self.oTb @ cam0_pose_inv @ self.P.T)[:2, :]
            H_fj[stack_cntr:stack_cntr+2, :] = H_fi_cam0
            H_xi_cam0 = -(dpi_dq(self.oTb @ landm_cam_homo) @ self.oTb @ circle_dot(landm_cam_homo))[:2, :]
            H_xj[stack_cntr:stack_cntr+2, 6*cam_state_idx:6*cam_state_idx+6] = H_xi_cam0
            
            z_hat_left = self.oTb @ landm_cam_homo
            z_hat_left = (z_hat_left / z_hat_left[2])[:2]
            r_j[stack_cntr:stack_cntr+2] = obs[:2] - z_hat_left

            if self.stereo_msckf:
                H_fi_cam1 = (dpi_dq(self.o1To0 @ self.oTb @ landm_cam_homo) @ self.o1To0 @ self.oTb @ cam0_pose_inv @ self.P.T)[:2, :]
                H_fj[stack_cntr+2:stack_cntr+4, :] = H_fi_cam1
                H_xi_cam1 = -(dpi_dq(self.o1To0 @ self.oTb @ landm_cam_homo) @ self.o1To0 @ self.oTb @ circle_dot(landm_cam_homo))[:2, :]
                H_xj[stack_cntr+2:stack_cntr+4, 6*cam_state_idx:6*cam_state_idx+6] = H_xi_cam1

                z_hat_right = self.o1To0 @ self.oTb @ landm_cam_homo
                z_hat_right = (z_hat_right / z_hat_right[2])[:2]
                r_j[stack_cntr+2:stack_cntr+4] = obs[2:] - z_hat_right

            stack_cntr += 4 if self.stereo_msckf else 2

        ########### nullspace projection ############
        left_null = null_space(H_fj.T)
        return left_null.T @ H_xj, left_null.T @ r_j
    
    def add_msckf_features(self, cam_id, geo_feats):
        for gfeat_id in geo_feats:
            
            if gfeat_id not in self.geo_feats_dict:
                if self.stereo_msckf:
                    self.geo_feats_dict[gfeat_id] = StereoGeoFeature(gfeat_id, self.Km, self.oTb)
                else:
                    self.geo_feats_dict[gfeat_id] = GeoFeature(gfeat_id, self.Km, self.oTb)
            geo_feat = self.geo_feats_dict[gfeat_id]
            # print('bp1: ', len(geo_feat.observations))
            geo_feat.add_observation(cam_id, geo_feats[gfeat_id])
            # print('bp2: ', len(geo_feat.observations))

    def remove_lost_msckf_features(self):
        # print('removing lost msckf features')
        n, m = self.state.num_cameras(), self.state.num_objects()
        jacobian_row_size = 0

        invalid_feat_ids = []
        processed_feat_ids = []
        for gfeat_id in self.geo_feats_dict:
            
            geo_feat = self.geo_feats_dict[gfeat_id]
            assert gfeat_id == geo_feat.id, "geo feature id does not match"

            # pass the features that are still being tracked
            if geo_feat.observations[-1][0] == self.timestep:
                # print('the feature is still being tracked')
                continue
            
            if len(geo_feat.observations) < self.msckf_vis_fnum_thre:
                # print('invalid due to not enough observations:', len(geo_feat.observations))
                invalid_feat_ids.append(gfeat_id)
                continue
            
            if not geo_feat.is_initialized:
                if not geo_feat.check_motion(self.state.cam_ids, self.state.cam_poses):
                    if self.geo_motion_check:
                        # print('invalid due to failed motion check')
                        invalid_feat_ids.append(gfeat_id)
                        continue
                    
                
                # initialize the feature position
                if self.stereo_msckf:
                    if not geo_feat.initialize_position(self.state.cam_ids, self.state.cam_poses, 
                                                        self.fsu_b, self.o0To1, self.msckf_feat_reproj_thre):
                        # print('invalid due to failed initialization')
                        invalid_feat_ids.append(gfeat_id)
                        continue
                else:
                    if not geo_feat.initialize_position(self.state.cam_ids, self.state.cam_poses):
                        # print('invalid due to failed initialization')
                        invalid_feat_ids.append(gfeat_id)
                        continue
                    
            processed_feat_ids.append(gfeat_id)
            jacobian_row_size += 4 * len(geo_feat.observations) - 3 if self.stereo_msckf else \
                  2 * len(geo_feat.observations) -3
        
        for gfeat_id in invalid_feat_ids:
            del self.geo_feats_dict[gfeat_id]
        
        # return if there are not enough lost features to process
        if len(processed_feat_ids) == 0:
            return
        
        # self.visu_msckf_features(self.timestep-1, processed_feat_ids)

        # construct the jacobian matrix and observation residual
        H_x = np.zeros((jacobian_row_size, 6*n+3*m))
        r = np.zeros(jacobian_row_size)
        stack_cntr = 0

        # print('number of features before gating test:', len(processed_feat_ids))
        feat_cnt = 0
        for gfeat_id in processed_feat_ids:
            geo_feat = self.geo_feats_dict[gfeat_id]
            H_xj, r_j = self.msckf_feature_jacobian(gfeat_id)
            
            # TODO: perform gating test
            if not self.gating_test(H_xj, r_j, (len(geo_feat.observations)-1)//2):
                continue

            # print('H_xj:', H_xj.shape, 'r_j:', r_j.shape)
            H_x[stack_cntr:stack_cntr+H_xj.shape[0], :] = H_xj
            r[stack_cntr:stack_cntr+r_j.shape[0]] = r_j
            stack_cntr += H_xj.shape[0]
            feat_cnt += 1

        H_x = H_x[:stack_cntr, :]
        r = r[:stack_cntr]

        ########## perform measurement update ##########
        qr = self.update_with_qr
        self.measurement_update(H_x, r, 'geo', qr)

        # remove all processed features
        for gfeat_id in processed_feat_ids:
            del self.geo_feats_dict[gfeat_id]
        
    def project_pnts(self, cam_pose, pnts, stereo=True):
        if pnts.ndim == 1:
            pnts = pnts[:, np.newaxis]
        assert pnts.shape[0] == 3, 'invalid pnts shape'


        cam0_pose_inv = np.linalg.inv(cam_pose)
        pnt_incam0 = cam0_pose_inv @ homogenize(pnts)
        proj_pnt_cam0 = self.Km @ (pnt_incam0[:3, :] / pnt_incam0[2, :])

        if stereo:
            assert self.o0To1 is not None, 'stereo relative camera pose not provided'
        
            cam1_pose = cam_pose @ self.o0To1
            cam1_pose_inv = np.linalg.inv(cam1_pose)
            pnt_incam1 = cam1_pose_inv @ homogenize(pnts)
            proj_pnt_cam1 = self.Km @ (pnt_incam1[:3] / pnt_incam1[2])
        
        return np.vstack((proj_pnt_cam0[:2, :], proj_pnt_cam1[:2, :])) \
            if stereo else proj_pnt_cam0[:2, :]

    def gating_test(self, H, r, dof):
        P1 = H @ self.state.cov @ H.T
        P2 = self.geo_obs_noise * np.eye(H.shape[0])

        # gamma = r.T @ np.linalg.inv(P1+P2) @ r
        gamma = r.T @ np.linalg.solve(P1+P2, np.eye(P1.shape[0])) @ r
        # perform chi-squared test
        if gamma < self.chi_squared_table[dof]:
            return True
        else:
            return False
    
    def measurement_update(self, H_x, r, type, qr=True):
        n = self.state.num_cameras()

        if qr and H_x.shape[0] > H_x.shape[1]:
            Q, R = np.linalg.qr(H_x, mode='reduced')

            H_thin = Q.T @ H_x
            r_thin = Q.T @ r
        else:
            H_thin = H_x
            r_thin = r

        # compute the Kalman gain
        noise_scale = self.geo_obs_noise if type == 'geo' else self.obj_obs_noise
        K = self.state.cov @ H_thin.T @ np.linalg.solve(H_thin @ self.state.cov @ H_thin.T + \
                                                        np.eye(H_thin.shape[0])*noise_scale,
                                                        np.eye(H_thin.shape[0]))
        innov = K @ r_thin

        to_update = True
        # for i in range(n):
        #     if np.linalg.norm(innov[6*i:6*i+3]) > 1e-3:
        #         to_update = False
        #         break
        
        if to_update:
            if type == 'obj':
                self.num_update += 1
            if type == 'geo':
                self.num_msckf_update += 1

            for i in range(n):
                self.state.cam_poses[:, 4*i:4*i+4] = self.state.cam_poses[:, 4*i:4*i+4] @ SE3_exp(innov[6*i:6*i+6])

            self.state.objects += innov[6*n:]
            self.state.cov = (np.eye(self.state.cov.shape[0]) - K @ H_thin) @ self.state.cov            

    def initialize_objects(self):
        invalid_obj_ids = []
        processed__obj_ids = []

        for obj_id in self.uninit_objs_dict:
            obj_feat = self.uninit_objs_dict[obj_id]
            assert obj_id == obj_feat.id, "obj feature id does not match"

            if len(obj_feat.observations) < self.obj_init_fnum_thre:
                continue
            
            if not obj_feat.is_initialized:
                if self.stereo_obj:
                    if not obj_feat.initialize_position(self.state.cam_ids, self.state.cam_poses, 
                                                        self.fsu_b, self.o0To1, self.obj_init_reproj_thre):
                        # print('stereo object initialization failed')
                        invalid_obj_ids.append(obj_id)
                        continue
                else:
                    if not obj_feat.initialize_position(self.state.cam_ids, self.state.cam_poses):
                        # print('monocular object initialization failed')
                        invalid_obj_ids.append(obj_id)
                        continue
            
            # try to add the landmark to the state
            obj_mean, Sigma_xf, Sigma_ff = self.new_obj_mean_cov(obj_id)

            if obj_mean is None:
                invalid_obj_ids.append(obj_id)
                continue
            
            obj_traj_dist = np.linalg.norm(obj_mean - self.state.cam_poses[:3, -1])
            if obj_traj_dist > self.obj_init_dist_max_thre or obj_traj_dist < self.obj_init_dist_min_thre:
                invalid_obj_ids.append(obj_id)
                continue

            self.state.objects = np.append(self.state.objects, obj_mean)
            processed__obj_ids.append(obj_id)
            self.state.cov = np.block([[self.state.cov, Sigma_xf], [Sigma_xf.T, Sigma_ff]])

            self.state.object_ids = np.append(self.state.object_ids, obj_id)
            self.state.object_init_frames = np.append(self.state.object_init_frames, self.state.cam_ids[-1])

        # delete invalid and processed objects
        for obj_id in invalid_obj_ids:
            del self.uninit_objs_dict[obj_id]

        for obj_id in processed__obj_ids:
            del self.uninit_objs_dict[obj_id]

    def new_obj_mean_cov(self, obj_id):
        n, m = self.state.num_cameras(), self.state.num_objects()
        obj_feat = self.uninit_objs_dict[obj_id]
        obj_landm_homo = homogenize(obj_feat.position)

        valid_cam_ids = []
        valid_observations = []
        for frame_id, pt_norm in obj_feat.observations:
            if frame_id in self.state.cam_ids:
                valid_cam_ids.append(frame_id)
                valid_observations.append(pt_norm)
        
        assert len(valid_cam_ids) == len(valid_observations)
        if len(valid_cam_ids) < self.obj_init_fnum_thre:
            return None, None, None
        
        if self.stereo_obj:
            jacobian_row_size = 4 * len(valid_cam_ids)
        else:
            jacobian_row_size = 2 * len(valid_cam_ids)

        Hx = np.zeros((jacobian_row_size, 6*n+3*m))
        Hf = np.zeros((jacobian_row_size, 3))
        z_hat = np.zeros(jacobian_row_size)
        V_aug = np.zeros((jacobian_row_size, jacobian_row_size))

        for i in range(len(valid_cam_ids)):
            cam_id = valid_cam_ids[i]
            cam_state_idx = np.where(self.state.cam_ids == cam_id)[0][0]
            cam0_pose = self.state.cam_poses[:, 4*cam_state_idx:4*cam_state_idx+4]
            cam0_pose_inv = np.linalg.inv(cam0_pose)

            if not self.stereo_obj:
                # jacobian and observation prediction for monocular camera
                Hxi = -(dpi_dq(self.oTb @ cam0_pose_inv @ obj_landm_homo) @ self.oTb @ circle_dot(cam0_pose_inv @ obj_landm_homo))[:2, :]
                Hx[2*i:2*i+2, 6*cam_state_idx:6*cam_state_idx+6] = Hxi
                Hfi = (dpi_dq(self.oTb @ cam0_pose_inv @ obj_landm_homo) @ self.oTb @ cam0_pose_inv @ self.P.T)[:2, :]
                Hf[2*i:2*i+2, :] = Hfi

                obs_pred = self.oTb @ cam0_pose_inv @ homogenize(obj_feat.position)
                z_hat[2*i:2*i+2] = (obs_pred / obs_pred[2])[:2]
                V_aug[2*i:2*i+2, 2*i:2*i+2] = self.V_obj
            else:
                # jacobian and observation prediction for stereo camera
                Hxi_left = -(dpi_dq(self.oTb @ cam0_pose_inv @ obj_landm_homo) @ self.oTb @ circle_dot(cam0_pose_inv @ obj_landm_homo))[:2, :]
                Hxi_right = -(dpi_dq(self.o1To0 @ self.oTb @ cam0_pose_inv @ obj_landm_homo) @ self.o1To0 @ self.oTb @ circle_dot(cam0_pose_inv @ obj_landm_homo))[:2, :]
                Hx[4*i:4*i+2, 6*cam_state_idx:6*cam_state_idx+6] = Hxi_left
                Hx[4*i+2:4*i+4, 6*cam_state_idx:6*cam_state_idx+6] = Hxi_right

                Hfi_left = (dpi_dq(self.oTb @ cam0_pose_inv @ obj_landm_homo) @ self.oTb @ cam0_pose_inv @ self.P.T)[:2, :]
                Hfi_right = (dpi_dq(self.o1To0 @ self.oTb @ cam0_pose_inv @ obj_landm_homo) @ self.o1To0 @ self.oTb @ cam0_pose_inv @ self.P.T)[:2, :]
                Hf[4*i:4*i+2, :] = Hfi_left
                Hf[4*i+2:4*i+4, :] = Hfi_right

                obs_pred_left = self.oTb @ cam0_pose_inv @ homogenize(obj_feat.position)
                obs_pred_right = self.o1To0 @ self.oTb @ cam0_pose_inv @ homogenize(obj_feat.position)
                z_hat[4*i:4*i+2] = (obs_pred_left / obs_pred_left[2])[:2]
                z_hat[4*i+2:4*i+4] = (obs_pred_right / obs_pred_right[2])[:2]
                V_aug[4*i:4*i+2, 4*i:4*i+2] = self.V_obj
                V_aug[4*i+2:4*i+4, 4*i+2:4*i+4] = self.V_obj

        z = np.concatenate([valid_observations[k] for k in range(len(valid_observations))])
        z_res = z - z_hat
        Q, R = np.linalg.qr(Hf, mode='complete')
        z_res1 = (Q.T @ z_res)[:3]
        Hx1 = (Q.T @ Hx)[:3, :]
        Hf1 = R[:3, :]
        V_aug1 = (Q.T @ V_aug @ Q)[:3, :3]
        Hf1_inv = np.linalg.solve(Hf1, np.eye(Hf1.shape[0]))
        p_init = obj_feat.position + Hf1_inv @ z_res1
        # p_init = obj_feat.position
        Sigma_xf = - self.state.cov @ Hx1.T @ Hf1_inv.T
        Sigma_ff = Hf1_inv @ (Hx1 @ self.state.cov @ Hx1.T + V_aug1) @ Hf1_inv.T

        # if np.linalg.matrix_rank(Sigma_ff) != Sigma_ff.shape[0]:
        #     print('singular object covariance')
        # if np.linalg.det(Sigma_ff) < 0:
        #     print('negative object covariance determinant')

        return p_init, Sigma_xf, Sigma_ff
    
    def obj_feature_jacobian_mono(self, cam_id, obj_id):
        n, m = self.state.num_cameras(), self.state.num_objects()
        cam_state_idx = np.where(self.state.cam_ids == cam_id)[0][0]
        cam_pose = self.state.cam_poses[:, 4*cam_state_idx:4*cam_state_idx+4]
        cam_pose_inv = np.linalg.inv(cam_pose)

        obj_state_idx = np.where(self.state.object_ids == obj_id)[0][0]
        obj_landm = self.state.objects[3*obj_state_idx:3*obj_state_idx+3]
        obj_landm_cam_homo = cam_pose_inv @ homogenize(obj_landm)

        H = np.zeros((2, 6*n+3*m))
        H_cam = -(dpi_dq(obj_landm_cam_homo) @ circle_dot(obj_landm_cam_homo))[:2, :]
        H[:, 6*cam_state_idx:6*cam_state_idx+6] = H_cam

        H_obj = (dpi_dq(obj_landm_cam_homo) @ cam_pose_inv @ self.P.T)[:2, :]
        H[:, 6*n+3*obj_state_idx:6*n+3*obj_state_idx+3] = H_obj
        z_hat = (obj_landm_cam_homo / obj_landm_cam_homo[2])[:2]

        return H, z_hat

    def obj_feature_jacobian_stereo(self, cam_id, obj_id):
        n, m = self.state.num_cameras(), self.state.num_objects()
        cam_state_idx = np.where(self.state.cam_ids == cam_id)[0][0]
        
        cam_pose_obs = self.state.cam_poses[:, 4*cam_state_idx:4*cam_state_idx+4]
        cam_pose_obs_inv = np.linalg.inv(cam_pose_obs)
        obj_state_idx = np.where(self.state.object_ids == obj_id)[0][0]
        obj_landm = self.state.objects[3*obj_state_idx:3*obj_state_idx+3]

        # predicted observation
        obj_landm_cam_homo = cam_pose_obs_inv @ homogenize(obj_landm)
        z_hat_left = self.oTb @ obj_landm_cam_homo
        z_hat_left = (z_hat_left / z_hat_left[2])[:2]
        obj_landm_cam1_homo = self.o1To0 @ self.oTb @ obj_landm_cam_homo
        z_hat_right = (obj_landm_cam1_homo / obj_landm_cam1_homo[2])[:2]
        z_hat = np.concatenate([z_hat_left, z_hat_right])

        H = np.zeros((4, 6*n+3*m))
        # jacobian with respect to the object
        H_obj_left = (dpi_dq(self.oTb @ obj_landm_cam_homo) @ self.oTb @ cam_pose_obs_inv @ self.P.T)[:2, :]
        H_obj_right = (dpi_dq(self.o1To0 @ self.oTb @ obj_landm_cam_homo) @ self.o1To0 @ self.oTb @ cam_pose_obs_inv @ self.P.T)[:2, :]
        H[:2, 6*n+3*obj_state_idx:6*n+3*obj_state_idx+3] = H_obj_left
        H[2:, 6*n+3*obj_state_idx:6*n+3*obj_state_idx+3] = H_obj_right

        # jacobian with respect to the camera poses
        cam_pose = self.state.cam_poses[:, 4*cam_state_idx:4*cam_state_idx+4]
        cam_pose_inv = np.linalg.inv(cam_pose)
        obj_landm_cam_homo = cam_pose_inv @ homogenize(obj_landm)
        rel_pose = cam_pose_obs_inv @ cam_pose
        proj_pose = self.oTb @ rel_pose
        H_cam_left = -(dpi_dq(proj_pose @ obj_landm_cam_homo) @ proj_pose @ circle_dot(obj_landm_cam_homo))[:2, :]
        H_cam_right = -(dpi_dq(self.o1To0 @ proj_pose @ obj_landm_cam_homo) @ self.o1To0 @ proj_pose @ circle_dot(obj_landm_cam_homo))[:2, :]

        H[:2, 6*cam_state_idx:6*cam_state_idx+6] = H_cam_left
        H[2:, 6*cam_state_idx:6*cam_state_idx+6] = H_cam_right
        
        return H, z_hat

    def is_valid_measurement(self, cam_id, obj_id, obj_meas):
        cam_state_idx = np.where(self.state.cam_ids == cam_id)[0][0]
        cam_pose = self.state.cam_poses[:, 4*cam_state_idx:4*cam_state_idx+4]
        cam_pose_inv = np.linalg.inv(cam_pose)

        obj_state_idx = np.where(self.state.object_ids == obj_id)[0][0]
        obj_landm = self.state.objects[3*obj_state_idx:3*obj_state_idx+3]
        obj_landm_cam0_homo = cam_pose_inv @ homogenize(obj_landm)

        if self.stereo_obj:
            z_left, z_right = obj_meas[:2], obj_meas[2:]
            
            z_hat_left = self.oTb @ obj_landm_cam0_homo
            z_hat_left = (self.Km @ (z_hat_left / z_hat_left[2])[:3])[:2]
            if np.linalg.norm(z_left - z_hat_left) > self.obj_update_reproj_thre:
                return False

            z_hat_right = self.o1To0 @ self.oTb @ obj_landm_cam0_homo
            z_hat_right = (self.Km @ (z_hat_right / z_hat_right[2])[:3])[:2]
            # z_hat = np.concatenate([z_hat_left, z_hat_right])
            if np.linalg.norm(z_right - z_hat_right) > self.obj_update_reproj_thre:
                return False
        else:
            z = obj_meas
            z_hat = self.oTb @ obj_landm_cam0_homo
            z_hat = (self.Km @ (z_hat / z_hat[2])[:3])[:2]
            if np.linalg.norm(z - z_hat) > self.obj_update_reproj_thre:
                return False
        
        return True
    
    def process_obj_measurements(self, cam_id, obj_feats_frame):
        obj_ids_update, obj_meass_update = [], []
        update_chance = False
        for obj_id, obj_meas in obj_feats_frame.items():
            if obj_id not in self.state.object_ids:
                # object not existing in the filter state
                if obj_id not in self.uninit_objs_dict:
                    self.uninit_objs_dict[obj_id] = StereoObjFeature(obj_id, self.Km, self.oTb)
                
                if not self.stereo_obj:
                    self.uninit_objs_dict[obj_id].add_observation(cam_id, (self.Km_inv @ homogenize(obj_meas))[:2])
                else:
                    obs_left = (self.Km_inv @ homogenize(obj_meas[:2]))[:2]
                    obs_right = (self.Km_inv @ homogenize(obj_meas[2:]))[:2]
                    self.uninit_objs_dict[obj_id].add_observation(cam_id, np.concatenate([obs_left, obs_right]))
            else:
                update_chance = True
                # object existing in the filter state
                if not self.is_valid_measurement(cam_id, obj_id, obj_meas):
                    continue

                if not self.stereo_obj:
                    obj_meass_update.append((self.Km_inv @ homogenize(obj_meas))[:2])
                else:
                    obs_left = (self.Km_inv @ homogenize(obj_meas[:2]))[:2]
                    obs_right = (self.Km_inv @ homogenize(obj_meas[2:]))[:2]
                    obj_meass_update.append(np.concatenate([obs_left, obs_right]))
                
                obj_ids_update.append(obj_id)

        if update_chance:
            self.num_update_chance += 1
        
        assert len(obj_ids_update) == len(obj_meass_update), 'number of object ids and measurements do not match'
        if len(obj_ids_update) == 0:
            return

        n, m = self.state.num_cameras(), self.state.num_objects()
        
        if not self.stereo_obj:
            H = np.zeros((2*len(obj_meass_update), 6*n+3*m))
            z_hat = np.zeros(2*len(obj_meass_update))
            z = np.zeros(2*len(obj_meass_update))
            for i in range(len(obj_ids_update)):
                obj_id = obj_ids_update[i]
                H_i, z_hat_i = self.obj_feature_jacobian_mono(cam_id, obj_id)
                H[2*i:2*i+2, :] = H_i
                z_hat[2*i:2*i+2] = z_hat_i
                z[2*i:2*i+2] = obj_meass_update[i]
        else:
            H = np.zeros((4*len(obj_meass_update), 6*n+3*m))
            z_hat = np.zeros(4*len(obj_meass_update))
            z = np.zeros(4*len(obj_meass_update))
            for i in range(len(obj_ids_update)):
                obj_id = obj_ids_update[i]
                H_i, z_hat_i = self.obj_feature_jacobian_stereo(cam_id, obj_id)
                H[4*i:4*i+4, :] = H_i
                z_hat[4*i:4*i+4] = z_hat_i
                z[4*i:4*i+4] = obj_meass_update[i]

        qr = self.update_with_qr
        self.measurement_update(H, z-z_hat, 'obj', qr)

    def find_common_obj_ids(self, neighbor_obj_ids):
        common_obj_ids_dict = {}
        for neighbor_id in neighbor_obj_ids.keys():
            obj_ids = neighbor_obj_ids[neighbor_id]
            common_mask = np.isin(self.state.object_ids, obj_ids)
            common_obj_ids_dict[neighbor_id] = self.state.object_ids[common_mask]

        return common_obj_ids_dict

    def share_common_mean_cov(self, common_obj_ids):
        n = self.state.num_cameras() # number of cameras

        common_obj_idxes = []
        for obj_id in common_obj_ids:
            common_obj_idxes.append(np.where(self.state.object_ids == obj_id)[0][0])
        common_obj_idxes = np.array(common_obj_idxes)[:, np.newaxis]

        indices = np.hstack([3*common_obj_idxes, 3*common_obj_idxes+1, 3*common_obj_idxes+2])
        indices = indices.reshape(-1)

        common_mean = self.state.objects[indices]
        common_cov = self.state.cov[6*n:, 6*n:][indices, :][:, indices]

        return common_mean, common_cov, indices

    def get_corresponding_mean_cov(self, obj_ids, obj_ids_all, means_all, covs_all):
        obj_idxes = []
        for obj_id in obj_ids:
            idx = np.where(obj_ids_all == obj_id)[0][0]
            obj_idxes.append(idx)
        obj_idxes = np.array(obj_idxes)[:, np.newaxis]    

        indices = np.hstack([3*obj_idxes, 3*obj_idxes+1, 3*obj_idxes+2])
        indices = indices.reshape(-1)

        means = means_all[indices]
        covs = covs_all[indices, :][:, indices]

        return means, covs

    def fuse_neighbor_info(self, common_obj_ids_dict, common_means_dict, common_covs_dict, debug=False):
        # find the object ids that are observed by all neighbors
        common_mask = np.ones(self.state.num_objects(), dtype=bool)
        for robot_id in common_obj_ids_dict.keys():
            common_obj_ids = common_obj_ids_dict[robot_id]
            common_mask = np.logical_and(common_mask, np.isin(self.state.object_ids, common_obj_ids))
        
        # fuse the means and covariances of the common objects from all neighbors
        if np.sum(common_mask) > 0:
            common_obj_ids_allNi = self.state.object_ids[common_mask]
            if len(common_obj_ids_allNi) > 0:
                common_means_allNi, common_covs_allNi = [], []
                
                for robot_id in common_means_dict.keys():
                    mean, cov = self.get_corresponding_mean_cov(common_obj_ids_allNi, common_obj_ids_dict[robot_id], common_means_dict[robot_id], common_covs_dict[robot_id])
                    common_means_allNi.append(mean)
                    common_covs_allNi.append(cov)

                self.consensus_average(common_obj_ids_allNi, common_means_allNi, common_covs_allNi, debug)
        else:
            common_obj_ids_allNi = []

        # fuse the means and covariances of the common objects one neighbor by one neighbor
        for robot_id in common_obj_ids_dict.keys():
            # mask out common objects that are already fused
            common_obj_ids = common_obj_ids_dict[robot_id]
            if len(common_obj_ids) > 0:
                if len(common_obj_ids_allNi) > 0:
                    common_mask = np.logical_not(np.isin(common_obj_ids, common_obj_ids_allNi))
                    if np.sum(common_mask) == 0:
                        continue

                    common_obj_ids_left = common_obj_ids[common_mask]
                    common_mean_left, common_cov_left = self.get_corresponding_mean_cov(common_obj_ids_left, 
                        common_obj_ids_dict[robot_id], common_means_dict[robot_id], common_covs_dict[robot_id])
                else:
                    common_obj_ids_left = common_obj_ids
                    common_mean_left, common_cov_left = common_means_dict[robot_id], common_covs_dict[robot_id]
                
                self.consensus_average(common_obj_ids_left, [common_mean_left], [common_cov_left], debug)


    def consensus_average(self, common_obj_ids, neighbor_common_means, neighbor_common_covs, debug=False):
        n = self.state.num_cameras() # number of cameras
        m = self.state.num_objects() # number of objects
    
        # fuse common mean and covariance in information space
        self_common_mean, self_common_cov, common_indices = self.share_common_mean_cov(common_obj_ids)
        if np.linalg.det(self_common_cov) < 1e-100:
            return
        
        # self_common_info_mat = np.linalg.inv(self_common_cov)
        self_common_info_mat = np.linalg.solve(self_common_cov, np.eye(self_common_cov.shape[0]))
        self_common_info_mean = self_common_info_mat @ self_common_mean
        
        num_neighbors = len(neighbor_common_means) + 1
        weights = [1 / num_neighbors] * num_neighbors # equal weights
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        fused_info_mat = self_common_info_mat
        fused_info_mean = self_common_info_mean
        assert len(neighbor_common_means) == len(neighbor_common_covs), \
            'number of neighbor common means and covariances do not match'

        for i in range(len(neighbor_common_means)):
            # neighbor_common_info_mat = np.linalg.inv(neighbor_common_covs[i])
            neighbor_common_info_mat = np.linalg.solve(neighbor_common_covs[i], np.eye(neighbor_common_covs[i].shape[0]))
            neighbor_common_info_mean = neighbor_common_info_mat @ neighbor_common_means[i]
            fused_info_mat += neighbor_common_info_mat
            fused_info_mean += neighbor_common_info_mean
            
        # divide by the number of neighbors including itself (average)
        fused_info_mat = fused_info_mat / (len(neighbor_common_means) + 1)
        fused_info_mean = fused_info_mean / (len(neighbor_common_means) + 1)
        
        # new common mean and covariance (the marginal distribution of common objects)
        fused_cov = np.linalg.solve(fused_info_mat, np.eye(fused_info_mat.shape[0]))
        fused_mean = fused_cov @ fused_info_mean
        ############## DEBUG ##############
        if debug:
            print('self cov det:', np.linalg.det(self_common_cov))
            for i in range(len(neighbor_common_means)):
                print('neighbor cov det:', np.linalg.det(neighbor_common_covs[i]))
            print('fused cov det:', np.linalg.det(fused_cov))

            for i in range(len(neighbor_common_means)):
                # print('neighbor mean:', neighbor_common_means[i])
                print('mean dist:', np.linalg.norm(self_common_mean - neighbor_common_means[i]))
            
            print('num of obj:', len(common_obj_ids))

            fused_cov_det = np.linalg.det(fused_cov)
            if fused_cov_det < 0:
                print('negative cov det after fusing')
                print('self cov det: ', np.linalg.det(self_common_cov))
                for i in range(len(neighbor_common_means)):
                    print('neighbor cov det: ', np.linalg.det(neighbor_common_covs[i]))
                print('cov det after fusing: ', np.linalg.det(fused_cov))

        # compute mean distance
        common_mu_dist = np.linalg.norm(self_common_mean - fused_mean)
        if common_mu_dist > self.common_mu_thre:
            alpha = 1 - self.common_mu_thre / common_mu_dist
            fused_info_mean = alpha * self_common_info_mean + (1-alpha) * fused_info_mean
            fused_info_mat = alpha * self_common_info_mat + (1-alpha) * fused_info_mat

            fused_cov = np.linalg.solve(fused_info_mat, np.eye(fused_info_mat.shape[0]))
            fused_mean = fused_cov @ fused_info_mean

        fused_cov = np.linalg.solve(fused_info_mat, np.eye(fused_info_mat.shape[0]))
        fused_mean = fused_cov @ fused_info_mean
        
        ########## debug ##########
        common_indices = common_indices + 6*n
        common_mask = np.zeros(6*n+3*m, dtype=bool)
        common_mask[common_indices] = True
        private_mask = np.logical_not(common_mask)
        private_indices = np.where(private_mask)[0]

        # compute conditional distribution, denote the common part as y and the rest as x
        Sigma_xx = self.state.cov[np.ix_(private_indices, private_indices)]
        Sigma_xy = self.state.cov[np.ix_(private_indices, common_indices)]
        Sigma_yy = self.state.cov[np.ix_(common_indices, common_indices)]
        # Sigma_yy_inv = np.linalg.inv(Sigma_yy)
        Sigma_yy_inv = np.linalg.solve(Sigma_yy, np.eye(Sigma_yy.shape[0]))
        
        mean_all = np.hstack([np.zeros(6*n), self.state.objects])
        mean_x = mean_all[private_indices]
        mean_y = mean_all[common_indices]
        
        A = Sigma_xy @ Sigma_yy_inv
        b = mean_x - A @ mean_y
        Sigma_cond = Sigma_xx - Sigma_xy @ Sigma_yy_inv @ Sigma_xy.T

        # update the mean and covariance using fused common mean and covariance
        new_mean_x = A @ fused_mean + b
        new_cov_xx = A @ fused_cov @ A.T + Sigma_cond
        new_cov_xy = A @ fused_cov

        new_mean_all = np.zeros_like(mean_all)
        new_cov_all = np.zeros_like(self.state.cov)
        new_mean_all[private_indices] = new_mean_x
        new_mean_all[common_indices] = fused_mean

        new_cov_all[np.ix_(private_indices, private_indices)] = new_cov_xx
        new_cov_all[np.ix_(private_indices, common_indices)] = new_cov_xy
        new_cov_all[np.ix_(common_indices, private_indices)] = new_cov_xy.T
        new_cov_all[np.ix_(common_indices, common_indices)] = fused_cov

        to_update = True
        for i in range(n):
            if np.linalg.norm(new_mean_all[6*i:6*i+6]) > self.pose_fuse_update_thre:
                to_update = False
                break
        
        for i in range(m):
            if np.linalg.norm(new_mean_all[6*n+3*i:6*n+3*i+3] - self.state.objects[3*i:3*i+3]) > self.obj_fuse_update_thre:
                # print('new_mean:', new_mean_all[6*n+3*i:6*n+3*i+3], 'old_mean:', self.state.objects[3*i:3*i+3])
                # print('dist:', np.linalg.norm(new_mean_all[6*n+3*i:6*n+3*i+3] - self.state.objects[3*i:3*i+3]))

                to_update = False
                break

        if to_update:
            # update the state according to the new mean and cov
            for i in range(n):
                self.state.cam_poses[:, 4*i:4*i+4] = self.state.cam_poses[:, 4*i:4*i+4] @ SE3_exp(new_mean_all[6*i:6*i+6])
                if debug:
                    print('cam pose update:', np.linalg.norm(new_mean_all[6*i:6*i+6]))

            self.state.objects = new_mean_all[6*n:]
            self.state.cov = new_cov_all
            self.num_fuse += 1

