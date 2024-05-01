import os
import pickle
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from src.agent import DSLAM
from src.utils import read_poses
import yaml
import argparse


def load_params(data_root, param_root, dataset, data_id):
    
    param_fname = os.path.join(param_root, '%s_%s.yaml' % (dataset, data_id))
    with open(param_fname, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # load calibrations
    calib_file = os.path.join(data_root, dataset, data_id, 'calib')
    with open(calib_file, 'rb') as f:
        calib = pickle.load(f)
    
    params['opt_T_b'] = calib['opt_T_b']
    params['K_mono'] = calib['K_mono']
    params['o0To1'] = calib['o0To1']
    params['fsu_b'] = calib['fsu_b']

    return params


def load_data(data_root, dataset, data_id):
    data_path = os.path.join(data_root, dataset, data_id)

    # load geometric feature measurements
    geo_feat_file = os.path.join(data_path, 'geo_features_messages')
    with open(geo_feat_file, 'rb') as f:
        geo_feat_messages = pickle.load(f)
        
    # load object feature measurements
    obj_feat_file = os.path.join(data_path, 'obj_features_messages')
    with open(obj_feat_file, 'rb') as f:
        obj_feat_messages = pickle.load(f)

    # load odometry measurements
    input_file = os.path.join(data_path, 'inputs.npz')
    robot_inputs = read_poses(input_file)

    # load ground truth poses
    gt_file = os.path.join(data_path, 'poses.npz')
    robot_gt_poses = read_poses(gt_file)

    # load gt landmarks
    gt_obj_file = os.path.join(data_path, 'obj_features_gt')
    with open(gt_obj_file, 'rb') as f:
        gt_objects = pickle.load(f)
    
    return geo_feat_messages, obj_feat_messages, robot_inputs, robot_gt_poses, gt_objects


if __name__ == "__main__":

    ########## parse arguments ##########
    parser = argparse.ArgumentParser(description='Multi-robot EKF with MSCKF')
    parser.add_argument('--dataset', type=str, default='kitti', help='dataset name')
    parser.add_argument('--data_id', type=str, default='00', help='data id')
    args = parser.parse_args()

    data_root = './Data'
    param_root = './params'
    save_root = './results'
    dataset = args.dataset
    data_id = args.data_id
    print('runing dataset:', dataset, 'data id:', data_id)
    save_path = os.path.join(save_root, dataset, data_id)
    os.makedirs(save_path, exist_ok=True)

    # parameters
    use_sim_data = False
    params = load_params(data_root, param_root, dataset, data_id)

    ######## load data ########
    geo_feat_messages, obj_feat_messages, robot_inputs, \
        robot_gt_poses, gt_objects = load_data(data_root, dataset, data_id)
    if dataset == 'simulated':
        for robot_id in range(len(robot_inputs)):
            robot_inputs[robot_id] = np.linalg.inv(robot_inputs[robot_id])
    
    num_robots = len(robot_inputs)
    max_seq_len = max([robot_inputs[i].shape[0] for i in range(num_robots)]) + 1
    print('num of robots:', num_robots, ' max seq len:', max_seq_len)

    # create agents
    robots = []
    for robot_id in range(num_robots):
        robots.append(DSLAM(robot_id, params, robot_gt_poses[robot_id][0]))
    # define the communication graph
    robot_neighbors = [[] for _ in range(num_robots)]
    for i in range(num_robots):
        for j in range(num_robots):
            if i != j:
                robot_neighbors[i].append(j)

    estim_cam_poses, pred_cam_poses = [], []
    estim_cam_trajs, pred_cam_trajs, gt_cam_trajs = [], [], []
    for robot_id in range(num_robots):
        estim_cam_poses.append([robot_gt_poses[robot_id][0]])
        pred_cam_poses.append([robot_gt_poses[robot_id][0]])

        estim_cam_trajs.append([robot_gt_poses[robot_id][0][:3, 3]])
        pred_cam_trajs.append([robot_gt_poses[robot_id][0][:3, 3]])
        gt_cam_trajs.append([robot_gt_poses[robot_id][0][:3, 3]])

    ########## only prediction for comparison ##########
    for i in range(max_seq_len-1):
        for robot_id in range(num_robots):
            if i >= robot_inputs[robot_id].shape[0]:
                continue
            
            pred_cam_poses[robot_id].append(pred_cam_poses[robot_id][-1] @ robot_inputs[robot_id][i])
            pred_cam_trajs[robot_id].append(pred_cam_poses[robot_id][-1][:3, 3])
            gt_cam_trajs[robot_id].append(robot_gt_poses[robot_id][i+1][:3, 3])
    
    for i in range(num_robots):
        pred_cam_poses[i] = np.array(pred_cam_poses[i])
        pred_cam_trajs[i] = np.array(pred_cam_trajs[i])
        gt_cam_trajs[i] = np.array(gt_cam_trajs[i])

    ########### time analysis ###########
    geo_update_times = [[] for _ in range(num_robots)]
    obj_init_times, obj_update_times, prediction_times = \
        [[] for _ in range(num_robots)], [[] for _ in range(num_robots)], [[] for _ in range(num_robots)]
    consensus_times = [[] for _ in range(num_robots)]

    ########## start distributed SLAM ##########
    for i in tqdm(range(max_seq_len)):

        for robot_id in range(num_robots):
            # print('i:', i)
            if i >= robot_inputs[robot_id].shape[0]:
                continue

            robot = robots[robot_id]
            
            ######### process geometric feature measurements #########
            if params['use_msckf']:
                tic = time.time()

                geo_frame_id = i
                geo_feats_frame = geo_feat_messages[robot_id][geo_frame_id]
                robot.add_msckf_features(i, geo_feats_frame)
                robot.remove_lost_msckf_features()
                toc = time.time()
                geo_update_times[robot_id].append(toc-tic)

            ######### process object feature measurements #########
            if params['include_obj']:
                tic = time.time()
                robot.initialize_objects()
                toc = time.time()
                obj_init_times[robot_id].append(toc-tic)

                tic = time.time()
                obj_feats_frame = obj_feat_messages[robot_id][i]
                robot.process_obj_measurements(i, obj_feats_frame)
                toc = time.time()
                obj_update_times[robot_id].append(toc-tic)

                assert len(robot.state.object_ids) == len(robot.state.objects) // 3

            ########## consensus step ##########
            if params['consensus_average']:
                tic = time.time()
                # receive neighbor's object ids
                neighbor_obj_ids = {}
                for neighbor_id in robot_neighbors[robot_id]:
                    neighbor = robots[neighbor_id]
                    neighbor_obj_ids[neighbor_id] = neighbor.state.object_ids
                
                # find common object ids
                common_obj_ids_dict = robot.find_common_obj_ids(neighbor_obj_ids)
                
                neighbor_common_means, neighbor_common_covs = {}, {}
                for neighbor_id in robot_neighbors[robot_id]:
                    if len(common_obj_ids_dict[neighbor_id]) == 0:
                        continue

                    neighbor = robots[neighbor_id]
                    common_mean, common_cov, _ = neighbor.share_common_mean_cov(common_obj_ids_dict[neighbor_id])
                    neighbor_common_means[neighbor_id] = common_mean
                    neighbor_common_covs[neighbor_id] = common_cov

                robot.fuse_neighbor_info(common_obj_ids_dict, neighbor_common_means, neighbor_common_covs)

                toc = time.time()
                consensus_times[robot_id].append(toc-tic)

            ######### prediction step #########
            tic = time.time()
            # read odometry measurements
            rel_pose = robot_inputs[robot_id][i]
            # pose prediction
            robot.predict(rel_pose, i+1)
            toc = time.time()
            prediction_times[robot_id].append(toc-tic)
            estim_cam_trajs[robot_id].append(robot.current_cam_pose()[:3, 3])
            estim_cam_poses[robot_id].append(robot.current_cam_pose())

    for i in range(num_robots):
        estim_cam_poses[i] = np.array(estim_cam_poses[i])
        estim_cam_trajs[i] = np.array(estim_cam_trajs[i])

    includes_obj_ids = []
    gt_objects_visu = []
    estim_objects = []
    for robot_id in range(num_robots):
        robot = robots[robot_id]
        dist_to_traj = np.linalg.norm(robot.state.objects.reshape((-1, 3))[:, np.newaxis, :] \
                                      - robot_gt_poses[robot_id][:, :3, 3].reshape((-1, 3))[np.newaxis, ...], axis=-1)
        dist_to_traj = np.min(dist_to_traj, axis=-1)
        valid_mask = dist_to_traj < 15

        estim_objects.append(robot.state.objects.reshape((-1, 3)))
        
        for obj_id in robot.state.object_ids:
            if obj_id in includes_obj_ids:
                continue
            gt_objects_visu.append(gt_objects[int(obj_id)])
            includes_obj_ids.append(obj_id)
    
    gt_objects_visu = np.array(gt_objects_visu)

    for robot_id in range(num_robots):
        print('robot %d has %d objects:' % (robot_id, len(robots[robot_id].state.object_ids)))
        print('robot %d has %d EKF update chances' % (robot_id, robots[robot_id].num_update_chance))
        print('robot %d EKF updated %d times' % (robot_id, robots[robot_id].num_update))
        print('robot %d fused %d times' % (robot_id, robots[robot_id].num_fuse))
        print('robot %d msckf updated %d times\n' % (robot_id, robots[robot_id].num_msckf_update))
        # print('robot %d object ids:' % robot_id, robots[robot_id].state.object_ids)

    ######## plot 3D camera trajectory and objects ########
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if dataset == 'kitti':
        r_colors = ['r', 'g', 'b']
    else:
        r_colors = plt.cm.rainbow(np.linspace(0, 1, num_robots))
        

    for robot_id in range(num_robots):
        ax.plot3D(gt_cam_trajs[robot_id][:, 0], gt_cam_trajs[robot_id][:, 1], gt_cam_trajs[robot_id][:, 2], \
                  c='black', label='ground truth r%d' % robot_id)
        ax.plot3D(estim_cam_trajs[robot_id][:, 0], estim_cam_trajs[robot_id][:, 1], estim_cam_trajs[robot_id][:, 2], \
                  c=r_colors[robot_id], label='estimation r%d' % robot_id)
        ax.plot3D(pred_cam_trajs[robot_id][:, 0], pred_cam_trajs[robot_id][:, 1], pred_cam_trajs[robot_id][:, 2], \
                  c='gray', label='prediction r%d' % robot_id)
    
    for robot_id in range(num_robots):
        ax.scatter3D(estim_objects[robot_id][:, 0], estim_objects[robot_id][:, 1], estim_objects[robot_id][:, 2], \
                     label='objects r%d' % robot_id)
    if len(gt_objects_visu) > 0:
        dist_to_traj = np.linalg.norm(gt_objects_visu[:, np.newaxis, :] - gt_cam_trajs[0][np.newaxis, ...], axis=-1)
        dist_to_traj = np.min(dist_to_traj, axis=-1)
        valid_mask = dist_to_traj < 15
        ax.scatter3D(gt_objects_visu[valid_mask, 0], gt_objects_visu[valid_mask, 1], gt_objects_visu[valid_mask, 2], \
                     label='gt objects')
        
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    if dataset == 'kitti':
        ax.legend()
    plt.show()

    ######## plot 2D camera trajectory and objects ########
    fig = plt.figure()
    ax = plt.axes()
    for robot_id in range(num_robots):
        ax.plot(gt_cam_trajs[robot_id][:, 0], gt_cam_trajs[robot_id][:, 1], c='black')
        ax.plot(pred_cam_trajs[robot_id][:, 0], pred_cam_trajs[robot_id][:, 1], c='gray')
        ax.plot(estim_cam_trajs[robot_id][:, 0], estim_cam_trajs[robot_id][:, 1], c=r_colors[robot_id])

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    plt.gca().set_aspect('equal')
    plt.show()

    ########## time analysis ##########
    for robot_id in range(num_robots):
        geo_update_times[robot_id] = np.array(geo_update_times[robot_id])
        obj_init_times[robot_id] = np.array(obj_init_times[robot_id])
        obj_update_times[robot_id] = np.array(obj_update_times[robot_id])
        prediction_times[robot_id] = np.array(prediction_times[robot_id])
        consensus_times[robot_id] = np.array(consensus_times[robot_id])

    time_analysis = {'geo_update': geo_update_times, \
                     'obj_update': obj_update_times, \
                     'obj_init': obj_init_times, \
                     'prediction': prediction_times, \
                     'consensus': consensus_times}

    ######### save results ###########
    object_ids = []
    for robot_id in range(num_robots):
        object_ids.append(robots[robot_id].state.object_ids)
    
    result_dict = {'estim_cam_poses': estim_cam_poses, \
                   'pred_cam_poses': pred_cam_poses, \
                   'object_ids': object_ids, \
                   'estim_objects': estim_objects, \
                   'time_analysis': time_analysis}

    consensus_name_str = 'consensus' if params['consensus_average'] else 'seperate'
    msckf_name_str = 'withmsckf' if params['use_msckf'] else 'nonmsckf'
    obj_name_str = 'withobj' if params['include_obj'] else 'nonobj'
    qr_name_str = 'withqr' if params['update_with_qr'] else 'nonqr'

    result_fname = os.path.join(save_path, 'result_%s_%s_%s.pkl' % (consensus_name_str, msckf_name_str, obj_name_str))
    if params['save_result']:
        with open(result_fname, 'wb') as f:
            pickle.dump(result_dict, f)


