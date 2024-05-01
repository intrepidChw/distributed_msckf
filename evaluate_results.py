import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.utils import read_poses
import argparse


if __name__ == "__main__":
    
    ########## parse arguments ##########
    parser = argparse.ArgumentParser(description='Multi-robot EKF with MSCKF')
    parser.add_argument('--dataset', type=str, default='kitti', help='dataset name')
    parser.add_argument('--data_id', type=str, default='00', help='data id')
    parser.add_argument('--result_fname1', type=str, default='result_seperate_withmsckf_withobj.pkl', help='result file name1')
    parser.add_argument('--result_fname2', type=str, default='result_consensus_withmsckf_withobj.pkl', help='result file name2')
    args = parser.parse_args()

    dataset = args.dataset
    data_id = args.data_id
    
    data_root = './Data'
    data_path = os.path.join(data_root, dataset, data_id)

    result_root = './results'
    # compare between consensus and seperate EKF
    result_fnames = [args.result_fname1, args.result_fname2]
    result1_fname, result2_fname = result_fnames[0], result_fnames[1]
    
    label1 = result1_fname.split('.')[0][7:]
    label2 = result2_fname.split('.')[0][7:]
    result1_filepath = os.path.join(result_root, dataset, data_id, result1_fname)
    result2_filepath = os.path.join(result_root, dataset, data_id, result2_fname)

    with open(result1_filepath, 'rb') as f:
        result1 = pickle.load(f)
    with open(result2_filepath, 'rb') as f:
        result2 = pickle.load(f)

    estim_poses_1 = result1['estim_cam_poses']
    estim_obj_ids_1 = result1['object_ids']
    estim_objects_1 = result1['estim_objects']

    estim_poses_2 = result2['estim_cam_poses']
    estim_obj_ids_2 = result2['object_ids']
    estim_objects_2 = result2['estim_objects']
    
    num_robot = len(estim_poses_1)
    assert len(estim_poses_1) == len(estim_poses_2), 'number of robots should be the same'
    
    # load ground truth poses
    gt_file = os.path.join(data_path, 'poses.npz')
    robot_gt_poses = read_poses(gt_file)

    # load ground truth objects
    gt_obj_file = os.path.join(data_path, 'obj_features_gt')
    with open(gt_obj_file, 'rb') as f:
        gt_objects = pickle.load(f)

    ########## evaluate robot pose estimation ##########
    # construct estimated trajectory for both rmse calculation and visualization
    estim_trajs_1, estim_trajs_2, gt_trajs = [], [], []
    for i in range(num_robot):
        assert len(estim_poses_1[i]) == len(estim_poses_2[i]) == len(robot_gt_poses[i]), \
            'length of traj lengths of estimation (consensus and seperate EKF) and ground truth should be the same'
        
        traj_1_robot_i, traj_2_robot_i, traj_gt_robot_i= [], [], []
        for j in range(estim_poses_2[i].shape[0]):
            traj_1_robot_i.append(estim_poses_1[i][j][:3, 3])
            traj_2_robot_i.append(estim_poses_2[i][j][:3, 3])
            traj_gt_robot_i.append(robot_gt_poses[i][j][:3, 3])
        
        estim_trajs_1.append(np.array(traj_1_robot_i))
        estim_trajs_2.append(np.array(traj_2_robot_i))
        gt_trajs.append(np.array(traj_gt_robot_i))
    
    # calculate RMSE
    traj_errors_1, traj_errors_2 = [], []
    rmse_1, rmse_2 = [], []
    for i in range(num_robot):
        traj_error_i_1, traj_error_i_2= [], []
        
        for j in range(len(estim_trajs_2[i])):
            traj_error_i_1.append(np.linalg.norm(estim_trajs_1[i][j] - gt_trajs[i][j]))
            traj_error_i_2.append(np.linalg.norm(estim_trajs_2[i][j] - gt_trajs[i][j]))
        
        traj_errors_1.append(traj_error_i_1)
        traj_errors_2.append(traj_error_i_2)

        rmse_i_1 = np.sqrt(np.sum(np.square(np.array(traj_error_i_1))) / len(traj_error_i_1))
        rmse_i_2 = np.sqrt(np.sum(np.square(np.array(traj_error_i_2))) / len(traj_error_i_2))
        
        rmse_1.append(rmse_i_1)
        rmse_2.append(rmse_i_2)

    
    print('evaluation of %s %s\n' % (dataset, data_id))

    if len(rmse_1) <=3:
        print('rmse %s:' % label1, rmse_1, np.average(rmse_1))
        print('rmse %s:' % label2, rmse_2, np.average(rmse_2), '\n')
    else:
        print('rmse %s:' % label1, np.average(rmse_1))
        print('rmse %s:' % label2, np.average(rmse_2), '\n')
    

    ########## evaluate object estimation ##########
    # find corresponding ground truth object for each estimated object
    gt_objects_1, gt_objects_2 = [], []
    obj_error_1, obj_error_2 = [], []
    for i in range(num_robot):
        gt_obj_i_1, gt_obj_i_2 = [], []
        obj_error_i_1, obj_error_i_2 = 0, 0

        for j in range(len(estim_objects_1[i])):

            gt_obj_i_1.append(gt_objects[int(estim_obj_ids_1[i][j])])
            obj_error_i_1 += np.linalg.norm(estim_objects_1[i][j] - gt_objects[int(estim_obj_ids_1[i][j])])
        for j in range(len(estim_objects_2[i])):
            
            gt_obj_i_2.append(gt_objects[int(estim_obj_ids_2[i][j])])
            obj_error_i_2 += np.linalg.norm(estim_objects_2[i][j] - gt_objects[int(estim_obj_ids_2[i][j])])
        
        gt_objects_1.append(np.array(gt_obj_i_1))
        gt_objects_2.append(np.array(gt_obj_i_2))

        if len(estim_objects_1[i]) > 0:
            obj_error_1.append(obj_error_i_1 / len(estim_objects_1[i]))
        else:
            obj_error_1.append(0)
        
        if len(estim_objects_2[i]) > 0:
            obj_error_2.append(obj_error_i_2 / len(estim_objects_2[i]))
        else:
            obj_error_2.append(0)

    obj_error_1, obj_error_2 = np.array(obj_error_1), np.array(obj_error_2)
    if len(obj_error_1) <= 3:
        print('obj error %s:' % label1, obj_error_1, np.average(obj_error_1))
        print('obj error %s:' % label2, obj_error_2, np.average(obj_error_2), '\n')
    else:
        print('obj error %s:' % label1, np.average(obj_error_1))
        print('obj error %s:' % label2, np.average(obj_error_2), '\n')

    ############ compute the object position difference ############
    obj_dist_1_arr, obj_dist_2_arr = [], []
    for i in range(num_robot):

        obj_dist_1_i, obj_dist_2_i = [], []
        for j in range(num_robot):
            if i == j:
                continue
            
            for k in range(len(estim_obj_ids_1[i])):
                obj_id_1 = estim_obj_ids_1[i][k]
                if obj_id_1 not in estim_obj_ids_1[j]:
                    continue
                
                idx_i = np.where(estim_obj_ids_1[i] == obj_id_1)[0][0]
                idx_j = np.where(estim_obj_ids_1[j] == obj_id_1)[0][0]

                dist = np.linalg.norm(estim_objects_1[i][idx_i] - estim_objects_1[j][idx_j])
                obj_dist_1_i.append(dist)

            for k in range(len(estim_obj_ids_2[i])):
                obj_id_2 = estim_obj_ids_2[i][k]
                if obj_id_2 not in estim_obj_ids_2[j]:
                    continue
                
                idx_i = np.where(estim_obj_ids_2[i] == obj_id_2)[0][0]
                idx_j = np.where(estim_obj_ids_2[j] == obj_id_2)[0][0]

                dist = np.linalg.norm(estim_objects_2[i][idx_i] - estim_objects_2[j][idx_j])
                obj_dist_2_i.append(dist)

        obj_dist_1_arr.append(np.array(obj_dist_1_i).mean())
        obj_dist_2_arr.append(np.array(obj_dist_2_i).mean())
    
    if len(obj_dist_1_arr) <= 3:
        print('obj dist %s:' % label1, obj_dist_1_arr, np.average(obj_dist_1_arr))
        print('obj dist %s:' % label2, obj_dist_2_arr, np.average(obj_dist_2_arr))
    else:
        print('obj dist %s:' % label1, np.average(obj_dist_1_arr))
        print('obj dist %s:' % label2, np.average(obj_dist_2_arr))

