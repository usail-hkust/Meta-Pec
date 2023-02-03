import copy
import numpy as np
import torch
import json
from torch.utils.data import Dataset


def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def dump_json(data, file, indent=None):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e


def sequence_embedding(seq, max_len, len_data, feature_num):
    re_seq = [upois + [[0.0] * feature_num] * (max_len - le) if le < max_len else upois[-max_len:] for upois, le in zip(seq, len_data)]

    return re_seq


def sequence(seq, max_len, len_data):
    mask = [[False] * le + [True] * (max_len-le) if le < max_len else [False] * max_len for le in len_data]
    last_road_mask = [[False] * (le - 1) + [True] + [False] * (max_len - le) if le < max_len else [False] * (max_len - 1) + [True] for le in len_data]
    re_seq = [upois + [0.0] * (max_len - le) if le < max_len else upois[-max_len:] for upois, le in zip(seq, len_data)]

    return re_seq, mask, last_road_mask


def handle_seq_data(road_types, road_features, road_embed_dim, train_len=None):
    len_data = [len(nowData) for nowData in road_types]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    re_road_type, mask, last_road_mask = sequence(road_types, max_len, len_data)
    re_road_feature = sequence_embedding(road_features, max_len, len_data, road_embed_dim)

    return re_road_type, re_road_feature, mask, last_road_mask, max_len

class Data(Dataset):
    def __init__(self, data, trajectories, opt, train_len=None):
        # embedding features
        cars, car_types, features, road_features, road_types, road_idx, fuel, traj_idxs = data
        road_types, road_features, mask, last_road_mask, max_len = handle_seq_data(road_types, road_features, opt.road_fea_dim_ori, train_len)
        self.cars = np.array(cars)
        self.car_types = np.array(car_types)
        self.features = np.array(features)
        self.road_types = np.array(road_types)
        self.road_features = np.array(road_features)
        self.mask = np.array(mask)
        self.last_road_mask = np.array(last_road_mask)
        self.targets = np.array(fuel)
        self.trajectories = trajectories
        self.traj_idxs = traj_idxs
        self.topk_trajs = []
        self.topk_trajs_times = []
        self.topk_trajs_locations = []
        self.max_len = max_len
        self.dataset = opt.dataset
        self.traj_fea_dim = opt.traj_fea_dim
        self.road_fea_dim = opt.road_fea_dim
        self.max_tra_num = opt.max_tra_num
        self.max_traj_len = opt.max_traj_len
        self.handleTrajectories()
        self.getUserTopkTrajs()
        self.road_idx = road_idx
        self.length = len(self.cars)

    def handleTrajectories(self):
        for c in self.trajectories:
            self.trajectories[c]['trajs'] = [traj + [[0.0] * self.traj_fea_dim] * (self.max_traj_len - len(traj)) if self.max_traj_len > len(traj)
                                             else traj[:self.max_traj_len] for traj in self.trajectories[c]['trajs']]
            self.trajectories[c]['times'] = [times + [0.0] * (self.max_traj_len - len(times)) if self.max_traj_len > len(times)
                                             else times[:self.max_traj_len] for times in self.trajectories[c]['times']]
            self.trajectories[c]['locations'] = [locations + [0.0] * (self.max_traj_len - len(locations)) if self.max_traj_len > len(locations)
                                             else locations[:self.max_traj_len] for locations in self.trajectories[c]['locations']]

    def getUserTopkTrajs(self):
        for i in range(len(self.traj_idxs)):
            car = self.cars[i]
            traj_idxs = self.traj_idxs[i]
            # trajectory
            c_traj = [self.trajectories[f'{car}']['trajs'][idx].copy() for idx in traj_idxs]
            c_times = [self.trajectories[f'{car}']['times'][idx].copy() for idx in traj_idxs]
            c_locations = [self.trajectories[f'{car}']['locations'][idx].copy() for idx in traj_idxs]

            traj_num = len(c_traj)
            c_traj = c_traj + [[[0.0] * self.traj_fea_dim] * self.max_traj_len] * (self.max_tra_num - traj_num) if self.max_tra_num > traj_num else c_traj[:self.max_tra_num]
            c_times = c_times + [[0.0] * self.max_traj_len] * (self.max_tra_num - traj_num) if self.max_tra_num > traj_num else c_times[:self.max_tra_num]
            c_locations = c_locations + [[0.0] * self.max_traj_len] * (self.max_tra_num - traj_num) if self.max_tra_num > traj_num else c_locations[:self.max_tra_num]

            self.topk_trajs.append(c_traj)
            self.topk_trajs_times.append(c_times)
            self.topk_trajs_locations.append(c_locations)

    def __getitem__(self, index):
        cars, car_types, features = self.cars[index], self.car_types[index], self.features[index]
        road_types, road_features, mask, last_road_mask = self.road_types[index], self.road_features[index], self.mask[index], self.last_road_mask[index],
        targets = self.targets[index]
        car_trajectories, car_traj_times, car_traj_locations = self.topk_trajs[index], self.topk_trajs_times[index], self.topk_trajs_locations[index]

        drive_features = road_features[:, 11:] if self.dataset == "VED" else road_features[:, 5:]
        road_features = road_features[:, :5]

        return [torch.tensor(cars), torch.tensor(car_types), torch.tensor(features), torch.tensor(car_trajectories),
                torch.tensor(car_traj_times), torch.tensor(car_traj_locations), torch.tensor(road_types),
                torch.tensor(road_features), torch.tensor(mask), torch.tensor(last_road_mask), drive_features, torch.tensor(targets)]

    def __len__(self):
        return self.length

class TaskBatchGenerator(object):
    def __init__(self, train_data, opt):
        super(TaskBatchGenerator).__init__()
        self.train_data = train_data
        self.data_num = len(self.train_data)
        self.batch_num = opt.task_batch_num
        self.batch_size = int(self.data_num / self.batch_num)

    def getTaskBatch(self):
        train_data = self.train_data.copy()
        np.random.shuffle(train_data)
        task_batches = []

        for b in range(self.batch_num):
            if b < self.batch_num - 1:
                task_batches.append(train_data[b * self.batch_size: (b+1) * self.batch_size])
            else:
                task_batches.append(train_data[b * self.batch_size:])

        return task_batches
