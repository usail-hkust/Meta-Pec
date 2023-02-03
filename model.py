import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from model_utils import *
import learn2learn as l2l
from utils import *

# device conf
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

class Pec(Module):
    def __init__(self, opt):
        super(Pec, self).__init__()
        self.opt = opt
        self.dim = opt.dim
        self.traj_fea_dim = opt.traj_fea_dim
        self.max_traj_len = opt.max_traj_len
        self.max_tra_num = opt.max_tra_num
        self.road_dim = opt.road_dim
        self.road_fea_dim = opt.road_fea_dim
        self.road_emb_dim = opt.road_emb_dim
        self.filter_size = opt.filter_size
        self.drive_fea_dim = opt.drive_fea_dim

        # embedding
        self.car_fea_trans = nn.Linear(opt.car_fea_dim, opt.feature_emb_dim, bias=False)
        self.car_embedding = nn.Embedding(opt.car_num, opt.car_dim)
        self.car_type_embedding = nn.Embedding(opt.car_type_num, self.dim)
        self.traj_fea_trans = nn.Linear(opt.traj_fea_dim, self.dim, bias=False)
        self.road_fea_trans = nn.Linear(opt.road_fea_dim, opt.road_emb_dim, bias=False)
        self.road_type_embedding = nn.Embedding(opt.road_type_num, opt.road_type_dim)

        # hierarchy
        self.pre_learn = preference_learning(self.dim, self.filter_size, self.max_traj_len, opt.max_time, opt.max_location, dropout=opt.traj_drop, device=device)
        self.traj_GRU = nn.GRU(self.dim, self.dim, batch_first=True, dropout=opt.traj_drop)

        # drive feature decoder
        self.behavior_decoder = behavior_prediction(self.dim, opt.drive_fea_dim, opt.car_type_num, opt.head_num, opt.encoder_drop)

        # RNN
        self.GRU = nn.GRU(self.dim + opt.drive_fea_dim, self.dim, batch_first=True, dropout=opt.GRU_drop)

        # NN
        self.att = Soft_Att(self.dim)
        self.gate_1 = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.gate_2 = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.MLP = MLP(self.dim, opt.activation, opt.alpha)

        # loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) == 2:
                stdv = 1.0 / math.sqrt(weight.size(1))
                weight.data.uniform_(-stdv, stdv)

    def MAEDriveLoss(self, h, h_y):
        AE = torch.abs(h - h_y)
        AE = torch.mean(AE, dim=1)
        AE = torch.sum(AE, dim=-1)
        MAE = torch.mean(AE)

        return MAE

    def MAELoss(self, y_hat, y):
        AE = torch.abs(y_hat - y)
        MAE = torch.mean(AE)

        return MAE

    def CalculateTripRepresentation(self, h, h_road, h_drive, last_road_mask):
        h_road = torch.cat([h_road, h_drive], dim=-1)

        # RNN
        h_road, _ = self.GRU(h_road)
        h_road = torch.sum(last_road_mask.unsqueeze(-1) * h_road, dim=1)

        # concatenate
        gate_2 = torch.sigmoid(self.gate_2(torch.cat([h, h_road], dim=-1)))
        h = h * gate_2 + h_road * (1 - gate_2)

        # MLP
        h = self.MLP(h)

        return h

    def forward(self, cars, car_types, features, trajs, traj_times, traj_locations, road_types, road_features, mask, last_road_mask, y_road_drive, y):
        # car features
        h_cars = self.car_embedding(cars)
        h_car_types = self.car_type_embedding(car_types)
        h_fea = self.car_fea_trans(features)
        h = torch.cat([h_cars, h_fea], dim=1)

        # road features
        h_road_types = self.road_type_embedding(road_types)
        h_road = self.road_fea_trans(road_features)
        h_road = torch.cat([h_road_types, h_road], dim=2)

        # traj features
        traj_patch_mask = torch.sum(torch.sum(trajs.reshape(-1, int(self.max_traj_len / self.filter_size), self.filter_size, self.traj_fea_dim), dim=-1), dim=-1) != 0
        traj_patch_empty_mask = torch.sum(traj_patch_mask, dim=-1) == 0
        traj_patch_mask[traj_patch_empty_mask] = True
        traj_mask = torch.sum(torch.sum(trajs, dim=-1), dim=-1) == 0
        trajs = trajs.reshape(-1, self.max_traj_len, self.traj_fea_dim)
        h_trajs = self.traj_fea_trans(trajs).reshape(-1, self.max_tra_num, self.max_traj_len, self.dim)

        # traj tf
        h_traj = self.pre_learn(h_trajs, traj_times, traj_locations, ~traj_patch_mask)
        h_hist = h_traj.reshape(-1, self.max_tra_num, self.dim)
        h_car_trajs = self.att(h, h_hist, traj_mask)

        # gate 1
        gate_1 = torch.sigmoid(self.gate_1(torch.cat([h, h_car_trajs], dim=-1)))
        h = h * gate_1 + h_car_trajs * (1 - gate_1)

        # road drive feature prediction
        h_drive = self.behavior_decoder(h_road, h_hist, traj_mask, car_types)
        mask = mask.unsqueeze(-1)
        h_drive *= ~mask

        # get ground truth representation
        h = self.CalculateTripRepresentation(h, h_road, h_drive, last_road_mask)

        # prediction
        y_hat = torch.matmul(h.unsqueeze(1), h_car_types.unsqueeze(2)).squeeze()

        # calculate score & loss
        loss_drive = self.MAEDriveLoss(h_drive, y_road_drive)
        loss = self.MAELoss(y_hat, y)
        loss += loss_drive

        return loss, y, y_hat

class MetaLearner(object):
    def __init__(self, train_data, opt):
        super(MetaLearner).__init__()
        self.model = trans_to_cuda(Pec(opt))
        self.maml = l2l.algorithms.MAML(self.model, lr=opt.lr, allow_unused=True, first_order=True)
        self.optimizer = torch.optim.Adam(self.maml.parameters(), lr=opt.meta_lr, weight_decay=opt.l2)
        self.task_sampler = TaskBatchGenerator(train_data=train_data, opt=opt)
        self.opt = opt
        self.max_task_data_num = opt.max_task_data_num

    def train_tasks(self):
        sampler = self.task_sampler.getTaskBatch()
        for batch in tqdm(sampler):
            self.optimizer.zero_grad()
            task_num = 0
            total_loss = 0.0

            for data in batch:
                # data
                support_data, query_data = data
                learner = self.maml.clone()
                learner.train()

                # inner train_test
                for epoch in range(self.opt.base_epoch):
                    val_loss = self.train_valid_base_model(learner, support_data, query_data)
                    total_loss += val_loss
                    task_num += 1

            total_loss /= task_num
            total_loss.backward()
            self.optimizer.step()

    def train_valid_base_model(self, learner, support_data, query_data):
        support_loader = DataLoader(support_data, num_workers=0, batch_size=self.opt.batch_size, shuffle=True, pin_memory=False)
        for i, data in enumerate(support_loader):
            loss, targets, scores = forward(learner, data)
            learner.adapt(loss)

        val_loss = self.validation(learner, query_data)

        return val_loss

    def validation(self, learner, query_data):
        query_loader = DataLoader(query_data, num_workers=0, batch_size=self.opt.batch_size, shuffle=True, pin_memory=False)
        data_num = len(query_data)
        total_loss = 0.0

        for data in query_loader:
            loss, targets, scores = forward(learner, data)
            loss *= data_num / self.max_task_data_num
            total_loss += loss

        return total_loss


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    cars, car_types, features, trajs, traj_times, traj_locations, road_types, road_features, mask, last_road_mask, targets_drive, targets = data
    cars = trans_to_cuda(cars).long()
    car_types = trans_to_cuda(car_types).long()
    features = trans_to_cuda(features).float()
    trajs = trans_to_cuda(trajs).float()
    traj_times = trans_to_cuda(traj_times).float()
    traj_locations = trans_to_cuda(traj_locations).float()
    road_types = trans_to_cuda(road_types).long()
    road_features = trans_to_cuda(road_features).float()
    mask = trans_to_cuda(mask).bool()
    last_road_mask = trans_to_cuda(last_road_mask).bool()
    targets_drive = trans_to_cuda(targets_drive).float()
    targets = trans_to_cuda(targets).float()

    loss, targets, scores = model(cars, car_types, features, trajs, traj_times, traj_locations, road_types, road_features, mask, last_road_mask, targets_drive, targets)

    return loss, targets, scores


def SquareError(scores, targets):
    SE = torch.square(scores - targets)

    return SE

def AbsolError(scores, targets):
    AE = torch.abs(scores - targets)

    return AE

def AbsolPercentageError(scores, targets):
    targets = torch.abs(targets)
    AE = torch.abs(scores - targets)
    PAE = AE / targets

    return PAE

def train_test(model, opt, train_data, test_data, isMetaLearning=False):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = DataLoader(train_data, num_workers=0 if not isMetaLearning else 8, batch_size=opt.batch_size, shuffle=True, pin_memory=False)
    for i, data in enumerate(tqdm(train_loader)):
        model.optimizer.zero_grad()
        loss, targets, scores = forward(model, data)
        loss.backward()
        model.optimizer.step()
        total_loss += float(loss.item())
    print('\tLoss:\t%.3f' % total_loss)

    MSE, MAE, MAPE = test(model, opt, test_data, ~isMetaLearning)

    return MSE, MAE, MAPE

def test(model, opt, test_data, isTestset=True):
    if not isTestset:
        num_workers = 8
    else:
        num_workers = 0
    model.eval()
    test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=opt.batch_size, shuffle=False, pin_memory=False)

    SE = []
    AE = []
    APE = []
    total_loss = 0.0

    for data in test_loader:
        loss, targets, scores = forward(model, data)
        total_loss += float(loss.item())

        # MSE & MAE
        se = SquareError(scores, targets)
        SE.append(se)
        ae = AbsolError(scores, targets)
        AE.append(ae)
        ape = AbsolPercentageError(scores, targets)
        APE.append(ape)

    MSE = float(torch.mean(torch.cat(SE, dim=0)))
    MAE = float(torch.mean(torch.cat(AE, dim=0)))
    MAPE = float(torch.mean(torch.cat(APE, dim=0)))
    if isTestset:
        print('Test predicting: ', datetime.datetime.now(), '========== Test Loss:\t%.3f' % total_loss)
    else:
        print('Valid predicting: ', datetime.datetime.now(), '========== Valid Loss:\t%.3f' % total_loss)

    return MSE, MAE, MAPE
