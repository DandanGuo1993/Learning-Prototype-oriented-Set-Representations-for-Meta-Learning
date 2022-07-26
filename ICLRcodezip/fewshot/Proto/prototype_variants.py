import torch.optim
import os
import configs
from data.datamgr import SimpleDataManager, SetDataManager
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import model_dict, get_resume_file
from abc import abstractmethod
from sinkhorn import SinkhornDistance

eplison = 0.1
sk = SinkhornDistance(eps=eplison, max_iter=200, reduction=None).cuda()


def OT_loss(input, proportion, center):
    train_batch = input.reshape(-1, input.shape[1], center.shape[1])
    w_i_true = torch.mean(sk(train_batch, center, proportion))
    return w_i_true


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # 
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  

    @abstractmethod
    def set_forward(self, x, is_feature, is_train):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x, is_train=False)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, ot_type):
        print_freq = 500

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.data.item()

            if i % print_freq == 0:
                print('{}==> Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(ot_type, epoch, i, len(train_loader),
                                                                              avg_loss / float(i + 1)))

    def test_loop(self, test_loader, ot_type):
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            # print(i)  # for debug
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%s ==> %d Test Acc = %4.2f%% +- %4.2f%%' % (ot_type, iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, hidden_dim, center_num, n_way, n_support, w_ot=False):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

        self.w_ot = w_ot
        if self.w_ot:
            center_pre = torch.empty((center_num, hidden_dim), requires_grad=True)
            nn.init.normal_(center_pre, 0, 1)
            self.centers = nn.Parameter(center_pre)

            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, int(hidden_dim / 2)),
                nn.LeakyReLU(True),
                nn.Linear(int(hidden_dim / 2), center_num),
                nn.Softmax(dim=1))

    def set_forward(self,x, is_feature = False, is_train=True):
        z_support, z_query  = self.parse_feature(x,is_feature)

        if self.w_ot:
            if is_train:
                z = torch.cat([z_support.contiguous(), z_query.contiguous()], dim=1)
                z_support = z_support.contiguous()
                z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  
                z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
                dists = euclidean_dist(z_query, z_proto)
                scores = -dists

                z_mean = z.mean(dim=1)
                ot_out = self.regressor(z_mean)
                return scores, z, ot_out
            else:
                z_support = z_support.contiguous()
                z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  
                z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

                dists = euclidean_dist(z_query, z_proto)
                scores = -dists
                return scores
        else:
            z_support   = z_support.contiguous()
            z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) 
            z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

            dists = euclidean_dist(z_query, z_proto)
            scores = -dists
            return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda()).long()

        if self.w_ot:
            scores, z, ot_out = self.set_forward(x)
        else:
            scores = self.set_forward(x)

        loss = self.loss_fn(scores, y_query)
        if self.w_ot:
            loss += OT_loss(z, ot_out, self.centers) + 2 * torch.mean(ot_out * torch.log(ot_out + 1e-10))
        return loss


def euclidean_dist( x, y):

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def test(test_loader, model, model_ot):
    acc = model.test_loop(test_loader, 'Without OT')
    acc_ot = 0.
    return acc, acc_ot


def train(base_loader, val_loader, model, model_ot, optimization, start_epoch, stop_epoch, checkpoint_dir,
          checkpoint_dir_ot, save_freq):
    print_freq = 1

    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
        optimizer_ot = torch.optim.Adam(model_ot.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0
    max_acc_ot = 0

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, optimizer, 'Without OT')  
        model.eval()

        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if epoch % print_freq == 0:
            acc = model.test_loop(val_loader, 'Without OT')
            if acc > max_acc:
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

            if (epoch % save_freq == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

 

    return model, model_ot


if __name__ == '__main__':
    dataset = 'miniImagenet'  
    backbone_name = 'ResNet10'  
    method_name = 'protonet'

    train_n_way = 5
    test_n_way = 5
    train_episode = 1000
    test_episode = 600
    n_shot = 10

    save_freq = 5
    start_epoch = 0
    stop_epoch = 40
    save_iter = -1  
    optimization = 'Adam'

    resume = False
    train_aug = True

    is_train = True

    np.random.seed(10)

    base_file = configs.data_dir[dataset] + 'base.json'
    if is_train:
        val_file = configs.data_dir[dataset] + 'val.json'
    else:
        val_file = configs.data_dir[dataset] + 'novel.json'

    center_num = 64
    if 'Conv' in backbone_name:
        hidden_dim = 1600
        image_size = 84
    elif 'ResNet' in backbone_name:
        hidden_dim = 512
        image_size = 224
    else:
        raise NotImplementedError

    n_query = max(1, int(16 * test_n_way / train_n_way))  

    train_few_shot_params = dict(n_way=train_n_way, n_support=n_shot)
    base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params, n_eposide=train_episode)
    base_loader = base_datamgr.get_data_loader(base_file, aug=train_aug)

    test_few_shot_params = dict(n_way=test_n_way, n_support=n_shot)
    val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params, n_eposide=test_episode)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    proto = ProtoNet(model_dict[backbone_name], hidden_dim, center_num, **train_few_shot_params, w_ot=False).cuda()
    proto_ot = ProtoNet(model_dict[backbone_name], hidden_dim, center_num, **train_few_shot_params, w_ot=True).cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s_center64' % (configs.save_dir, dataset, backbone_name, method_name)
    if train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' % (train_n_way, n_shot)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_dir_ot = '%s/checkpoints/%s/%s_%s_ot_center64' % (configs.save_dir, dataset, backbone_name, method_name)
    if train_aug:
        checkpoint_dir_ot += '_aug'
    checkpoint_dir_ot += '_%dway_%dshot' % (train_n_way, n_shot)
    if not os.path.isdir(checkpoint_dir_ot):
        os.makedirs(checkpoint_dir_ot)

    if is_train:
        if resume:
            resume_file = get_resume_file(checkpoint_dir)
            if resume_file is not None:
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                proto.load_state_dict(tmp['state'])
            resume_file = get_resume_file(checkpoint_dir_ot)
            if resume_file is not None:
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                proto_ot.load_state_dict(tmp['state'])

        proto , proto_ot= train(base_loader, val_loader, proto, proto_ot, optimization, start_epoch, stop_epoch,
                                checkpoint_dir, checkpoint_dir_ot, save_freq)
    else:
        if resume:
            resume_file = get_resume_file(checkpoint_dir, is_train)
            if resume_file is not None:
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                proto.load_state_dict(tmp['state'])
            else:
                raise ValueError('Not a trained model')
            resume_file = get_resume_file(checkpoint_dir_ot, is_train)
            if resume_file is not None:
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                proto_ot.load_state_dict(tmp['state'])
            else:
                raise ValueError('Not a trained model')

            acc, acc_ot = test(val_loader, proto, proto_ot)
        else:
            raise NotImplementedError
