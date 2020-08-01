from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.build_gen import *
from datasets.dataset_read import dataset_read
from utils.avgmeter import AverageMeter
import time
from tensorboardX import SummaryWriter


# Training settings
class Solver(object):
    def __init__(
            self, 
            args,
            batch_size=64, 
            source='svhn', 
            target='mnist',
            learning_rate=0.0002, 
            interval=100, 
            optimizer='adam', 
            num_k=4,
            all_use=False, 
            checkpoint_dir=None, 
            save_epoch=10,
            num_classifiers_train=2, 
            num_classifiers_test=20,
            init='kaiming_u', 
            use_init=False, 
            dis_metric='L1'
    ):

        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.num_classifiers_train = num_classifiers_train
        self.num_classifiers_test = num_classifiers_test
        self.init = init
        self.dis_metric = dis_metric
        self.use_init = use_init

        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False

        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(
            source, target,
            self.batch_size,
            scale=self.scale,
            all_use=self.all_use
        )
        print('load finished!')

        self.G = Generator(source=source, target=target)
        self.C = Classifier(
            source=source, target=target,
            num_classifiers_train=self.num_classifiers_train,
            num_classifiers_test=self.num_classifiers_test,
            init=self.init,
            use_init=self.use_init
        )

        if args.eval_only:
            self.G.torch.load('{}/{}_to_{}_model_epoch{}_G.pt'.format(
                self.checkpoint_dir,
                self.source,
                self.target,
                args.resume_epoch)
            )

            self.C.torch.load('{}/{}_to_{}_model_epoch{}_C.pt'.format(
                self.checkpoint_dir,
                self.source,
                self.target,
                args.resume_epoch)
            )

        self.G.cuda()
        self.C.cuda()
        self.interval = interval
        self.writer = SummaryWriter()

        self.opt_c, self.opt_g = self.set_optimizer(
            which_opt=optimizer,
            lr=learning_rate
        )
        self.lr = learning_rate

        # Learning rate scheduler
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g,
            float(args.max_epoch)
        )
        self.scheduler_c = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_c,
            float(args.max_epoch)
        )

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(
                self.G.parameters(),
                lr=lr, 
                weight_decay=0.0005,
                momentum=momentum
            )

            self.opt_c = optim.SGD(
                self.C.parameters(),
                lr=lr, 
                weight_decay=0.0005,
                momentum=momentum
            )

        if which_opt == 'adam':
            self.opt_g = optim.Adam(
                self.G.parameters(),
                lr=lr, 
                weight_decay=0.0005,
                amsgrad=False
            )

            self.opt_c = optim.Adam(
                self.C.parameters(),
                lr=lr, 
                weight_decay=0.0005,
                amsgrad=False
            )

        return self.opt_c, self.opt_g

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()

    @staticmethod
    def entropy(x):
        b = F.softmax(x) * F.log_softmax(x)
        b = -1.0 * b.sum()
        return b

    @staticmethod
    def discrepancy(out1, out2):
        l1loss = torch.nn.L1Loss()
        return l1loss(F.softmax(out1, dim=1), F.softmax(out2, dim=1))

    @staticmethod
    def discrepancy_mse(out1, out2):
        mseloss = torch.nn.MSELoss()
        return mseloss(F.softmax(out1, dim=1), F.softmax(out2, dim=1))

    @staticmethod
    def discrepancy_cos(out1, out2):
        cosloss = torch.nn.CosineSimilarity()
        return 1 - cosloss(F.softmax(out1, dim=1), F.softmax(out2, dim=1))

    @staticmethod
    def discrepancy_slice_wasserstein(p1, p2):
        p1 = torch.sigmoid(p1)
        p2 = torch.sigmoid(p2)
        s = p1.shape
        if s[1] > 1:
            proj = torch.randn(s[1], 128).cuda()
            proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
            p1 = torch.matmul(p1, proj)
            p2 = torch.matmul(p2, proj)
        p1 = torch.topk(p1, s[0], dim=0)[0]
        p2 = torch.topk(p2, s[0], dim=0)[0]
        dist = p1 - p2
        wdist = torch.mean(torch.mul(dist, dist))

        return wdist

    def train(self, epoch, record_file=None, loss_process='mean', func='L1'):
        criterion = nn.CrossEntropyLoss().cuda()

        # Various measurements for the discrepancy
        dis_dict = {
            'L1': self.discrepancy,
            'MSE': self.discrepancy_mse,
            'Cosine': self.discrepancy_cos,
            'SWD': self.discrepancy_slice_wasserstein
        }

        self.G.train()
        self.C.train()
        torch.cuda.manual_seed(1)
        batch_time = AverageMeter()
        data_time = AverageMeter()

        batch_num = min(
            len(self.datasets.data_loader_A), 
            len(self.datasets.data_loader_B)
        )

        end = time.time()

        for batch_idx, data in enumerate(self.datasets):
            data_time.update(time.time() - end)
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']

            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break

            img_s = img_s.cuda()
            img_t = img_t.cuda()

            imgs_st = torch.cat((img_s, img_t), dim=0)

            label_s = label_s.long().cuda()

            # Step1: update the whole network using source data
            self.reset_grad()
            feat_s = self.G(img_s)
            outputs_s = self.C(feat_s)

            loss_s = []
            for index_tr in range(self.num_classifiers_train):
                loss_s.append(criterion(outputs_s[index_tr], label_s))

            if loss_process == 'mean':
                loss_s = torch.stack(loss_s).mean()
            else:
                loss_s = torch.stack(loss_s).sum()

            loss_s.backward()
            self.opt_g.step()
            self.opt_c.step()

            # Step2: update the classifiers using target data
            self.reset_grad()
            feat_st = self.G(imgs_st)
            outputs_st = self.C(feat_st)
            outputs_s = [
                outputs_st[0][:self.batch_size], 
                outputs_st[1][:self.batch_size]
            ]
            outputs_t = [
                outputs_st[0][self.batch_size:], 
                outputs_st[1][self.batch_size:]
            ]

            loss_s = []
            loss_dis = []
            for index_tr in range(self.num_classifiers_train):
                loss_s.append(criterion(outputs_s[index_tr], label_s))

            if loss_process == 'mean':
                loss_s = torch.stack(loss_s).mean()
            else:
                loss_s = torch.stack(loss_s).sum()

            for index_tr in range(self.num_classifiers_train):
                for index_tre in range(index_tr + 1, self.num_classifiers_train):
                    loss_dis.append(dis_dict[func](outputs_t[index_tr], outputs_t[index_tre]))

            if loss_process == 'mean':
                loss_dis = torch.stack(loss_dis).mean()
            else:
                loss_dis = torch.stack(loss_dis).sum()

            loss = loss_s - loss_dis

            loss.backward()
            self.opt_c.step()

            # Step3: update the generator using target data
            self.reset_grad()

            for index in range(self.num_k+1):
                loss_dis = []
                feat_t = self.G(img_t)
                outputs_t = self.C(feat_t)

                for index_tr in range(self.num_classifiers_train):
                    for index_tre in range(index_tr + 1, self.num_classifiers_train):
                        loss_dis.append(dis_dict[func](outputs_t[index_tr], outputs_t[index_tre]))

                if loss_process == 'mean':
                    loss_dis = torch.stack(loss_dis).mean()
                else:
                    loss_dis = torch.stack(loss_dis).sum()

                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()

            batch_time.update(time.time() - end)

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{}]\t '
                      'Loss: {:.6f}\t '
                      'Discrepancy: {:.6f} \t '
                      'Lr C: {:.6f}\t'
                      'Lr G: {:.6f}\t'
                      'Time: {:.3f}({:.3f})\t'
                      .format(epoch + 1, batch_idx,
                              batch_num, loss_s.data,
                              loss_dis.data,
                              self.opt_c.param_groups[0]['lr'],
                              self.opt_g.param_groups[0]['lr'],
                              batch_time.val, batch_time.avg))

                if record_file:
                    record = open(record_file, 'a')
                    record.write('Dis Loss: {}, Cls Loss: {}, Lr C: {}, Lr G: {} \n'
                                 .format(loss_dis.data.cpu().numpy(),
                                         loss_s.data.cpu().numpy(),
                                         self.opt_c.param_groups[0]['lr'],
                                         self.opt_g.param_groups[0]['lr']))
                    record.close()

    def test(self, epoch, record_file=None, save_model=False):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.eval()
        self.C.eval()
        test_loss = 0
        correct = 0
        size = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.dataset_test):
                img = data['T']
                label = data['T_label']
                img, label = img.cuda(), label.long().cuda()
                feat = self.G(img)
                outputs = self.C(feat)
                test_loss += criterion(outputs[0], label).data
                k = label.data.size()[0]
                output_ensemble = torch.zeros(outputs[0].shape).cuda()

                for index in range(len(outputs)):
                    output_ensemble += outputs[index]

                pred_ensemble = output_ensemble.data.max(1)[1]
                correct += pred_ensemble.eq(label.data).cpu().sum()
                size += k
            test_loss = test_loss / size

        print('\nTest set: Average loss: {:.4f}\t Ensemble Accuracy: {}/{} ({:.2f}%)'
              .format(test_loss, correct, size, 100. * float(correct) / size))

        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G, '{}/{}_to_{}_model_epoch{}_G.pt'
                       .format(self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C, '{}/{}_to_{}_model_epoch{}_C.pt'
                       .format(self.checkpoint_dir, self.source, self.target, epoch))

        if record_file:
            record = open(record_file, 'a')
            print('Recording {}'.format(record_file))
            record.write('Accuracy: {:.2f}'.format(100. * float(correct) / size))
            record.write('\n')
            record.close()

        self.writer.add_scalar('Test/loss', test_loss, epoch)
        self.writer.add_scalar('Test/ACC_en', 100. * float(correct) / size, epoch)
