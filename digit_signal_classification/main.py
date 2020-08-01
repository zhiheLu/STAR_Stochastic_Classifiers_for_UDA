from __future__ import print_function
import argparse
import torch
from solver import Solver
import os
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Stochastic Classifier Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.02)')
parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper parameter for generator update')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='N',
                    help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--source', type=str, default='svhn', metavar='N',
                    help='source dataset')
parser.add_argument('--target', type=str, default='mnist', metavar='N',
                    help='target dataset')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
parser.add_argument('--num_classifiers_train', type=int, default=2, metavar='N',
                    help='the number of classifiers used in training')
parser.add_argument('--num_classifiers_test', type=int, default=20, metavar='N',
                    help='the number of classifiers used in testing')
parser.add_argument('--gpu_devices', type=str, default='0', help='the device you use')
parser.add_argument('--loss_process', type=str, default='sum', 
                    help='mean or sum of the loss')
parser.add_argument('--log_dir', type=str, default='record', metavar='N',
                    help='the place to store the logs')
parser.add_argument('--init', type=str, default='kaiming_u', metavar='N',
                    help='the initialization method')
parser.add_argument('--use_init', action='store_true', default=False,
                    help='whether use initialization')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.cuda and args.gpu_devices:
    torch.cuda.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

print(args)


def main():
    solver = Solver(
        args, source=args.source,
        target=args.target,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        num_k=args.num_k,
        all_use=args.all_use,
        checkpoint_dir=args.checkpoint_dir,
        save_epoch=args.save_epoch,
        num_classifiers_train=args.num_classifiers_train,
        num_classifiers_test=args.num_classifiers_test,
        init=args.init,
        use_init=args.use_init
    )

    record_time = time.strftime('%Y-%m-%d-%H-%M-%S')

    if args.use_init:
        record_dir = '{}/{}_{}'.format(
            args.log_dir,
            args.source,
            args.target
        )
    else:
        record_dir = '{}/{}_{}'.format(
            args.log_dir,
            args.source,
            args.target,
        )

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)

    record_train = '{}/train_{}.out'.format(record_dir, record_time)
    record_test = '{}/test_{}.out'.format(record_dir, record_time)

    # Log the configures into log file
    record = open(record_test, 'a')
    record.write('Configures: {} \n'.format(args))
    record.close()

    if args.eval_only:
        solver.test(0)
    else:
        for epoch in range(args.max_epoch):
            solver.train(epoch, record_file=record_train, loss_process=args.loss_process)

            if epoch % 1 == 0:
                solver.test(epoch, record_file=record_test, save_model=args.save_model)


if __name__ == '__main__':
    main()
