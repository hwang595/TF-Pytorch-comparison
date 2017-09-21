import sys
import math
import threading
import argparse
import time

import torch
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors
from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
import torch.distributed as dist

import torch.nn as nn
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply
import torch.nn.functional as F

from torchvision import datasets, transforms

from resnet import *

'''this is a trial code, we use Cifar10 on ResNet-32 with batch normalization'''

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    
    print('')
    if not isinstance(grad_input[0], type(None)):
        print('grad_input size:', grad_input[0].size())
        print('grad_input norm:', grad_input[0].data.norm())
    print('grad_output size:', grad_output[0].size())
    #print('grad_output size:', grad_output)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def naive_lr_scheduler(optimizer, lr_decay=0.1):
    """we use this naive learning rate scheduler to decay the learning rate in resnet"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--enable-gpu', type=bool, default=False, help='using GPU or CPU')
    args = parser.parse_args()
    return args


class ResNet_Learner:
    def __init__(self, rank, world_size, args):
        self._step_changed = False
        self._update_step = False
        self._new_step_queued = 0
        self._rank = rank
        self._world_size = world_size
        self._cur_step = 0
        self._next_step = self._cur_step + 1
        # we use this counter to determine when we should decay the learning rate
        self._iteration_counter = -1
        self._step_fetch_request = False
        self.max_num_epochs = args.epochs
        self.lr = args.lr
        # using GPU or CPU
        self.enable_gpu = args.enable_gpu
        self.momentum = args.momentum


    def build_model(self):
        self.network = resnet20()

        # only for test use
        self.module = self.network

        self.criterion = nn.CrossEntropyLoss()

        # this is only used for test
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)

    def train_and_test(self, train_loader=None, test_loader=None):
        # iterate of epochs
        for i in range(self.max_num_epochs):
            self.network.train()
            epoch_start_time = time.time()
            batch_counter = 0
            for batch_idx, (train_images, y_batch) in enumerate(train_loader):
                batch_counter += len(train_images)
                iteration_start_time = time.time()
                self._iteration_counter += 1

                if self.enable_gpu:
                    train_images = Variable(train_images.cuda())
                    train_labels = Variable(y_batch.cuda())
                else: 
                    train_images = Variable(train_images)
                    train_labels = Variable(y_batch)

                self.optimizer.zero_grad()

                logits = self.network(train_images)
                loss = self.criterion(logits, train_labels)
                
                loss.backward()
                #backward(loss)

                self.optimizer.step()

                # load the training info
                print('Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {}, Time Cost: {}'.format(
                       i, batch_counter, len(train_loader.dataset), 
                      100. * batch_idx / len(train_loader), loss.data[0], 
                      time.time()-iteration_start_time))
                
                if self._iteration_counter == 40000 or self._iteration_counter == 60000:
                    naive_lr_scheduler(optimizer=self.optimizer)
            # on batch end
            # change model to eval mode first
            self.network.eval()
            test_loss = 0
            # we use batched strategy and gather them together to avoid run out of memory issue
            #logits_collector = []
            #labels_colloector = []
            for test_batch_idx, (test_images, test_labels) in enumerate(test_loader):
                if self.enable_gpu:
                    test_images = Variable(test_images.cuda(), volatile=True)
                    test_labels = Variable(test_labels.cuda(), volatile=True)
                else:
                    test_images = Variable(test_images, volatile=True)
                    test_labels = Variable(test_labels, volatile=True)

                test_logits = self.network(test_images)

                if test_batch_idx == 0:
                    logits_collector=test_logits.data
                    labels_colloector=test_labels.data
                else:
                    logits_collector=torch.cat((logits_collector, test_logits.data), dim=0)
                    labels_colloector=torch.cat((labels_colloector, test_labels.data), dim=0)
                test_loss += self.criterion(test_logits, test_labels).data[0] # sum up batch loss

            prec1 = accuracy(logits_collector, labels_colloector)

            print('Epoch: %s  Step: %d  Top-1-Error @ 1: %f Loss: %f Time: %f' %
                (str(i), i*len(train_loader), (100.0-prec1[0].numpy()[0])/100.0, test_loss, (time.time()-epoch_start_time)))


if __name__ == "__main__":
    args = add_fit_args(argparse.ArgumentParser(description='PyTorch MNIST Example'))

    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    '''

    # load training and test set here:
    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False)

    resnet_learner = ResNet_Learner(rank=0, world_size=1, args=args)
    resnet_learner.build_model()

    if args.enable_gpu:
        resnet_learner.network.cuda()

    resnet_learner.train_and_test(train_loader=train_loader, test_loader=test_loader)