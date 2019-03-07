
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import gc
import os
cwd = os.getcwd()
sys.path.append(cwd+'/../networks/')
import torch
import time
import datetime
import socket
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
from data import build_train_dataset, build_test_dataset 
import util
import numpy as np
import argparse
import shutil
from datetime import datetime
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

from maskresnet import resnet, BinActive
import maskresnet

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true',
	help='set if only CPU is available')
parser.add_argument('--data', action='store', default='../lmdb_imagenet',
	help='dataset path')
parser.add_argument('--arch', action='store', default='resnet',
	help='the architecture for the network: nin')
parser.add_argument('--lr', action='store', type=float, default=0.001,
	help='the intial learning rate')
parser.add_argument('--epochs', action='store', default='5',
	help='fisrt train epochs',type=float)
parser.add_argument('--retrain_epochs', action='store', default='5',
	help='re-train epochs',type=int)
parser.add_argument('--print_freq', action='store', default=10,
	help='re-train epochs',type=int)
parser.add_argument('--save_name', action='store', default='first_model',
	help='save the first trained model',type=str)
parser.add_argument('--load_name', action='store', default='first_model',
	help='load pretrained model',type=str)
parser.add_argument('--root_dir', action='store', default='./model_resnet_beta_final/',
	help='root dir for different experiments',type=str)
parser.add_argument('--pretrained', action='store', default=None,
	help='the path to the pretrained model')
parser.add_argument('--evaluate', action='store_true',
	help='evaluate the model')
parser.add_argument('--resume', type=str, default='../model.tar',
	help='resume the model from checkpoint')
parser.add_argument('--partpal', 
                    default=0, type=int, help='parallel mode')

###>> Train Mask
parser.add_argument('--finetune_weight',type=bool,default=False)
parser.add_argument('--alpha',action='store',type=float, default=1e-8,
	help='regularization loss')
parser.add_argument('--initialnum',action='store',default=None)
parser.add_argument('--decay',type=float,default=1)
parser.add_argument('--direction',type=bool,default=False)
###>>

args = parser.parse_args()
print('==> Options:',args)
print(args.direction)

best_prec1 = 0
train_dataset_size = 1281167
test_dataset_size = 50000
class_size = 1000

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)


############check################
def ifornot_dir(directory):
##please give current dir
	import os
	if not os.path.exists(directory):
		os.makedirs(directory)

ifornot_dir(args.root_dir)

###########Data Augmentation#####
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/data',
                                        train=True, 
                                        download=False, 
                                        transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data', 
                                       train=False,
                                       download=False, 
                                       transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
	lr = float(args.lr) * (0.1 ** int(epoch / 2))
	print('Learning rate:', lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
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

def save_state(state, is_best, save_name, root_dir):
	print('==> Saving model ...')
	torch.save(state, root_dir + '/' + save_name+'.pth.tar')
	if is_best:
		shutil.copyfile(filename, root_dir+save_name+'_best.pth.tar')

def train(epoch, sample_weights=torch.Tensor(np.ones((50000,1))/50000.0)):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	###>>
	alpha = args.alpha if not args.finetune_weight else 0 
	###>>

	# switch to train mode
	model.train()

	end = time.clock()

	trainloader = build_train_dataset('sample_batch', sample_weights)
	
	#print('==> Starting one epoch ...')
	for batch_idx, (data, target, _) in enumerate(trainloader):
		data_time.update(time.clock() - end)
		bin_op.binarization()
		data, target = Variable(data.cuda()), Variable(target.cuda())
		optimizer.zero_grad()
		output = model(data)
		# backwarding
		loss = criterion(output, target)
		prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
		
		top1.update(prec1[0], data.size(0))
		top5.update(prec5[0], data.size(0))
		
		if not args.finetune_weight:
			count,regloss,mask_lists = 0,0,[]
			for m in model.modules():
				if isinstance(m, maskresnet.BinConv2d):
					if m.mask.requires_grad == True:
						filtermask = BinActive()(m.mask)
						mask_lists.append(filtermask)
					count += 1
			direction = 1 if args.direction else 0
			regloss = sum(list(map(lambda x: torch.norm(x+direction,1),mask_lists)))
		else:
			regloss = torch.Tensor([0]).cuda()
		loss = loss + alpha*regloss 

		losses.update(loss.data[0], data.size(0))

		loss.backward()
		# restore weights
		bin_op.restore()
		bin_op.updateBinaryGradWeight()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.clock() - end)
		end = time.clock()

		all_batch_idx = (epoch-1) * len(trainloader) + batch_idx
		if batch_idx % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
			'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
			'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
			'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
			'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
			epoch, batch_idx, len(trainloader), batch_time=batch_time,
			data_time=data_time, loss=losses, top1=top1, top5=top5))
		if epoch <= 10:
			if batch_idx % (args.print_freq * 50) == 0 and batch_idx != 0:
				eval_info = eval_test()
				save_state({
						'arch': args.arch,
						'state_dict': model.state_dict(),
						'best_prec1': best_prec1,
						'optimizer' : optimizer.state_dict(),
					}, False, 'sample_test_{:}'.format((epoch - 1) * 5 + (batch_idx / (args.print_freq * 50))), checkpointdir)
				get_prun_number((epoch - 1) * 5 + (batch_idx / (args.print_freq * 50)))
	
	gc.collect()
	return trainloader

def test(save_name, best_prec1, testloader_in, epoch):

	#global best_acc

	model.eval()
	test_loss = 0
	correct = 0
	bin_op.binarization()

	for i, (data, target) in enumerate(testloader_in):
		data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)                         
		output = model(data)
		test_loss += criterion(output, target).data[0]
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	bin_op.restore()
	acc = 100. * correct.item() / len(testloader_in.dataset)
	test_loss /= len(testloader_in.dataset)
	print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
		test_loss * 1000, correct.item(), len(testloader_in.dataset),
		100. * correct.item() / len(testloader_in.dataset)))
	
	if acc > best_prec1:
		save_state({
		'arch': args.arch,
		'state_dict': model.state_dict(),
		'best_prec1': best_prec1,
		'optimizer' : optimizer.state_dict(),
		}, False, 'best', checkpointdir)

	save_state({
		'arch': args.arch,
		'state_dict': model.state_dict(),
		'best_prec1': best_prec1,
		'optimizer' : optimizer.state_dict(),
		}, False, 'vaild_{:}'.format(epoch), checkpointdir)

	writer.add_scalar('vali_acc_top1/valid_top1', acc, epoch )
	writer.add_scalar('vali_acc_loss/valid_loss', test_loss, epoch )

	best_prec1 = max(acc, best_prec1)
	return best_prec1

def get_prun_number(epoch):
    count = 0#,prun_num = 0,0
    for m in model.modules():
        prun_num = 0
        if isinstance(m,maskresnet.BinConv2d):
            if count == int(args.initialnum):
                prun_num += BinActive()(m.mask).eq(-1).sum()
            writer.add_scalar('prun_num/' + str(count), prun_num, epoch)
            count += 1

def eval_test():

	model.eval()
	test_loss = 0
	correct = 0
	bin_op.binarization()

	testloader = build_test_dataset()
	for i, (data, target) in enumerate(testloader):
		data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)                         
		output = model(data)
		test_loss += criterion(output, target).data[0]
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	bin_op.restore()
	acc = 100. * correct.item() / len(testloader.dataset)
	test_loss /= len(testloader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
		test_loss * 1000, correct.item(), len(testloader.dataset),
		100. * correct.item() / len(testloader.dataset)))

	return acc, test_loss


def reset_learning_rate(optimizer, lr=0.001):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return

# define the model

def model_components():
	print('==> building model',args.arch,'...')
	if args.arch == 'resnet':
		model = resnet('ResNet_imagenet', pretrained=args.pretrained, num_classes=1000, depth=18, dataset='imagenet')
	else:
		raise Exception(args.arch+' is currently not supported')

	
	#load model
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			new_state_dict = checkpoint['state_dict']
			for k in list(new_state_dict.keys()):
			    if 'weight'  in k:
			        new_state_dict[k.replace('conv.weight','weight')] = new_state_dict.pop(k)
			    if 'bias' in k:
			        new_state_dict[k.replace('conv.bias','bias')] = new_state_dict.pop(k)
			#jown_state = model.state_dict()
			own_state = model.state_dict()
			for name, param in own_state.items():
				if name in new_state_dict:
					own_state[name].copy_(new_state_dict[name])
				else:
					print(name)
			del checkpoint
		else:
			raise Exception(args.resume+' is found.')
	else:
		print('==> Initializing model parameters ...')
		for m in model.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				c = float(m.weight.data[0].nelement())
				m.weight.data.normal_(0, 1./c)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					m.weight.data = m.weight.data.zero_().add(1.0)

	#data parallel
	if args.partpal==1:
		model.features = torch.nn.DataParallel(model.features)
		model.cuda()
	else:
		model = torch.nn.DataParallel(model).cuda()
	cudnn.benchmark = True

	# define solver and criterio
	optimizer = optim.Adam(filter(lambda para:para.requires_grad,model.parameters()), lr=args.lr, weight_decay=0.000001)

	criterion = nn.CrossEntropyLoss().cuda()
	criterion_seperated = nn.CrossEntropyLoss(reduce=False).cuda()
	# define the binarization operator
	bin_op = util.BinOp(model, 'FL_Full', True if args.finetune_weight else False)
	print('bin_op',len(bin_op.target_modules))
	return model, optimizer, criterion, criterion_seperated, bin_op

def generate_mask():
	count = 0
	state_list = []
	for m in model.modules():
		if isinstance(m,maskresnet.BinConv2d):
			if count == int(args.initialnum):
				m.binact = True
			state_list.append(str(int(m.binact)))
			count += 1
 

def get_grad_state():
    if str(args.initialnum):
        if not args.finetune_weight:
            for name,para in model.named_parameters():
                para.requires_grad = False
            count = 0
            for module in model.modules():
                if isinstance(module,maskresnet.BinConv2d):
                    if count == int(args.initialnum):
                        module.mask.requires_grad = True 
                    count += 1 
    else:
        for name,para in model.named_parameter():
            if 'mask' in name:
                para.requires_grad=False 


if __name__ == '__main__':
        #test
    model, optimizer, criterion, criterion_seperated, bin_op = model_components()

    train_batch_size = 1000
    test_batch_size = 1000
    total_classes = 1000
    best_acc = 0

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    generate_mask()
    get_grad_state()

    for epoch in range(1, int(args.epochs)+1):
        train(epoch)
        best_acc = test(args.save_name, best_acc, build_test_dataset(), epoch)
