
import os
import socket
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from tensorboardX import SummaryWriter
from model import NIN
from model import MaskBinActiveConv2d
import util



def save_state(model, acc, idx=-1):
    print('==> Saving model ...')
    state = {
        'acc': acc,
        'state_dict': model.state_dict(),
    }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)
    torch.save(state, 'modulesresidual/nin_finetune_weight_{:}.pth.tar'.format('best' if idx == -1 else str(idx)))


def train(epoch, model):
    global l1_loss
    model.train()
    sumloss = 0

    mask_lists = []
    for m in model.modules():
        if isinstance(m, MaskBinActiveConv2d):
            mask_lists.append(m.mask)

    for batch_idx, (data, target) in enumerate(trainloader):
        # process the weights including binarization
        bin_op.binarization()

        # forwarding
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        # backwarding
        alpha = 1e-6
        loss = criterion(output, target) + alpha * sum(list(map(lambda x: torch.norm(x.abs(), 1), mask_lists)))
        loss.backward()
        sumloss = sumloss + loss.data[0]

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.data[0],
                optimizer.param_groups[0]['lr']))

    writer.add_scalar('train_loss', sumloss / len(trainloader), epoch)
    return


def test(epoch, model):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = data.cuda(), target.cuda()

        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    if epoch % 5 == 0:
        save_state(model, best_acc, epoch)

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    writer.add_scalar('vali_acc', acc, epoch)
    return


def adjust_learning_rate(optimizer, epoch):
    # update_list = [120, 200, 240, 280]
    decay_rate = 1000 ** (-epoch / args.epochs)
    lr = args.lr * (decay_rate)
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
                        help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data',
                        help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
                        help='the architecture for the network: nin')
    parser.add_argument('--pretrained', action='store', default=None,
                        help='the path to the pretrained model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 10e-4)')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--VGG', action='store', default='VGG11',
                        help='the architecture for the network: VGG')
    parser.add_argument('--Device', type=int, default=0,
                        help='Device for cuda')
    parser.add_argument('--act', action='store', default='active')
    parser.add_argument('--finetune_weight', type=bool, default=False)
    parser.add_argument('--initialnum', action='store', default=None)

    args = parser.parse_args()
    print('==> Options:', args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    torch.cuda.set_device(args.Device)
    # prepare the data
    if not os.path.isfile(args.data + '/train_data'):
        raise Exception('Please assign the correct data path with --data <DATA_PATH>')

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

    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)


    initialstr = '00000000' if not args.initialnum else ''.join(
        ['1' if str(idx) in args.initialnum else '0' for idx in range(8)])
    print('====> initial', initialstr)

    # define the model

    model = NIN(binact=initialstr)

    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        # best_acc = 0.81*100
        print(best_acc)
        new_state_dict = {}
        for k, v in pretrained_model['state_dict'].items():
            if 'filtermask' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    model.cuda()

    count = 0


    for para in model.named_parameters():
        if 'mask' in para[0]:
            if args.finetune_weight:
                para[1].requires_grad = False
            else:
                if count == int(args.initialnum[-1]):
                    para[1].requires_grad = True
                else:
                    para[1].requires_grad = False
                count = count + 1
        if 'mask' not in para[0]:
            if args.finetune_weight:
                para[1].requires_grad = True
            else:
                para[1].requires_grad = False

    # if args.initialnum:
    #     for name,para in model.named_parameters():
    #         para.requires_grad = False
    #     for module in model.modules():
    #         if isinstance(module, MaskBinActiveConv2d) and module.binact:
    #             module.mask.requires_grad = True

    for name, para in model.named_parameters():
        print(name, para.requires_grad)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model, True if args.finetune_weight else False)

    print(len(bin_op.target_modules))

    # start training
    global writer, name_rec
    name_rec = 'Resnet_Bin_initial_weight' if not args.initialnum else 'Resnet_Bin_fintune_{:}_{:}_lr_{:}'.format(
        'weight' if args.finetune_weight else 'mask', args.initialnum, args.lr)
    print(name_rec)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('runs', current_time + '_' + socket.gethostname())
    writer = SummaryWriter(logdir + name_rec)

    test(0)
    for epoch in range(1, args.epochs + 1):
        tlr = adjust_learning_rate(optimizer, epoch)
        writer.add_scalar('learning_rate', tlr, epoch)
        train(epoch, model)

        if args.initialnum :
            prun_num = 0
            for m in model.modules():
                if isinstance(m, MaskBinActiveConv2d) and m.binact==True:
                    prun_num += m.filtermask.eq(-1).sum().item()
            writer.add_scalar('prunum', prun_num, epoch)

        test(epoch, model)