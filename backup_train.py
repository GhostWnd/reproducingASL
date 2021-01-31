# -*- coding: utf-8 -*-

import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os
from torch.optim import lr_scheduler

from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset', default = '/home/s2118392/cw3/MSCOCO/')
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')

idx_to_class = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
                11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
                57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
                63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 
                72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
                78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
                85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
def main():
    args = parser.parse_args()
    args.batch_size = args.batch_size

    # setup model
    print('creating model...')
    #state = torch.load(args.model_path, map_location='cpu')
    #args.num_classes = state['num_classes']
    args.do_bottleneck_head = True
    model = create_model(args).cuda()
    #model.load_state_dict(state['model'], strict=True)
    model.train()
    classes_list = np.array(list(idx_to_class.values()))
    print('done\n')

    # Data loading code
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])

    instances_path_val = os.path.join(args.data, 'annotations/instances_val2017.json')
    #instances_path_train = os.path.join(args.data, 'annotations/instances_val2017.json')#temprarily use val as train
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2017.json')

    data_path_val = os.path.join(args.data, 'val2017')
    #data_path_train = os.path.join(args.data, 'val2017')#temporarily use val as train
    data_path_train = os.path.join(args.data, 'train2017')

    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
    train_dataset = CocoDetection(data_path_train,
                                instances_path_train,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    
    criterion = AsymmetricLoss()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=0.0001)#尝试新的optimizer
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    total_step = len(train_loader)
    
    highest_mAP = 0
    trainInfoList = []
    Sig = torch.nn.Sigmoid()

    for epoch in range(5):
        for i, (inputData, target) in enumerate(train_loader):
            #model.train()
            inputData = inputData.cuda()
            target = target.cuda()
            target = target.max(dim=1)[0]
            #Sig = torch.nn.Sigmoid()
            output = Sig(model(inputData))
            #output[output<args.thre] = 0
            #output[output>=args.thre]=1
            #print(output.shape) #(batchsize, channel, imhsize, imgsize)
            #print(inputData.shape) #(batchsize, numclasses)
            #print(output[0])
            #print(target[0])
            

            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            #store information
            if i % 10 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, 5, i, total_step, loss.item()))
                
            if (i+1) % 400 == 0:
                #储存相应迭代模型
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-{}-{}.ckpt'.format(epoch+1, i+1)))
                #modelName = 'models/' + 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)
                mAP_score = validate_multi(val_loader,  model, args)
                model.train()
                if mAP_score > highest_mAP:
                    highest_mAP = mAP_score
                    print('current highest_mAP = ', highest_mAP)
                    torch.save(model.state_dict(), os.path.join(
                            'models/', 'model-highest.ckpt'))
    #scheduler.step()#修改学习率   
    
def validate_multi(val_loader, model, args):
    print("starting actuall validation")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    mAP_meter = AverageMeter()

    Sig = torch.nn.Sigmoid()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(args.thre).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time,
                prec=prec, rec=rec))
            print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                    .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    print(
        '--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("mAP score:", mAP_score)

    return mAP_score
if __name__ == '__main__':
    main()
