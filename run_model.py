# -*- coding: utf-8 -*-

from __future__ import print_function, division
 
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import time
import os
import scipy.io
import yaml
import math
from tqdm import tqdm
from model import ft_net_dense
version =  torch.__version__
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.add_argument('--no-gpu', dest='gpu', action='store_false')
parser.set_defaults(gpu=True)
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_net_dense', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--quantization', default='None', choices=['None', 'fp16', 'int8'], help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
print(opt, config)
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

if 'linear_num' in config:
    opt.linear_num = config['linear_num']

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir


quantization = opt.quantization

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.    
h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir


image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                            shuffle=False, num_workers=16) for x in ['gallery','query']}

image_datasets_train = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['train_all','query']}
dataloaders_train = {x: torch.utils.data.DataLoader(image_datasets_train[x], batch_size=16,
                                            shuffle=False, num_workers=16) for x in ['train_all','query']}


class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available() and opt.gpu

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    try:
        network.load_state_dict(torch.load(save_path))
    except: 
        if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
            print("Compiling model...")
            # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
            torch.set_float32_matmul_precisioquantizationn('high')
            network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
        network.load_state_dict(torch.load(save_path))

    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders,use_gpu, end_early=None):
    #features = torch.FloatTensor()
    count = 0
    pbar = tqdm()

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n

        # For quantization calibration routine
        if end_early is not None and end_early<count:
            break

        pbar.update(n)
        if use_gpu:
            ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()
        else:
            ff = torch.FloatTensor(n,opt.linear_num).zero_()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            if use_gpu:
                input_img = Variable(img.cuda())
            else:
                input_img = Variable(img)
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
        start = iter*opt.batchsize
        end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        features[ start:end, :] = ff
    pbar.close()
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)


######################################################################
# Load Collected data Trained model
#---------------------------

model_structure = ft_net_dense(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.classifier.classifier = nn.Sequential()



####################################
# For Quantization Calibration     #
####################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')





####################################################################
#                      INT8 QUANTIZATION                           #
####################################################################
if quantization == 'int8':
    use_gpu = False
    num_calibration_batches = 1
    # myModel = load_model(saved_model_dir + float_model_file).to('cpu')

    myModel = model.to('cpu')

    print("Size of model before quantization")
    print_size_of_model(model)
    myModel.eval()

    # Fuse Conv, bn and relu
    # myModel = fuse_all_conv_bn(myModel)
    myModel.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    myModel.qconfig = torch.ao.quantization.qconfig.default_qconfig
    torch.ao.quantization.prepare(myModel, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')

    criterion = nn.CrossEntropyLoss()
    # # Calibrate with the training set
    print('Calibrating.....')
    evaluate(myModel, criterion, dataloaders_train['train_all'], neval_batches=num_calibration_batches)

    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.ao.quantization.convert(myModel, inplace=True)
    # You may see a user warning about needing to calibrate the model. This warning can be safely ignored.
    # This warning occurs because not all modules are run in each model runs, so some
    # modules may not be calibrated.
    print('Post Training Quantization: Convert done')
    print("Size of model after quantization")
    print_size_of_model(myModel)

####################################################################
#                      F16 QUANTIZATION                            #
####################################################################
elif quantization == 'fp16':
    use_gpu = False
    print("Size of model before quantization")
    print_size_of_model(model)

    model.eval()
    model.fuse_model()

    myModel = torch.ao.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.float16)  # the target dtype for quantized weights

    print('Post Training Quantization: Convert done')
    print("Size of model after quantization")
    print_size_of_model(myModel)

####################################################################
#                      NO QUANTIZATION GPU                         #
####################################################################
elif quantization == 'None':
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    myModel = model


# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(myModel,dataloaders['gallery'], use_gpu)
    query_feature = extract_feature(myModel,dataloaders['query'], use_gpu)
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('output_mat/pytorch_result.mat',result)
scipy.io.savemat(f'output_mat/result_{quantization}.mat',result)
print(opt.name)
result = './model/%s/result.txt'%opt.name
os.system('python evaluate_gpu.py | tee -a %s'%result)