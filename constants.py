import operator
import os
import uuid
from datetime import datetime

import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UNIQUE_ID = str(uuid.uuid4()).split('-')[0]
PROJECT_NAME = "activation_hmse"
FILE_TIME_STAMP = datetime.now().strftime("%d-%b-%Y__%H:%M:%S")
FOLDERS_TO_EXCLUDE_COPYING = ['results_tmp', 'model_zoo', '__pycache__', '.git', '.idea',
                              'ultralytics']
IMAGE_INPUT = 'image_input'
CALL_MODULE = 'call_module'
CALL_METHOD = 'call_method'
CALL_FUNCTION = 'call_function'
OUTPUT = 'output'
WEIGHTS_N_BITS = 'weights_n_bits'
ACTIVATIONs_N_BITS = 'activations_n_bits'
INCLUDE_IN_MP = 'include_in_mp'
ACTIVATION_QUANT_STR = 'activation_quant'

GPU = 'GPU'
GPUALG3090 = "GPUALG3090"
GPUALG = "GPUALG"
GPUALG2 = "GPUALG2"
GPU6000 = "GPUALG6000"
GPUALGALL = "GPUALGALL"

# Local copies:
if os.path.isdir('/local_datasets/ImageNet/ILSVRC2012_img_val_TFrecords'):
    VAL_DIR = '/local_datasets/ImageNet/ILSVRC2012_img_val_TFrecords'
    TRAIN_DIR = '/local_datasets/ImageNet/ILSVRC2012_img_train'
else:
    # Netapp copies:
    VAL_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords'
    TRAIN_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train'

COCO_TRAIN_DIR = '/Vols/vol_design/tools/swat/datasets_src/COCO/images/train2017'
COCO_VAL_DIR = '/Vols/vol_design/tools/swat/datasets_src/COCO/images/val2017'
COCO_DIR = '/Vols/vol_design/tools/swat/datasets_src/COCO'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TORCHVISION = 'torchvision'
TIMM = 'timm'
MODELS_LIBS = [TORCHVISION, TIMM]

TIMM_IMAGENET_RESULTS = ('/Vols/vol_design/tools/swat/users/ariell/repos/sony_fork/git/pytorch-image-models/'
                         'results/results-imagenet.csv')

RESNET18 = 'resnet18'
RESNET50 = 'resnet50'
MBV2 = 'mobilenetv2'
VIT = 'vit'
DEIT = 'deit'
SWIN = 'swin'
VIT_SMALL = 'vit_s'
VIT_BASE = 'vit_b'
DEIT_TINY = 'deit_t'
DEIT_SMALL = 'deit_s'
DEIT_BASE = 'deit_b'
SWIN_TINY = 'swin_t'
SWIN_SMALL = 'swin_s'
SWIN_BASE = 'swin_b'

SOLVER_TIME_LIMIT = 120  # 2 minutes
FLOAT_BIT_WIDTH = 32
MP_BIT_WIDTH = [2, 3, 4, 6, 8]
LINEAR_OPS = [(torch.nn.Conv1d,),
              (torch.nn.Conv2d,),
              (torch.nn.Conv3d,),
              (torch.nn.Linear,)]

ACTIVATION_OPS = [(torch.nn.ReLU,),
                  (torch.nn.ReLU6,),
                  (torch.nn.Identity,)]

LINEAR_QUANTIZE_OPERATORS = [torch.nn.Linear, torch.matmul, operator.matmul]
REDUNDANT_OPERATORS = [torch.nn.Identity, torch.nn.Dropout]

SIGMOID_MINUS = 4

PARAM_SEARCH_ITERS = 15
PARAM_SEARCH_STEPS = 100

ORIGINAL_W = 'original_w'
SVD_W_SCORES = 'svd_w_scores'
LAYER_COMPRESSION_CONFIG = 'layer_compression_config'
SIZE = 'size'
MSE = 'mse'