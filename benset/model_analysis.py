import os
import sys
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())
from keras.utils import plot_model
import deephar

from deephar.config import mpii_dataconf
from deephar.config import human36m_dataconf
from deephar.config import ntu_dataconf
from deephar.config import ntu_pe_dataconf
from deephar.config import ModelConfig
from deephar.config import DataConfig

from deephar.models import modify_model

from deephar.data import MpiiSinglePerson
from deephar.data import Human36M
from deephar.data import PennAction
from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.losses import pose_regression_loss
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import keras.backend as K

from deephar.callbacks import SaveModel

from deephar.trainer import MultiModelTrainer
from deephar.models import compile_split_models
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
#from datasetpath import datasetpath

from mpii_tools import MpiiEvalCallback
from h36m_tools import H36MEvalCallback
from ntu_tools import NtuEvalCallback

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 6
cfg = ModelConfig((num_frames,) + ntu_dataconf.input_shape, pa17j3d,
        num_actions=[5], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=192, num_visual_features=192)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)


"""Build the full model"""
full_model = spnet.build(cfg)
full_model.summary()
plot_model(full_model, to_file='output\model.png')

#num_actions = 5
#modified_model = modify_model(full_model, cfg, num_actions)

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(
         "E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0429\\weights_3dp+ntu_ar_048.hdf5",
         by_name=True)

