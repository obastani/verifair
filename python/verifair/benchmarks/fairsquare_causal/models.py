from .helper import *

from . import M_BNc_F_DT_V3_D2_N44_Q
from . import M_BNc_F_NN_V3_H2_Q
from . import M_BNc_F_SVM_V6_Q

_DISTS = {
    'dist2': 'BNc',
}

_MODELS = {
    'dt4': 'DT_V3_D2_N44',
    'nn2': 'NN_V3_H2',
    'svm4': 'SVM_V6',
}

def all_dists():
    return _DISTS.keys()

def all_models():
    return _MODELS.keys()

def get_model_name(model, dist):
    name = ''
    name += 'M_'
    name += _DISTS[dist]
    name += '_F_'
    name += _MODELS[model]
    name += '_Q'
    return name

def get_model(model, dist):
    name = get_model_name(model, dist)
    ldict = locals()
    exec('fn = {}.sample'.format(name))
    return ldict['fn']
