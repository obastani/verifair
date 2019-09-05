from .helper import *

from . import M_BN_F_DT_A_Q
from . import M_BN_F_DT_V2_D2_N16_Q
from . import M_BN_F_DT_V2_D2_N4_Q
from . import M_BN_F_DT_V2_D3_N14_Q
from . import M_BN_F_DT_V3_D2_N44_Q
from . import M_BN_F_NN_V2_H1_Q
from . import M_BN_F_NN_V2_H2_Q
from . import M_BN_F_NN_V3_H2_Q
from . import M_BN_F_SVM_A_Q
from . import M_BN_F_SVM_V3_Q
from . import M_BN_F_SVM_V4_Q
from . import M_BN_F_SVM_V5_Q
from . import M_BN_F_SVM_V6_Q

from . import M_BNc_F_DT_A_Q
from . import M_BNc_F_DT_V2_D2_N16_Q
from . import M_BNc_F_DT_V2_D2_N4_Q
from . import M_BNc_F_DT_V2_D3_N14_Q
from . import M_BNc_F_DT_V3_D2_N44_Q
from . import M_BNc_F_NN_V2_H1_Q
from . import M_BNc_F_NN_V2_H2_Q
from . import M_BNc_F_NN_V3_H2_Q
from . import M_BNc_F_SVM_A_Q
from . import M_BNc_F_SVM_V3_Q
from . import M_BNc_F_SVM_V4_Q
from . import M_BNc_F_SVM_V5_Q
from . import M_BNc_F_SVM_V6_Q

from . import M_ind_F_DT_A_Q
from . import M_ind_F_DT_V2_D2_N16_Q
from . import M_ind_F_DT_V2_D2_N4_Q
from . import M_ind_F_DT_V2_D3_N14_Q
from . import M_ind_F_DT_V3_D2_N44_Q
from . import M_ind_F_NN_V2_H1_Q
from . import M_ind_F_NN_V2_H2_Q
from . import M_ind_F_NN_V3_H2_Q
from . import M_ind_F_SVM_A_Q
from . import M_ind_F_SVM_V3_Q
from . import M_ind_F_SVM_V4_Q
from . import M_ind_F_SVM_V5_Q
from . import M_ind_F_SVM_V6_Q

_DISTS = {
    'dist0': 'ind',
    'dist1': 'BN',
    'dist2': 'BNc',
}

_MODELS = {
    'dt0': 'DT_A',
    'dt1': 'DT_V2_D2_N16',
    'dt2': 'DT_V2_D2_N4',
    'dt3': 'DT_V2_D3_N14',
    'dt4': 'DT_V3_D2_N44',
    'nn0': 'NN_V2_H1',
    'nn1': 'NN_V2_H2',
    'nn2': 'NN_V3_H2',
    'svm0': 'SVM_A',
    'svm1': 'SVM_V3',
    'svm2': 'SVM_V4',
    'svm3': 'SVM_V5',
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
