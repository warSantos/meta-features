import os
import numpy as np
from collections import Counter
from itertools import product

from sklearn.model_selection import StratifiedKFold

# Local libraries

from src.dist_mfs.mf.centbased import MFCent
from src.dist_mfs.mf.knnbased import MFKnn
from src.dist_mfs.mf.bestk import kvalues
from src.dist_mfs.inout.general import get_data

from src.stat_mfs.stat import StatisticalMFs
from src.stat_mfs.info import InfoMFS
from src.dist_mfs.dist import DistMFs
from src.encoder_mfs.encoder import EncoderMFs
    
if __name__ == "__main__":

    #mf = StatisticalMFs()
    #mf.build()
    #mf = InfoMFS()
    #mf.build()
    mf = EncoderMFs()
    mf.build()
    #mf = DistMFs()
    #mf.build()