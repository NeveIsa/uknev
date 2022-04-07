import sys
sys.path.append("../")

import numpy as np
import tensorly as tl
import random

from unknownev import blockimputer as bi

import pytest

def test_getblock():
    shape = (100,90,70)

    # # # # #
    # # # # #
    # # # # #
    # # # # #
    # # # # #
    # # # # #
    
    blockShape = (random.randint(1,100), random.randint(1,80), random.randint(1,70))
    strideShape = (random.randint(1,20), random.randint(1,20), random.randint(1,20))

    
    tensor = tl.tensor(np.random.random(shape))
    totalblocks = bi.get_block_indices(tensor, block_size=blockShape, stride=strideShape)
    print(totalblocks)
    blockranges = []
    for bid in range(totalblocks):
        r = bi.get_block_indices(tensor, block_size=blockShape, stride=strideShape, block_id=bid)
        blockranges.append(str(r))

    assert len(set(blockranges))==totalblocks
