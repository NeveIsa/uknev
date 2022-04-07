import sys
sys.path.append("../")

import numpy as np
import tensorly as tl
import random

from unknownev import blockimputer as bi

import pytest

def test_getblock():
    X,Y,Z = 10,10,10
    shape = (X,Y,Z)
    blockShape = (random.randint(1,X), random.randint(1,Y), random.randint(1,Z))
    strideShape = (random.randint(1,X//2), random.randint(1,Y//2), random.randint(1,Z//2))
    
    tensor = tl.tensor(np.random.random(shape))
    totalblocks = bi.get_block_indices(tensor, block_size=blockShape, stride=strideShape)
    print(totalblocks)
    blockranges = []
    for bid in range(totalblocks):
        r = bi.get_block_indices(tensor, block_size=blockShape, stride=strideShape, block_id=bid)
        blockranges.append(str(r))

    assert len(set(blockranges))==totalblocks
    assert len(set(blockranges))==len(blockranges)

    print(blockranges)
