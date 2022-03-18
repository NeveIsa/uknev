from tqdm import tqdm
import tensorly as tl
import numpy as np

def getblock(_img,  block_id=None, block_size=(50,50,5), stride=(10,10,5)):
    h,w,d = _img.shape
    
    stride_h, stride_w, stride_d = stride
    
    
    block_size_h = block_size[0]
    block_size_w = block_size[1]
    block_size_d = block_size[2]
    
    
    # hmax = h - 1 - block_size_h
    # wmax = w - 1 - block_size_w
    
    blocks_h = list(range(0,h,stride_h))
    blocks_w = list(range(0,w,stride_w))
    blocks_d = list(range(0,d,stride_d))
    
    
    if block_id == None:
        return len(blocks_h) * len(blocks_w) * len(blocks_d)#, len(blocks_h) , len(blocks_w)
    
    else:
        block_index_Y = block_id % len(blocks_h)
        block_index_X = block_id % len(blocks_w)
        block_index_Z = block_id % len(blocks_d)
        
        #print(block_index_Y,block_index_X)
        
        block_h_start = block_index_Y * stride_h
        block_h_end = min(block_h_start + block_size_h, h)
        
        block_w_start = block_index_X * stride_w
        block_w_end = min(block_w_start + block_size_w, w)
        
        block_d_start = block_index_Z * stride_d
        block_d_end = min(block_d_start + block_size_d, d)
        
        #print(block_h_start,block_h_end, block_w_start,block_w_end)
        block = _img[block_h_start:block_h_end, block_w_start:block_w_end, block_d_start:block_d_end]

        return block
    
def setblock(_img, block2set, block_id=None, block_size=(50,50,5), stride=(10,10,5)):
    h,w,d = _img.shape
    
    stride_h, stride_w, stride_d = stride
    
    #print(block_size, block2set.shape)
    
    #assert block2set.shape == block_size
    
    block_size_h = block_size[0]
    block_size_w = block_size[1]
    block_size_d = block_size[2]
    
    # hmax = h - 1 - block_size_h
    # wmax = w - 1 - block_size_w
    
    blocks_h = list(range(0,h,stride_h))
    blocks_w = list(range(0,w,stride_w))
    blocks_d = list(range(0,d,stride_d))
    
    if block_id == None:
        return len(blocks_h) * len(blocks_w) * len(blocks_d)#, len(blocks_h) , len(blocks_w)
    
    else:
        block_index_Y = block_id % len(blocks_h)
        block_index_X = block_id % len(blocks_w)
        block_index_Z = block_id % len(blocks_d)
        
        #print(block_index_Y,block_index_X)
        
        block_h_start = block_index_Y * stride_h
        block_h_end = min(block_h_start + block_size_h, h)
        
        block_w_start = block_index_X * stride_w
        block_w_end = min(block_w_start + block_size_w, w)
        
        block_d_start = block_index_Z * stride_d
        block_d_end = min(block_d_start + block_size_d, d)
        
        #print(block_h_start,block_h_end, block_w_start,block_w_end)
        _img[block_h_start:block_h_end, block_w_start:block_w_end, block_d_start:block_d_end] = block2set

        
def factors2tensor(a,b,c):
    
    rank_a = a.shape[1]
    rank_b = b.shape[1]
    rank_c = c.shape[1]
    
    rank = rank_a
    
    assert rank_a == rank_b == rank_c
    
    context = tl.context(a)
    w = tl.tensor(np.ones(rank), **context)
    t = tl.cp_tensor.cp_to_tensor( (w ,(a,b,c)) )
        
    return t
        
def blockimpute(tensor, mask, decomposer, block_rank=1, block_shape=(20,20,5), stride=(10,10,3),progressbar=True):
    """
        Signature of decomposer callback --->
                (w,f),e = imputer(tensor, mask, rank) 
    """
    assert tensor.shape == mask.shape
    
    tensor_copy = tl.copy(tensor)
    
    blockIdMax = getblock(tensor)
    
    if progressbar:
        blockIds = tqdm(range(blockIdMax), leave=False, desc=f"shape={block_shape} | stride={stride} | rank={block_rank}",ncols=100)
    else:
        blockIds = range(blockIdMax)
        
    for blkid in blockIds:
        
        tblock = getblock(tensor,block_id = blkid, block_size = block_shape, stride=stride)
        mblock = getblock(mask,block_id=blkid, block_size = block_shape, stride=stride)
        
        assert tblock.shape == mblock.shape
        
        (w,f),e = decomposer(tblock, mask = mblock, rank = block_rank)
        a,b,c = f
        reconstructed_block = factors2tensor(a,b,c)
     
        
        setblock(tensor_copy, block2set=reconstructed_block, block_id=blkid, block_size=block_shape, stride=stride)
        
    error = tl.norm(  mask*(tensor - tensor_copy)  ) / tl.norm(mask * tensor)
    
    return error