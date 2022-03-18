import tensorly as tl
import numpy as np
from tensorly import tenalg
from tqdm import tqdm

def random_mask(tensor,missing_fraction):
    rmask = np.random.random(tensor.shape)
    rmask =  (rmask >= missing_fraction)*1
    
    print(f"Mask (available percent): {rmask.sum()*100//np.prod(rmask.shape)} %")
    
    return rmask
    
    
def impute_lowlevel(tensor, rank, mode=1,  mask=np.ones(1), A=np.zeros(1), B=np.zeros(1), C=np.zeros(1)):    
    if mask.all():
         mask = np.ones(tensor.shape)
    else:
         pass
        
    assert tensor.shape == mask.shape
        
    if A.sum()==0 and B.sum()==0 and C.sum()==0:
        maxim = tensor.max()  ### set the initial random factors to as big as half of the maximum observed entry
        dim0,dim1,dim2 = tensor.shape
        A=np.random.rand(dim0,rank)*maxim/2
        B=np.random.rand(dim1,rank)*maxim/2
        C=np.random.rand(dim2,rank)*maxim/2
    
    assert 0 <= mode <= 2
    
    if mode==1: pass
    elif mode==0:
        __ = tl.transpose(tensor)
        __ = tl.moveaxis(__,1,2)
        tensor = __
        A,B,C = C,A,B
        # (4, 3, 2) -> (2, 4, 3) 
        
        # do same for mask
        __m = tl.transpose(mask)
        __m = tl.moveaxis(__m, 1,2)
        mask = __m
        
    elif mode==2:
        # mode=2
        __ = tl.transpose(tensor)
        __ = tl.moveaxis(__,0,1)
        tensor = __
        A,B,C = B,C,A
        # (4, 3, 2) -> (3, 2, 4)
        
        # do same for mask
        __m = tl.transpose(mask)
        __m = tl.moveaxis(__m, 0,1)
        mask = __m
        
    
    
    # gen ACFLAT
    AC = np.einsum('ij,jk->jik',A,C.T)
    ACFLAT = np.zeros( (np.prod(AC.shape[1:]),AC.shape[0]) )
            
    for i in range(AC.shape[0]):
        ACFLAT[:,i] = AC[i].ravel("F")
        
    
    
    # gen tensorflat
    tensorflat = np.zeros( (np.prod([tensor.shape[0],tensor.shape[2]]), tensor.shape[1] ) )
    maskflat = np.zeros( (np.prod([mask.shape[0],mask.shape[2]]), mask.shape[1] ) )
    
    for i in range(tensor.shape[2]):
        tensorflat[i*tensor.shape[0]:(i+1)*tensor.shape[0],:] = tensor[:,:,i]
        maskflat[i*mask.shape[0]:(i+1)*mask.shape[0],:] = mask[:,:,i]
        

        
    #print(ACFLAT.shape, B.T.shape, tensorflat.shape)
    #print(B.shape)
    if mask.all():
        Btranspose,residuals,__rank_of_ACFLAT,singular_values = np.linalg.lstsq(ACFLAT, tensorflat, rcond=None)
        B = Btranspose.T
    else:
        for i in range(tensorflat.shape[1]):
            ACFLAT_masked = ACFLAT * np.outer(maskflat[:,i],np.ones(ACFLAT.shape[1]))
            tensorflat_masked = tensorflat[:,i]*maskflat[:,i]
            
            
            # generating the factors one component at a time
            B[i,:],residuals,__rank_of_ACFLAT,singular_values = np.linalg.lstsq(ACFLAT_masked, tensorflat_masked , rcond=None)    
    
    # Move axis back
    if mode==0:
        __ = tl.moveaxis(tensor,1,2)
        __ = tl.transpose(__)
        tensor = __
        # A,B,C = C,A,B
        A,B,C = B,C,A
        
        # same for mask
        __m = tl.moveaxis(mask,1,2)
        __m = tl.transpose(__m)
        mask = __m
        
    elif mode==2:
        __ = tl.moveaxis(tensor,1,0)
        __ = tl.transpose(__)
        tensor = __
        # A,B,C = B,C,A
        A,B,C = C,A,B
        
        # same for mask
        __m = tl.moveaxis(mask,1,0)
        __m = tl.transpose(__m)
        mask = __m
    
    #print(residuals)
    weights = np.repeat(1,rank)
    reconstructed = tl.cp_to_tensor(       (weights,(A,B,C))         )
    rec_err = tl.norm(tensor - reconstructed) / tl.norm(tensor)
    
    inv_mask = 1 - mask
    imp_err = tl.norm( ( tensor - reconstructed ) * inv_mask) / tl.norm( tensor * inv_mask)

    return A,B,C,rec_err, imp_err


def impute(tensor, rank=1, mask=np.ones(1), n_iters=100, tol=1e-7,verbose=True, A=np.zeros(1), B=np.zeros(1),C=np.zeros(1)):
    """
        inputs:  tensor, rank=1, mask=np.ones(1), n_iters=100, tol=1e-7, verbose=True
        outputs: a,b,c, recon_errors, impute_error 
    """
    recon_errors = []
    impute_errors = []
    
    # initialize factors
    if A.sum()==0 and B.sum()==0 and C.sum()==0:
        a,b,c,re,ie = impute_lowlevel(tensor,mode=0, rank=rank)
    else:
        a,b,c = A,B,C
    
    last_re = np.inf
    
    if verbose==True:
        tqdm_context = tqdm(range(n_iters), colour='#00FFD1', ncols=100)
    else: 
        tqdm_context = range(n_iters)
        
    break_count = 0
    
    for xyz in tqdm_context:
        a,b,c,re,ie = impute_lowlevel(tensor,A=a,B=b,C=c,mode=xyz%3, rank=rank, mask=mask)
        recon_errors.append(re)
        impute_errors.append(ie)
        
        
        if verbose==True: tqdm_context.set_postfix(re="{:.7f}".format(re), ie="{:.7f}".format(ie))
        
        if last_re - re > tol:
            pass
        else:
            break_count+=1
            if break_count > 10: 
                break
            
        last_re = re
        
    return a,b,c, np.array(recon_errors) , np.array(impute_errors)

