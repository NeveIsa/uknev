# uknev


This library implements imputation of incomplete 3-way tensors using CP decomposition.

Below we show how to impute a `tensor` of shape (m,n,p) with a mask `missing_mask`.
`missing_mask` also of shape (m,n,p) and has entries 0 when the data is missing, 1 when available.

`a,b,c` are the factor matrices of shape mx7 nx7 and px7 respectively. 

`re` gives the reconstruction error and `ie` gives the imputation error. 
Both the errors are relative/normalized mean squared errors with respect to the original tensor and the missing entries respectively. 

```
from unknownev.imputer import impute
a,b,c,re, ie = impute(tensor, rank=7, tol=1e-10,  n_iters=10000, mask=missing_mask)

```



