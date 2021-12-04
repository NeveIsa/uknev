import numpy as np
import datatable as dt

SHANGHAI = [31.15 , 121.35]

LAT_MIN, LAT_MAX = 31.0, 31.30
LON_MIN,LON_MAX = 121.20, 121.50

LAT_SLOTS,LON_SLOTS = 30,30

LAT_SLOT_SIZE = (LAT_MAX-LAT_MIN)/LAT_SLOTS
LON_SLOT_SIZE = (LON_MAX-LON_MIN)/LON_SLOTS

LAT_POINTS = np.arange(LAT_MIN,LAT_MAX, LAT_SLOT_SIZE)
LON_POINTS = np.arange(LON_MIN,LON_MAX,LON_SLOT_SIZE)

print('Reading csv... please wait')
print()
df = dt.fread('../sh_filtered_20x20.csv').to_pandas()

def getweekday( wk=1, dy=1 ):
    
    _df = df [ (df.weekofmonth==wk) & (df.dayofweekofmonth==dy) ]
    
    
    _weekdf_grouped = _df.groupby(['gridX','gridY','hour']).count()
    
    MAX_GRID_X = int(df.gridX.max())
    MAX_GRID_Y = int(df.gridY.max())
    
    _tensor = np.zeros((MAX_GRID_X+1, MAX_GRID_Y+1, 24))
    
    
    gx,gy,hours = zip(*_weekdf_grouped.index)
    
    gx,gy,hours = list(map(int,gx)),list(map(int,gy)),list(map(int,hours))
    
    _tensor[gx,gy,hours] = _weekdf_grouped.vid
    
    _matrix = np.zeros(( (MAX_GRID_X+1)*(MAX_GRID_Y+1) , 24))
    
    #print(_tensor.shape)
    #print(_matrix.shape)
    
    for _h in range(24):
        _matrix[:,_h] = _tensor[:,:,_h].flatten()
        
    return _tensor,_matrix

print('Generating week1')
WEEK1 = list(   map(  lambda d: getweekday(1,d), range(1,8)  )    )


print('Generating week2')
WEEK2 = list(   map(  lambda d: getweekday(2,d), range(1,8)  )    )

print('Generating week3')
WEEK3 = list(   map(  lambda d: getweekday(3,d), range(1,8)  )    )

del(df)
