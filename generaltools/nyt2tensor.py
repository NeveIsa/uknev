import datatable as dt
import datetime
import numpy as np
import seaborn as sns
import pandas as pd


def load(foldername):
    yellow = dt.fread(f"{foldername}/yellow_tripdata_2019-12.csv").to_pandas()[["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]]
    green = dt.fread(f"{foldername}/green_tripdata_2019-12.csv").to_pandas()[["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID"]]
    forhire = dt.fread(f"{foldername}/fhv_tripdata_2019-12.csv").to_pandas()[["pickup_datetime","dropoff_datetime","PULocationID","DOLocationID"]]
    forhirehighvol = dt.fread(f"{foldername}/fhvhv_tripdata_2019-12.csv").to_pandas()[["pickup_datetime","dropoff_datetime","PULocationID","DOLocationID"]]


    yellow.rename(columns={"tpep_pickup_datetime": "put", "tpep_dropoff_datetime": "dot", "PULocationID":"pul", "DOLocationID":"dol"}, inplace=True)
    green.rename(columns={"lpep_pickup_datetime": "put", "lpep_dropoff_datetime": "dot", "PULocationID":"pul", "DOLocationID":"dol"}, inplace=True)
    forhire.rename(columns={"pickup_datetime":"put","dropoff_datetime":"dot", "PULocationID":"pul","DOLocationID":"dol"}, inplace=True)
    forhirehighvol.rename(columns={"pickup_datetime":"put","dropoff_datetime":"dot", "PULocationID":"pul","DOLocationID":"dol"}, inplace=True)

    
    yellow.columns, green.columns, forhire.columns, forhirehighvol.columns

    data = pd.DataFrame()
    data = data.append(yellow)
    data = data.append(green)
    data = data.append(forhire)
    data = data.append(forhirehighvol)
    
    del yellow
    del green
    del forhire
    del forhirehighvol

    data["puth"] = data["put"].dt.hour
    data["doth"] = data["dot"].dt.hour # data.dot is a function, hence need to data["dot"]

    MAX_LOC_ID = max(data.pul.max(),data.dol.max())
    MAX_LOC_ID

    grouped = data.groupby(["pul","dol","doth"]).count()
    grouped.head()


    print("MAX_LOC_ID:",MAX_LOC_ID)
    
    pu=sorted(data.pul.unique())
    do=sorted(data.dol.unique())
    
    for i in range(1,MAX_LOC_ID+1):
        if not i in pu: print("pu",i)
        if not i in do: print("do",i)
    
    TRIP_TENSOR = np.zeros((MAX_LOC_ID, MAX_LOC_ID, 24))
    
    
    for i in grouped.index:
        # MAX_LOC_ID - 1 since it starts from 1
        TRIP_TENSOR[i[0]-1,i[1]-1,i[2]] = grouped.loc[i].put


    return TRIP_TENSOR
    
    # summed = TRIP_TENSOR.sum(axis=2)
    # X,Y,Z = [],[],[]
    
    # count=0
    # for x in range(MAX_LOC_ID):
    #     for y in range(MAX_LOC_ID):
    #         if np.random.rand() > 0:
    #             X.append(x)
    #             Y.append(y)
    #             z = summed[x,y]
    #             if z>200000: z = 20000
    #             Z.append(z)
    
                
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_trisurf(X,Y,Z, color="red", alpha=0.5)

        
