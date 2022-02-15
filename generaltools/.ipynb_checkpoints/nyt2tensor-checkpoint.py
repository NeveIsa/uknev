import datatable as dt
import datetime
import numpy as np
import seaborn as sns
import pandas as pd


def load():
    yellow = dt.fread("data/yellow_tripdata_2019-12.csv").to_pandas()[["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]]
    green = dt.fread("data/green_tripdata_2019-12.csv").to_pandas()[["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID"]]
    forhire = dt.fread("data/fhv_tripdata_2019-12.csv").to_pandas()[["pickup_datetime","dropoff_datetime","PULocationID","DOLocationID"]]
    forhirehighvol = dt.fread("data/fhvhv_tripdata_2019-12.csv").to_pandas()[["pickup_datetime","dropoff_datetime","PULocationID","DOLocationID"]]


    yellow.rename(columns={"tpep_pickup_datetime": "put", "tpep_dropoff_datetime": "dot", "PULocationID":"pul", "DOLocationID":"dol"}, inplace=True)
    green.rename(columns={"lpep_pickup_datetime": "put", "lpep_dropoff_datetime": "dot", "PULocationID":"pul", "DOLocationID":"dol"}, inplace=True)
    forhire.rename(columns={"pickup_datetime":"put","dropoff_datetime":"dot", "PULocationID":"pul","DOLocationID":"dol"}, inplace=True)
    forhirehighvol.rename(columns={"pickup_datetime":"put","dropoff_datetime":"dot", "PULocationID":"pul","DOLocationID":"dol"}, inplace=True)
