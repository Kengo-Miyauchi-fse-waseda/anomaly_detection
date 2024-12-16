import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta

def arrange_data(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    sorted_df = df.sort_values(by="Timestamp")
    return sorted_df

def extract_specific_id(df,id):
    data = df[df['Turbine_ID'] == id]
    return data

def extract_specific_terms(df,start,end):
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    filtered_df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)]
    return filtered_df

def get_months(df):
    data_by_month = df.groupby(df["Timestamp"].dt.to_period("M")).size()
    return data_by_month

def new_time(time,n,type):
    if(type=="months"):
        new_time = time + relativedelta(months=n)
    if(type=="minutes"):
        span = 10
        new_time = time + timedelta(minutes=n * span)
    return new_time