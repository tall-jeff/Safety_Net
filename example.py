###########################################################################
# Initial data wrangling + model building to explain the concept to XXXXXXX
###########################################################################

from contextlib import contextmanager
import sys, os
import numpy as np
import pandas as pd
from pandas import read_csv
from datetime import datetime, timedelta
import pmdarima as pm
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preporcessing as ppc
from pmdarima import arima

temp_input_file = "input.csv"
temp_output_file = "output.csv"

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

pd.set_option('precision', 0)

Final_Results=pd.DataFrame()

i=0

df=pd.DataFrame(read_csv(input_file))

df["Date"] = pd.to_datetime(df["trns_dttm"]) - pd.to_datetime(df["trns_dttm"]).dt.weekday * timedelta(days=1)
df["Product_Decision"] = df["co_nm_txt"] + "|" + df["rle_srvc_cd"] + "_" + df["prof_cd"] + "_" + df["decsn_cd"]

df = df.drop(["co_nm_txt", "rle_srvc_cd", "prod_cd", "decsn_cd", "trns_dttm"], axis=1)

Master_Data = pd.pivot_table(df, values="cnt", index=["Date"],columns=["Product_Decision"], aggfunc=np.sum, fill_value=0, margins=False)

Num_Columns = (len(Master_Data.columns))

for i in range (0, Num_Columns):
  actual_values = Master_Data.iloc[:, i].values
  
  train = actual_values[:int((len(actual_values)-4))]
  recent = actual_values[int((len(avtual_values)-4)):]
  
  with supress_stdout():
    pipe = pipeline.Pipeline([
          ("fourirer", ppc.FourierFeaturizer(m=7, k=3)),
          ("arima", arima.AutoArima(stepwise=True, trace=1, error_action="ignore", seasonal=False, suppress_warnings=True))
    ])
    pipe.fit(train)
    
    predicted, out_sample_confidence_interval = pipe.predict(n_periods=4, return_conf_int=True, alpha=.0001)
    
    in_sample_predicted, in_sample_confidence interval = pipe.predict_in_sample(exogenous=None, return_conf_int = True)
    
    Complete_Set = pd.DataFrame()
    
    in_sample=pd.DataFrame(in_sample_confidence_interval)
    in_sample["Forecast"] = "NotForecast"
    outsample = pd.DataFrame(out_sample_confidence_interval)
    out_sample["Forecast"] = "Forecast"
    Full_Confint = pd.concat([in_sample, out_sample], ignore_index=True)
    
    Full_Confint["Product_Decision"] = Master_Data.columns[i]
    
    values = pd.DataFrame(actual_values)
    Complete_Set=pd.concat([Full_Confint.iloc[:, 3], Full_Confint.iloc[:, 2], Full_Confint.iloc[:, 0], values, Full_Confint.iloc[:, 1]], axis=1)
    Complete_Set.columns = ["Product_Decision", "Forecast", "Lower_CI", "CaseCount", "Upper_CI"]
    Complete_Set = Complete_Set.set_index(Master_Data.index)
    Complete_Set.reset_index(level=0, inplace=True)
    
    Complete_Set["Lower_CI"].values[Complete_Set["Lower_CI"] < 0] = 0.4
    Complete_Set["Upper_CI"].values[Complete_Set["Upper_CI"] < 0] = 0.4
    Complete_Set=Complete_Set.round()
    
    Complete_Set.drop(Complete_Set.head(1).index, inplace=True)
    
    Complete_Set = Complete_Set[["Product_Decision", "Forecast", "Date", "Lower_CI", "CaseCount", "Upper_CI"]]
    Final_Results = pd.concat([Final_Results, Complete_Set])
    
    i=i+1

  Final_Results[["co_nm_txt", "PD"]] = Final_Results.Product_Decision.str.split("|", expand = True)
  Final_Results = Final_Results.drop(["PD"], axis = 1)
  pd.DataFrame(Final_Results).to_csv(output_file, index = False)
