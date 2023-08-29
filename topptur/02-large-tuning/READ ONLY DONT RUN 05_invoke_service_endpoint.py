# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Invoking a model from a service endpoint

# COMMAND ----------

import os

os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(scope="llmtraining", key="model_api_token")

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://dbc-11ce6ca4-7321.cloud.databricks.com/serving-endpoints/t5-large-summary4/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()
  
 
data = pd.DataFrame([{
  "prompt": "I actually did not receive it, was offered a refund and we parted ways. Was sent a pair of speakers, that only work overseas because of the power plug. was told to dispose of them any way I desireed. I finally broke away from my 35mm Nikkon and decided to try a digital one...one draw back, the LCD screen is impossible to view in bright light. I had to move to shade or snap and hope they came out. I was able to view in the index at the pics to see if they turned out before I moved on which helped. There is a ev+1.5 to ev-1.5 and a brightness control along with choice of event setting. We bought our camera this past spring to use at a wedding.  It was awesome.",
}])
response = score_model(data)
response


# COMMAND ----------


