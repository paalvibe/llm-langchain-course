# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Test the query endpoint

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7777"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(f"""
driver_proxy_api = '{driver_proxy_api}'
cluster_id = '{ctx.clusterId}'
port = {port}
""")

# COMMAND ----------

import requests
import json

def request_dolly_v2_7b(prompt, driver_proxy_api, temperature=1.0, max_new_tokens=1024):
  token = dbutils.secrets.get(scope="llmtraining", key="model_api_token")
  url = driver_proxy_api
  
  headers = {
      "Content-Type": "application/json",
      "Authentication": f"Bearer {token}"
  }
  data = {
    "prompt": prompt,
    "temperature": temperature,
    "max_new_tokens": max_new_tokens,
  }

  response = requests.post(url, headers=headers, data=json.dumps(data))
  return response.text


request_dolly_v2_7b("What is databricks?", driver_proxy_api=driver_proxy_api)

# COMMAND ----------


