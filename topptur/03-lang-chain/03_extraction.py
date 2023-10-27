# Databricks notebook source
# MAGIC %md
# MAGIC # Extraction with langchain on Databricks
# MAGIC
# MAGIC We use a mistral model served from another cluster which has GPU.
# MAGIC
# MAGIC Can be run on a non-gpu cluster.
# MAGIC
# MAGIC ## What is Langchain?
# MAGIC
# MAGIC LangChain is an intuitive open-source Python framework build automation around LLMs), and allows you to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing.

# COMMAND ----------

# Huggingface login not needed since open model
# from huggingface_hub import notebook_login

# # Login to Huggingface to get access to the model
# notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.
# MAGIC
# MAGIC Takes about 8m on g4dn.xlarge cluster (16gb, 4 cores).

# COMMAND ----------

# MAGIC %pip install -q -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# import torch
# from transformers import pipeline

# generate_text = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, 
#                          revision='d632f0c8b75b1ae5b26b250d25bfba4e99cb7c6f',
#                          trust_remote_code=True, device_map="auto", return_full_text=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can create a prompt that either has only an instruction or has an instruction with context:

# COMMAND ----------

# MAGIC %md Get constants from constants table

# COMMAND ----------



# COMMAND ----------

constants_table = "training.llm_langchain_shared.server_constants"

# COMMAND ----------

constants_df = spark.read.table(constants_table)
display(constants_df)

# COMMAND ----------

# MAGIC %md Create llm object connecting to mistral server

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
# from langchain.llms import HuggingFacePipeline
from langchain.llms import Databricks

# # template for an instrution with no input
# prompt = PromptTemplate(
#     input_variables=["instruction"],
#     template="{instruction}")

# # template for an instruction with input
# prompt_with_context = PromptTemplate(
#     input_variables=["instruction", "context"],
#     template="{instruction}\n\nInput:\n{context}")

# import os
# os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
cluster_id = '1026-140257-76wqxww4'
# cluster_id = '1023-124355-2d7zvdkz'
port = 7777
host = "dbc-11ce6ca4-7321.cloud.databricks.com"

# TODO: this cluster ID is a place holder, please replace `cluster_id` with the actual cluster ID of the server proxy app's cluster
llm = Databricks(host=host, cluster_id=cluster_id, cluster_driver_port="7777", api_token=api_token,)

# # hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
# llm = Databricks(cluster_id="0000-000000-xxxxxxxx"
#                  cluster_driver_port="7777",
#                  transform_input_fn=transform_input,
#                  transform_output_fn=transform_output,)

# llm_chain = LLMChain(llm=llm, prompt=prompt)
# llm_context_chain = LLMChain(llm=llm, prompt=prompt_with_context)

# COMMAND ----------

# To help construct our Chat Messages
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

# We will be using a chat model, defaults to gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI

# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's let's find a confusing text online.
# MAGIC Source: https://www.smithsonianmag.com/smart-news/long-before-trees-overtook-the-land-earth-was-covered-by-giant-mushrooms-13709647/

# COMMAND ----------

confusing_text = """
For the next 130 years, debate raged.
Some scientists called Prototaxites a lichen, others a fungus, and still others clung to the notion that it was some kind of tree.
“The problem is that when you look up close at the anatomy, it’s evocative of a lot of different things, but it’s diagnostic of nothing,” says Boyce, an associate professor in geophysical sciences and the Committee on Evolutionary Biology.
“And it’s so damn big that when whenever someone says it’s something, everyone else’s hackles get up: ‘How could you have a lichen 20 feet tall?’”
"""
Let's take a look at what prompt will be sent to the LLM

print ("------- Prompt Begin -------")

final_prompt = prompt.format(text=confusing_text)
print(final_prompt)

print ("------- Prompt End -------")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at what prompt will be sent to the LLM

# COMMAND ----------

print ("------- Prompt Begin -------")

final_prompt = prompt.format(text=confusing_text)
print(final_prompt)

print ("------- Prompt End -------")

# COMMAND ----------

# MAGIC %md
# MAGIC Finally let's pass it through the LLM

# COMMAND ----------

output = llm(final_prompt)
print (output)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


