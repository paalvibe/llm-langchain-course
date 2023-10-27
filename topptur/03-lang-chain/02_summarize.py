# Databricks notebook source
# MAGIC %md
# MAGIC # Summarize with langchain on Databricks
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
# MAGIC ## Summarize
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

from langchain import PromptTemplate, LLMChain
# from langchain.llms import HuggingFacePipeline
from langchain.llms import Databricks

# import os
# os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
cluster_id = '1026-140257-76wqxww4'
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

from langchain import PromptTemplate, LLMChain

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

llm_chain = LLMChain(llm=llm, prompt=prompt)
llm_context_chain = LLMChain(llm=llm, prompt=prompt_with_context)

# COMMAND ----------

# Create our template
template = """
%INSTRUCTIONS:
Please summarize the following piece of text.
Respond in a manner that a 5 year old would understand.

%TEXT:
{text}
"""

# Create a LangChain prompt template that we can insert values to later
prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's let's find a confusing text online.
# MAGIC Source: https://www.smithsonianmag.com/smart-news/long-before-trees-overtook-the-land-earth-was-covered-by-giant-mushrooms-13709647/

# COMMAND ----------

confusing_text = """
Professor Kristen Nygaard was a Norwegian computer scientist and is considered to be the father of Simula and object-oriented programming, together with Ole-Johan Dahl.

His field was informatics and he won the two most prestigious international prizes specific to that field: The ACM A.M. Turing Award (considered to be the "Nobel Prize of Computing"), in 2001 and the IEEE John von Neumann Medal. He was made Commander of The Order of Saint Olav by the King of Norway in 2000, and received a number of other honors and awards.

With Ole-Johan Dahl, he produced the initial ideas for object-oriented (OO) programming in the 1960s at the Norwegian Computing Center (NR - Norsk Regnesentral) as part of the Simula I (1961-1965) and Simula 67 (1965-1968) simulation programming languages

Nygaard and Dahl were the first to develop the concepts of class, subclass (allowing information hiding), inheritance, dynamic object creation, etc., all important aspects of the OO paradigm. Their work has led to a fundamental change in how software systems are designed and programmed, resulting in reusable, reliable, scalable applications that have streamlined the process of writing software code and facilitated software programming. Current object-oriented programming languages include C++ and Java, both widely used in programming a wide range of applications from large-scale distributed systems to small, personal applications, including personal computers, home entertainment devices, and stand-alone arcade applications.

Nygaard worked as a part-time professor emeritius in 1977, and a full-time professor emeritus from 1984-1996 at the University of Oslo.
"""

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

# MAGIC %md
# MAGIC Task:
# MAGIC
# MAGIC Get a summary which will be interesting to read for a 15 year old gamer.
# MAGIC
# MAGIC Post your best prompt and result to the slack channel #202310-langchain-topptur

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


