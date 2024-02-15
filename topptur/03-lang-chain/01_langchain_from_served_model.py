# Databricks notebook source
# MAGIC %md
# MAGIC # Use langchain with llm served from Databricks
# MAGIC
# MAGIC We use a databricks proxy endpoint in front of OpenAI ChatGPT service.
# MAGIC
# MAGIC Can be run on a non-gpu cluster.
# MAGIC
# MAGIC ## What is Langchain?
# MAGIC
# MAGIC LangChain is an intuitive open-source Python framework build automation around LLMs), and allows you to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference examples

# COMMAND ----------

# MAGIC %pip install -q -U langchain mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md This is how you would used an open model instead
# MAGIC
# MAGIC We currently use OpenAI service, since we exhaust our GPU limitations when training too many people in parallel.
# MAGIC
# MAGIC To use an open model, the llm can be run like this, but must be done run with a GPU
# MAGIC
# MAGIC ``````python
# MAGIC from vllm import LLM
# MAGIC
# MAGIC # it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of Mistral-7B-Instruct-v0. in https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/commits/main
# MAGIC model = "mistralai/Mistral-7B-Instruct-v0.1"
# MAGIC revision = "3dc28cf29d2edd31a0a7b8f0b21637059815b4d5"
# MAGIC
# MAGIC llm = LLM(model=model, revision=revision)
# MAGIC ``````


# COMMAND ----------

import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import Databricks
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")
llm = Databricks(host=host, endpoint_name="azure_openai_training", max_tokens=1024)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Prompt parameters

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can create a prompt that either has only an instruction or has an instruction with context:

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.llms import Databricks

# template for an instruction with no input
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

# MAGIC %md
# MAGIC Example predicting using a simple instruction:

# COMMAND ----------

print(llm_chain.predict(instruction="Explain to me the difference between nuclear fission and fusion.").lstrip())

# COMMAND ----------

context = """George Washington (February 22, 1732[b] - December 14, 1799) was an American military officer, statesman,
and Founding Father who served as the first president of the United States from 1789 to 1797."""

print(llm_context_chain.predict(instruction="When was George Washington president?", context=context).lstrip())

# COMMAND ----------

print(llm_chain.predict(instruction="When was George Washington president?").lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Non-context Example

# COMMAND ----------

print(llm_chain.predict(instruction="What determines how fast you can reply to the requests?").lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Context Example

# COMMAND ----------

context = """Haakon IV Haakonsson, byname Haakon The Old, Norwegian Håkon Håkonsson, or Håkon Den Gamle, (born 1204, Norway—died December 1263, Orkney Islands), king of Norway (1217–63) who consolidated the power of the monarchy, patronized the arts, and established Norwegian sovereignty over Greenland and Iceland. His reign is considered the beginning of the “golden age” (1217–1319) in medieval Norwegian history."""

print(llm_context_chain.predict(instruction="What characterized Haakon IV Haakonson?", context=context).lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Question to compare with embeddings task later

# COMMAND ----------

print(llm_context_chain.predict(instruction="How did Harald Hardrade help the Icelanders?", context="").lstrip())

# COMMAND ----------

print(llm_context_chain.predict(instruction="Who was Harald Hardrade a brother of?", context="").lstrip())

# COMMAND ----------

print(llm_context_chain.predict(instruction="When did Harald Hardrade marry Ellisif?", context="").lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task:
# MAGIC
# MAGIC Play around with the context and the instructions.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task: Get a good answer about Entur

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Get an even better answer by providing context

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task: Get a good description of Oslo

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Task: Try to get a description of Oslo from a French perspective (French people, not French language)

# COMMAND ----------


