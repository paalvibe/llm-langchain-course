# Databricks notebook source
# MAGIC %md
# MAGIC # reformatting with langchain on Databricks
# MAGIC
# MAGIC We use a mistral model served from another cluster which has GPU.
# MAGIC
# MAGIC Can be run on a non-gpu cluster.
# MAGIC
# MAGIC ## What is Langchain?
# MAGIC
# MAGIC LangChain is an intuitive open-source Python framework build automation around LLMs), and allows you to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing.
# MAGIC
# MAGIC Examples from here:
# MAGIC
# MAGIC https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%201%20-%20Fundamentals.ipynb

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

raw_dict = constants_df.toPandas().to_dict()
names = raw_dict['name'].values()
vars = raw_dict['var'].values()
constants = dict(zip(names, vars))
cluster_id = constants['cluster_id']
port = constants['port']
host = constants['host']

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
# cluster_id = '1026-140257-76wqxww4'
# cluster_id = '1023-124355-2d7zvdkz'
# port = 7777
# host = "dbc-11ce6ca4-7321.cloud.databricks.com"

# TODO: this cluster ID is a place holder, please replace `cluster_id` with the actual cluster ID of the server proxy app's cluster
llm = Databricks(host=host, cluster_id=cluster_id, cluster_driver_port=port, api_token=api_token,)

# # hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
# llm = Databricks(cluster_id="0000-000000-xxxxxxxx"
#                  cluster_driver_port="7777",
#                  transform_input_fn=transform_input,
#                  transform_output_fn=transform_output,)

# llm_chain = LLMChain(llm=llm, prompt=prompt)
# llm_context_chain = LLMChain(llm=llm, prompt=prompt_with_context)

# COMMAND ----------

# # To help construct our Chat Messages
# from langchain.schema import HumanMessage
# from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

# # We will be using a chat model, defaults to gpt-3.5-turbo
# from langchain.chat_models import ChatOpenAI

# # To parse outputs and get structured data back
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's let's find a confusing text online.
# MAGIC Source: https://www.smithsonianmag.com/smart-news/long-before-trees-overtook-the-land-earth-was-covered-by-giant-mushrooms-13709647/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Parsers Method 1: Prompt Instructions & String Parsing

# COMMAND ----------

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# COMMAND ----------

# How you would like your response structured. This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# COMMAND ----------

# See the prompt template you created for formatting
format_instructions = output_parser.get_format_instructions()
print (format_instructions)

# COMMAND ----------

template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

promptValue = prompt.format(user_input="welcom to califonya!")

print(promptValue)

# COMMAND ----------

llm_output = llm(promptValue)
llm_output

# COMMAND ----------

# MAGIC %md
# MAGIC # Output Parsers Method 2: Fill out a base model

# COMMAND ----------

# MAGIC %md
# MAGIC Example 1: Simple
# MAGIC
# MAGIC Let's get started by defining a simple model for us to extract from.

# COMMAND ----------


from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

# COMMAND ----------

# MAGIC %md Then let's create a chain (more on this later) that will do the extracting for us

# COMMAND ----------

text = "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
promptValue = prompt.format(user_input=text)
llm(promptValue)

# COMMAND ----------

from langchain.chains import create_extraction_chain
# Schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "integer"},
        "hair_color": {"type": "string"},
    },
    "required": ["name", "height"],
}

# Input 
inp = """Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde."""

# Run chain
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
chain = create_extraction_chain(schema, llm)
chain.run(inp)

# COMMAND ----------

from langchain.chains.openai_functions import create_structured_output_chain

chain = create_structured_output_chain(Person, llm, prompt)
chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
)

# COMMAND ----------



# COMMAND ----------


