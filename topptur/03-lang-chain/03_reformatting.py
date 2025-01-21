# Databricks notebook source
# MAGIC %md
# MAGIC # Reformatting with langchain on Databricks
# MAGIC
# MAGIC We use a databricks proxy endpoint in front of OpenAI ChatGPT service.
# MAGIC
# MAGIC Can be run on a non-gpu cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.
# MAGIC
# MAGIC Takes about 8m on g4dn.xlarge cluster (16gb, 4 cores).

# COMMAND ----------

# MAGIC %pip install -q -U langchain langchain-core langchain-community
# MAGIC %pip install -q -U mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import Databricks
import os
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")
llm = Databricks(host=host, endpoint_name="azure_openai_training", max_tokens=1024)

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
# MAGIC # Task: generate and reformat a badly formatted email
# MAGIC
# MAGIC 1. Sub-task 1: Generate a badly written email with syntax errors and grammatical errors, about consultants.
# MAGIC 2. Sub-task 2: Correct the badly written email to a well written email.

# COMMAND ----------


