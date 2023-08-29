# Databricks notebook source
# MAGIC %md
# MAGIC ## Deploy small fine-tuned model
# MAGIC
# MAGIC Based on blogpost https://www.databricks.com/blog/2023/03/20/fine-tuning-large-language-models-hugging-face-and-deepspeed.html
# MAGIC
# MAGIC This demonstrates basic fine-tuning with the `t5-small` model. This notebook should be run on an instance with 1 Ampere architecture GPU, such as an A10. Use Databricks Runtime 12.2 ML GPU or higher. on AWS you can use `g5.xlarge`.
# MAGIC
# MAGIC This requires a few additional Python libraries, including an update to the very latest `transformers`, and additional CUDA tools.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install packages
# MAGIC
# MAGIC We need some bleeding edge packages to get it to run.
# MAGIC These must be installed in each notebook.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/huggingface/transformers

# COMMAND ----------

# MAGIC %pip install 'accelerate>=0.20.3' datasets evaluate rouge-score

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Check that necessary packages are available

# COMMAND ----------

# Load new libs
dbutils.library.restartPython() 
from transformers.utils import check_min_version
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.32")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load paths and env vars

# COMMAND ----------

import sys
sys.path.insert(0, '..')
import envsetup
envsetup.setup_env(dbutils, spark)

# COMMAND ----------

# MAGIC %md
# MAGIC Set additional environment variables to enable integration between Hugging Face's training and MLflow hosted in Databricks (and make sure to use the shared cache again!)
# MAGIC You can also set `HF_MLFLOW_LOG_ARTIFACTS` to have it log all checkpoints to MLflow, but they can be large.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the model vars
# MAGIC
# MAGIC We define some paths used to find the model.

# COMMAND ----------

import os
T5_SMALL_SUMMARY_MODEL_PATH = f"{envsetup.REVIEWS_DEST_PATH}/{envsetup.SMALL_TUNED_MODEL}"
os.environ['T5_SMALL_SUMMARY_MODEL_PATH'] = T5_SMALL_SUMMARY_MODEL_PATH
T5_SMALL_SUMMARY_MODEL_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Check if the model files are still around
# MAGIC
# MAGIC We check that the files are still here.
# MAGIC
# MAGIC If not, run the model fine tuning in ./02_small_fine_tuning

# COMMAND ----------

# MAGIC %sh
# MAGIC MODELFILE=$T5_SMALL_SUMMARY_MODEL_PATH/spiece.model
# MAGIC if [ -f $MODELFILE ]; then
# MAGIC    echo "Tuned model $MODELFILE already exists, no need to build again."
# MAGIC else
# MAGIC    echo "Tuned model $MODELFILE does not exist."
# MAGIC fi
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check that the cleaned reviews are available
# MAGIC
# MAGIC We use this for testing the model

# COMMAND ----------

CLEANED_REVIEWS_PATH = f"{envsetup.REVIEWS_DEST_PATH}/cleaned"
os.environ['CLEANED_REVIEWS_PATH'] = CLEANED_REVIEWS_PATH

# COMMAND ----------

# MAGIC %sh
# MAGIC # Show contents of cleaned data directory
# MAGIC echo "Dir: $CLEANED_REVIEWS_PATH"
# MAGIC ls $CLEANED_REVIEWS_PATH | head -n 3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the tuned model

# COMMAND ----------

from pyspark.sql.functions import collect_list, concat_ws, col, count, pandas_udf
from transformers import pipeline
import pandas as pd

summarizer_pipeline = pipeline("summarization",\
  model=T5_SMALL_SUMMARY_MODEL_PATH,\
  tokenizer=T5_SMALL_SUMMARY_MODEL_PATH,\
  num_beams=10, min_new_tokens=50)
summarizer_broadcast = sc.broadcast(summarizer_pipeline)

@pandas_udf('string')
def summarize_review(reviews):
  pipe = summarizer_broadcast.value(("summarize: " + reviews).to_list(), batch_size=8, truncation=True)
  return pd.Series([s['summary_text'] for s in pipe])

camera_reviews_df = spark.read.format("delta").load(CLEANED_REVIEWS_PATH)

review_by_product_df = camera_reviews_df.groupBy("product_id").\
  agg(collect_list("review_body").alias("review_array"), count("*").alias("n")).\
  filter("n >= 10").\
  select("product_id", "n", concat_ws(" ", col("review_array")).alias("reviews")).\
  withColumn("summary", summarize_review("reviews"))

display(review_by_product_df.select("reviews", "summary").limit(2))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Deploy model as a service endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Measure the latency of a review
# MAGIC
# MAGIC
# MAGIC What would the latency be like for such a model? if latency is important, then one might serve the model using GPUs (using Databricks Model Serving). Test latency on a single input, and run this on a GPU cluster.

# COMMAND ----------

sample_review = "summarize: " + review_by_product_df.select("reviews").head(1)[0]["reviews"]

summarizer_pipeline = pipeline("summarization",\
  model=T5_SMALL_SUMMARY_MODEL_PATH,\
  tokenizer=T5_SMALL_SUMMARY_MODEL_PATH,\
  num_beams=10, min_new_tokens=50, device="cuda:0")

# COMMAND ----------

# MAGIC %time summarizer_pipeline(sample_review, truncation=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the model to MLFlow
# MAGIC This model can even be managed by MLFlow by wrapping up its usage in a simple custom `PythonModel`.
# MAGIC
# MAGIC This way it can be usd by other workflows or services as an endpoint to produce a review.

# COMMAND ----------

import os

os.environ['MLFLOW_EXPERIMENT_NAME'] = envsetup.SMALL_TUNED_ML_EXPERIMENT_PATH
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

import mlflow
import torch

# Define the model wrapper class
class ReviewModel(mlflow.pyfunc.PythonModel):
  
    def load_context(self, context):
        self.pipeline = pipeline("summarization", \
            model=context.artifacts["pipeline"], tokenizer=context.artifacts["pipeline"], \
            num_beams=10, min_new_tokens=50, \
            device=0 if torch.cuda.is_available() else -1)
    
    def predict(self, context, model_input):
        prompt = model_input['prompt']
        texts = ("summarize: " + prompt).to_list()
        pipe = self.pipeline(texts, truncation=True, batch_size=8)
        return pd.Series([s['summary_text'] for s in pipe])

# COMMAND ----------

# MAGIC %md Copy everything but the checkpoints, which are large and not necessary to serve the model

# COMMAND ----------

# MAGIC %sh mkdir -p /tmp/$EMAIL/$SMALL_TUNED_MODEL ; rm -r /tmp/$EMAIL/$SMALL_TUNED_MODEL/* ; cp $T5_SMALL_SUMMARY_MODEL_PATH/* /tmp/$EMAIL/$SMALL_TUNED_MODEL 2>/dev/null

# COMMAND ----------

# Configure MLflow Python client to register model in Unity Catalog
import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

envsetup.SMALL_TUNED_ML_EXPERIMENT_PATH

# COMMAND ----------

# Define an experiment path to log the model and register it
mlflow.set_experiment(envsetup.SMALL_TUNED_ML_EXPERIMENT_PATH)
last_run_id = mlflow.search_runs(filter_string="tags.mlflow.runName	= 't5-small-fine-tune-reviews'")['run_id'].item()

# We need to setup the model schema in Unity Catalog if it doesn't exist
MODELS_SCHEMA = envsetup.MODELS_SCHEMA
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {envsetup.MODELS_SCHEMA}")

MODEL_NAME = envsetup.SMALL_TUNED_MODEL_UC
registered_model_name = MODEL_NAME
# "sean_t5_small_fine_tune_reviews"

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt": ["I actually did not receive it, was offered a refund and we parted ways. Was sent a pair of speakers, that only work overseas because of the power plug. was told to dispose of them any way I desireed. I finally broke away from my 35mm Nikkon and decided to try a digital one...one draw back, the LCD screen is impossible to view in bright light. I had to move to shade or snap and hope they came out. I was able to view in the index at the pics to see if they turned out before I moved on which helped. There is a ev+1.5 to ev-1.5 and a brightness control along with choice of event setting. We bought our camera this past spring to use at a wedding.  It was awesome."]})

artifacts_path = f"/tmp/{envsetup.EMAIL}/{envsetup.SMALL_TUNED_MODEL}/"

artifact_title = "review_summarizer"

with mlflow.start_run(run_id=last_run_id) as run:
  mlflow.pyfunc.log_model(artifacts={"pipeline": artifacts_path}, 
    artifact_path=artifact_title,
    python_model=ReviewModel(),
    pip_requirements=["torch", "transformers", "accelerate", "sentencepiece", "datasets", "evaluate", "rouge-score"],
    registered_model_name=registered_model_name,
    input_example=input_example,
    signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC This model can then be deployed as a real-time endpoint! Check the `Models` and `Endpoints` tabs to the left in Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model to Unity Catalog
# MAGIC
# MAGIC By default, MLflow registers models in the Databricks workspace model registry. To register models in Unity Catalog instead, we follow the [documentation](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html) and set the registry server as Databricks Unity Catalog.
# MAGIC
# MAGIC In order to register a model in Unity Catalog, there are [several requirements](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#requirements), such as Unity Catalog must be enabled in your workspace.
# MAGIC

# COMMAND ----------

# Register model to Unity Catalog
# This may take 2.2 minutes to complete

UC_PATH = envsetup.SMALL_TUNED_MODEL_UC

print(f"UC_PATH: {UC_PATH}")
registered_name = UC_PATH # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

result = mlflow.register_model(
    "runs:/" + run.info.run_id + f"/{artifact_title}",
    registered_name,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Test run the model

# COMMAND ----------

import mlflow
import pandas as pd
mlflow.set_registry_uri('databricks-uc')

loaded_model = mlflow.pyfunc.load_model(f"models:/trainingmodels.dev_paldevibe_llmtopptur.t5-small-summary/8")

# Make a prediction using the loaded model
ret = loaded_model.predict(
    {
        "prompt": "I’m really quite amazed. I purchased the Orbi 4g LTE modem-router for non-cable WiFi connection and matched the Orbi to 4 Arlo Essential security cameras. Should work, right? Never did anything like this before. Like so many of these electronic combinations that I’ve theorized about in the past, I assumed this wouldn’t work either. I talked to several advisors – Orbi technicians, Arlo community contributors. Everyone said it should work. And it did!!!! Amazing!! The Orbi LBR20 was a bit of a challenge. The Obi app was skittish … worked sometimes, sometimes got stuck. The password used to set-up my Orbi account (Netgear) was not the same as the password needed to activate the Orbi modem-router for Internet connection; the Orbi password is printed on the label at the bottom of the Orbi unit … Oh, that password. PureTalk (really AT&T) indicated that they had coverage for my barn’s Zip Code but could not guarantee the Sim card would work with Orbi LBR20 … not quite the confidence booster I needed. Ok … so brave as I am, I purchased the PureTalk Sim card and a 6 Gig data plan. After everything arrived, I first tried to connect the Orbi wireless modem-router to the Internet using the Orbi app as instructed. Nothing automatic here. After several tries, the Orbi app asked for the APN number of my Internet service provider. What’s an APN for goodness sake? Googled “PureTalk APN” and came up with “RESELLER”. I entered “RESELLER … then my PureTalk phone number as the only User ID I knew and then my PureTalk password. No contact and no Internet. Frustration!! The next day, I called PureTalk and asked for the APN number, user ID and password. After transferring through 4 different technical service associates, I received the information (APN = RESELLER, User ID = Not set up yet, Password = Not set up yet). So, I entered “RESELLER” and nothing else (no User ID and no Password). Success!! Internet access!!! Oh, so you don’t need a password in this case!! But … I was now connected to the Internet through a Orbi 4g LTE wireless modem-router … no cable. Once I got the password and APN right, Orbi activated as promised.",
    }
)
ret[0]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Deployment
# MAGIC
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `t5-small-summary` model.

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = envsetup.SMALL_TUNED_MODEL
databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model
endpoint_config = {
  "name": f"{endpoint_name}4",
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": "GPU_BIG",
      "workload_size": "Small",
      "scale_to_zero_enabled": "True"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')

# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)

if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(deploy_response.json())

# COMMAND ----------


