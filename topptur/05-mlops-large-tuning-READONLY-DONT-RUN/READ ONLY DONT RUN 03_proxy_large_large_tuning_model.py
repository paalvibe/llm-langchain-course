# Databricks notebook source
# MAGIC %md
# MAGIC ## Deploy large fine-tuned model
# MAGIC
# MAGIC Based on blogpost https://www.databricks.com/blog/2023/03/20/fine-tuning-large-language-models-hugging-face-and-deepspeed.html
# MAGIC
# MAGIC This demonstrates basic fine-tuning with the `t5-large` model. This notebook should be run on an instance with 1 Ampere architecture GPU, such as an A10. Use Databricks Runtime 12.2 ML GPU or higher. on AWS you can use `g5.xlarge`.
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
T5_LARGE_SUMMARY_MODEL_PATH = f"{envsetup.REVIEWS_DEST_PATH}/{envsetup.LARGE_TUNED_MODEL}"
os.environ['T5_LARGE_SUMMARY_MODEL_PATH'] = T5_LARGE_SUMMARY_MODEL_PATH
T5_LARGE_SUMMARY_MODEL_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Check if the model files are still around
# MAGIC
# MAGIC We check that the files are still here.
# MAGIC
# MAGIC If not, run the model fine tuning in ./02_large_fine_tuning

# COMMAND ----------

# MAGIC %sh
# MAGIC MODELFILE=$T5_LARGE_SUMMARY_MODEL_PATH/spiece.model
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
  model=T5_LARGE_SUMMARY_MODEL_PATH,\
  tokenizer=T5_LARGE_SUMMARY_MODEL_PATH,\
  num_beams=10, min_new_tokens=50)
summarizer_broadcast = sc.broadcast(summarizer_pipeline)

@pandas_udf('string')
def summarize_review(reviews):
  print("reviews", repr(reviews))
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
# MAGIC ## Deploy model with proxy endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Serve with flask

# COMMAND ----------

# Define parameters to generate text
def gen_text_for_serving(reviews):
    #texts = ("summarize: " + prompt).to_list()
    pipe = summarizer_broadcast.value(["summarize: " + reviews], batch_size=8, truncation=True)
    return pd.Series([s['summary_text'] for s in pipe])

# COMMAND ----------


pd.DataFrame({
    [rewview]

})
print(gen_text_for_serving("I actually did not receive it, was offered a refund and we parted ways. Was sent a pair of speakers, that only work overseas because of the power plug. was told to dispose of them any way I desireed. I finally broke away from my 35mm Nikkon and decided to try a digital one...one draw back, the LCD screen is impossible to view in bright light. I had to move to shade or snap and hope they came out. I was able to view in the index at the pics to see if they turned out before I moved on which helped. There is a ev+1.5 to ev-1.5 and a brightness control along with choice of event setting. We bought our camera this past spring to use at a wedding.  It was awesome."))

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("large_fine_tuning")

@app.route('/', methods=['POST'])
def serve_large_fine_tuning_instruct():
  resp = gen_text_for_serving(**request.json)
  return jsonify(resp)

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
  model=T5_LARGE_SUMMARY_MODEL_PATH,\
  tokenizer=T5_LARGE_SUMMARY_MODEL_PATH,\
  num_beams=10, min_new_tokens=50, device="cuda:0")

# COMMAND ----------

# MAGIC %time summarizer_pipeline(sample_review, truncation=True)

# COMMAND ----------


