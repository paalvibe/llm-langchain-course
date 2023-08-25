# Databricks notebook source
# MAGIC %md
# MAGIC ## Fine-Tuning with t5-small
# MAGIC
# MAGIC Based on blogpost https://www.databricks.com/blog/2023/03/20/fine-tuning-large-language-models-hugging-face-and-deepspeed.html
# MAGIC
# MAGIC This demonstrates basic fine-tuning with the `t5-small` model. This notebook should be run on an instance with 1 Ampere architecture GPU, such as an A10. Use Databricks Runtime 12.2 ML GPU or higher.
# MAGIC
# MAGIC This requires a few additional Python libraries, including an update to the very latest `transformers`, and additional CUDA tools:

# COMMAND ----------

# Install from source to get minimum version 4.33.0.dev0
%pip install git+https://github.com/huggingface/transformers

# COMMAND ----------

# MAGIC %pip install 'accelerate>=0.20.3' datasets evaluate rouge-score

# COMMAND ----------

# Load new libs
dbutils.library.restartPython() 

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
# MAGIC ## Run the model fine tuning
# MAGIC
# MAGIC We will now run the fine tuning script using the reviews CSVs that were prepared in the 00_data_preparation notebook.
# MAGIC
# MAGIC It will take about 1hour on an g5.xlarge AWS instance.
# MAGIC
# MAGIC If the model exists, we do not need to retrain it.

# COMMAND ----------

import os

os.environ['MLFLOW_EXPERIMENT_NAME'] = f"/Users/{envsetup.EMAIL}/fine-tuning-t5"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

# COMMAND ----------

os.environ['MLFLOW_EXPERIMENT_NAME']

# COMMAND ----------

# MAGIC %sh 
# MAGIC # Check that script is available
# MAGIC ls $SUMMARIZATION_SCRIPT_PATH/run_summarization.py

# COMMAND ----------

# MAGIC %md
# MAGIC The `run_summarization.py` script is simply obtained from [transformers examples](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py). Copy it into a repo of your choice, or simply sparse check-out the transformers repo and include only `examples/pytorch/summarization`. Either way, edit the paths below to correspond to the location of the runner script.

# COMMAND ----------

# MAGIC %sh 
# MAGIC # Check that csvs are there
# MAGIC echo $TRAINING_CSVS_PATH
# MAGIC ls $TRAINING_CSVS_PATH

# COMMAND ----------

T5_SMALL_SUMMARY_MODEL_PATH = f"{envsetup.REVIEWS_DEST_PATH}/t5-small-summary"
os.environ['T5_SMALL_SUMMARY_MODEL_PATH'] = T5_SMALL_SUMMARY_MODEL_PATH
T5_SMALL_SUMMARY_MODEL_PATH

# COMMAND ----------

ls $T5_SMALL_SUMMARY_MODEL_PATH/*.model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if model already exists

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
# MAGIC ### Tuning the model
# MAGIC
# MAGIC Example result:
# MAGIC ```
# MAGIC 100%|██████████| 281/281 [02:02<00:00,  2.29it/s]
# MAGIC [INFO|modelcard.py:452] 2023-08-25 15:20:41,345 >> Dropping the following result as it does not have all the necessary fields:
# MAGIC {'task': {'name': 'Summarization', 'type': 'summarization'}, 'metrics': [{'name': 'Rouge1', 'type': 'rouge', 'value': 28.0495}]}
# MAGIC ***** eval metrics *****
# MAGIC   epoch                   =        8.0
# MAGIC   eval_gen_len            =     7.0363
# MAGIC   eval_loss               =     2.5175
# MAGIC   eval_rouge1             =    28.0495
# MAGIC   eval_rouge2             =    17.1989
# MAGIC   eval_rougeL             =     27.726
# MAGIC   eval_rougeLsum          =    27.7482
# MAGIC   eval_runtime            = 0:02:03.92
# MAGIC   eval_samples            =      17946
# MAGIC   eval_samples_per_second =    144.815
# MAGIC   eval_steps_per_second   =      2.268
# MAGIC   
# MAGIC   Command took 58.80 minutes... (on g5.xlarge AWS instance)
# MAGIC   ```

# COMMAND ----------

# %sh 
# # Comment in this code to rerun trainging. NOT NEEDED IF MODEL EXISTS at T5_SMALL_SUMMARY_MODEL_PATH

# export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS && python \
#     $SUMMARIZATION_SCRIPT_PATH/run_summarization.py \
#     --model_name_or_path t5-small \
#     --do_train \
#     --do_eval \
#     --train_file $TRAINING_CSVS_PATH/camera_reviews_train.csv \
#     --validation_file $TRAINING_CSVS_PATH/camera_reviews_val.csv \
#     --source_prefix "summarize: " \
#     --output_dir $REVIEWS_DEST_PATH/t5-small-summary \
#     --optim adafactor \
#     --num_train_epochs 8 \
#     --bf16 \
#     --per_device_train_batch_size 64 \
#     --per_device_eval_batch_size 64 \
#     --predict_with_generate \
#     --run_name "t5-small-fine-tune-reviews"

# COMMAND ----------

# MAGIC %sh
# MAGIC # Show some outputs in the model directory
# MAGIC echo "$T5_SMALL_SUMMARY_MODEL_PATH"
# MAGIC ls $T5_SMALL_SUMMARY_MODEL_PATH/*.model
# MAGIC ls $T5_SMALL_SUMMARY_MODEL_PATH/*.json

# COMMAND ----------

# MAGIC %md
# MAGIC Same inference code as before, just built using the fine-tuned model that was produced above:

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

display(review_by_product_df.select("reviews", "summary").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve as model checkpoint
# MAGIC This model can even be managed by MLFlow by wrapping up its usage in a simple custom `PythonModel`.
# MAGIC
# MAGIC This way it can be usd by other workflows or services as an endpoint to produce a review.

# COMMAND ----------

import mlflow
import torch

class ReviewModel(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context):
    self.pipeline = pipeline("summarization", \
      model=context.artifacts["pipeline"], tokenizer=context.artifacts["pipeline"], \
      num_beams=10, min_new_tokens=50, \
      device=0 if torch.cuda.is_available() else -1)
    
  def predict(self, context, model_input): 
    texts = ("summarize: " + model_input.iloc[:,0]).to_list()
    pipe = self.pipeline(texts, truncation=True, batch_size=8)
    return pd.Series([s['summary_text'] for s in pipe])

# COMMAND ----------

# MAGIC %md Copy everything but the checkpoints, which are large and not necessary to serve the model

# COMMAND ----------

# MAGIC %sh rm -r /tmp/t5-small-summary ; mkdir -p /tmp/t5-small-summary ; cp $T5_SMALL_SUMMARY_MODEL_PATH/* /tmp/t5-small-summary

# COMMAND ----------

# Define an experiment path to log the model and register it
experiment_path = f"/Users/{envsetup.EMAIL}/fine-tuning-t5"
mlflow.set_experiment(experiment_path)
last_run_id = mlflow.search_runs(filter_string="tags.mlflow.runName	= 't5-small-fine-tune-reviews'")['run_id'].item()

with mlflow.start_run(run_id=last_run_id):
  mlflow.pyfunc.log_model(artifacts={"pipeline": "/tmp/t5-small-summary"}, 
    artifact_path="review_summarizer", 
    python_model=ReviewModel(),
    registered_model_name="sean_t5_small_fine_tune_reviews")

# COMMAND ----------

# MAGIC %md
# MAGIC This model can then be deployed as a real-time endpoint! Check the `Models` and `Endpoints` tabs to the left in Databricks.
# MAGIC
# MAGIC What would the latency be like for such a model? if latency is important, then one might serve the model using GPUs (coming soon to Databricks Model Serving). Test latency on a single input, and run this on a GPU cluster:

# COMMAND ----------

# MAGIC %md
# MAGIC ## Measure the latency of a review

# COMMAND ----------

sample_review = "summarize: " + review_by_product_df.select("reviews").head(1)[0]["reviews"]

summarizer_pipeline = pipeline("summarization",\
  model=T5_SMALL_SUMMARY_MODEL_PATH,\
  tokenizer=T5_SMALL_SUMMARY_MODEL_PATH,\
  num_beams=10, min_new_tokens=50, device="cuda:0")

# COMMAND ----------

# MAGIC %time summarizer_pipeline(sample_review, truncation=True)

# COMMAND ----------


