# Databricks notebook source
# MAGIC %md
# MAGIC ## Fine-Tuning with t5-large
# MAGIC
# MAGIC Based on blogpost https://www.databricks.com/blog/2023/03/20/fine-tuning-large-language-models-hugging-face-and-deepspeed.html
# MAGIC
# MAGIC This demonstrates basic fine-tuning with the `t5-large` model. This notebook should be run on an instance with 1 Ampere architecture GPU, such as an A10. Use Databricks Runtime 12.2 ML GPU or higher. On AWS, you can use `g5.4xlarge` with 64gb ram, 1  GPU.
# MAGIC
# MAGIC This requires a few additional Python libraries, including an update to the very latest `transformers`, and additional CUDA tools:

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
# MAGIC ## Run the model fine tuning
# MAGIC
# MAGIC We will now run the fine tuning script using the reviews CSVs that were prepared in the 00_data_preparation notebook.
# MAGIC
# MAGIC It will take about 1hour on an g5.xlarge AWS instance.
# MAGIC
# MAGIC If the model exists, we do not need to retrain it.

# COMMAND ----------

import os

os.environ['MLFLOW_EXPERIMENT_NAME'] = envsetup.LARGE_TUNED_ML_EXPERIMENT_PATH
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
# MAGIC ls /dbfs/Users/pal.de.vibe@knowit.no/

# COMMAND ----------

# MAGIC %sh 
# MAGIC # Check that csvs are there
# MAGIC echo $TRAINING_CSVS_PATH
# MAGIC ls $TRAINING_CSVS_PATH

# COMMAND ----------

import os
T5_LARGE_SUMMARY_MODEL_PATH = f"{envsetup.REVIEWS_DEST_PATH}/t5-large-summary"
os.environ['T5_LARGE_SUMMARY_MODEL_PATH'] = T5_LARGE_SUMMARY_MODEL_PATH
T5_LARGE_SUMMARY_MODEL_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if model already exists

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
# MAGIC ### Tuning the model
# MAGIC
# MAGIC It took almost 7 hours on an AWS g5.4xlarge single instance (64gb RAM, 1 GPU).
# MAGIC
# MAGIC Example result:
# MAGIC ```
# MAGIC ***** eval metrics *****
# MAGIC   epoch                   =        4.0
# MAGIC   eval_gen_len            =     6.6744
# MAGIC   eval_loss               =     2.0709
# MAGIC   eval_rouge1             =    31.1279
# MAGIC   eval_rouge2             =    20.0984
# MAGIC   eval_rougeL             =    30.8081
# MAGIC   eval_rougeLsum          =     30.806
# MAGIC   eval_runtime            = 0:17:31.07
# MAGIC   eval_samples            =      17946
# MAGIC   eval_samples_per_second =     17.074
# MAGIC   eval_steps_per_second   =      1.423
# MAGIC ERROR:asyncio:Task exception was never retrieved
# MAGIC future: <Task finished name='Task-18' coro=<ScriptMagics.shebang.<locals>._handle_stream() done, defined at /databricks/python/lib/python3.10/site-packages/IPython/core/magics/script.py:211> exception=ValueError('Separator is not found, and chunk exceed the limit')>
# MAGIC ...
# MAGIC
# MAGIC THIS ERROR DID NOT PREVENT THE CREATION OF THE MODEL, IT SEEMS. COULD HAVE BEEN AN ISSUE IN THE NOTEBOOK,
# MAGIC NOT STOPPING THE MODEL TRAINING. Also it happened after we had reached 100%.
# MAGIC
# MAGIC Command took 6.59 hours
# MAGIC ```

# COMMAND ----------

# %sh 
# Comment in this code to rerun trainging. NOT NEEDED IF MODEL EXISTS at T5_LARGE_SUMMARY_MODEL_PATH

# export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS && python \
#     $SUMMARIZATION_SCRIPT_PATH/run_summarization.py \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --train_file $TRAINING_CSVS_PATH/camera_reviews_train.csv \
#     --validation_file $TRAINING_CSVS_PATH/camera_reviews_val.csv \
#     --source_prefix "summarize: " \
#     --output_dir $REVIEWS_DEST_PATH/t5-large-summary \
#     --optim adafactor \
#     --num_train_epochs 4 \
#     --bf16 \
#     --per_device_train_batch_size 12 \
#     --per_device_eval_batch_size 12 \
#     --predict_with_generate \
#     --run_name "t5-large-fine-tune-reviews"

# COMMAND ----------

# MAGIC %sh
# MAGIC # Show some outputs in the model directory
# MAGIC echo "$T5_LARGE_SUMMARY_MODEL_PATH"
# MAGIC ls -lh $T5_LARGE_SUMMARY_MODEL_PATH/*.model
# MAGIC ls -lh $T5_LARGE_SUMMARY_MODEL_PATH/*.json

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### TEACHER ONLY: Manual backup of model files
# MAGIC
# MAGIC Do once to preserve trained files

# COMMAND ----------

# %sh
# MODEL_BACKUP_DIR=/dbfs/Users/$EMAIL/backups/models
# echo "$MODEL_BACKUP_DIR"
# ls $MODEL_BACKUP_DIR/$LARGE_TUNED_MODEL

# MODELFILE=$MODEL_BACKUP_DIR/$LARGE_TUNED_MODEL/spiece.model
# if [ -f $MODELFILE ]; then
#    echo "Tuned model backup $MODELFILE already exists, no need to copy again."
# else
#    echo "Tuned model backup $MODELFILE does not exist. Make a backup."
#    mkdir -p $MODEL_BACKUP_DIR/$LARGE_TUNED_MODEL
#    cp $T5_LARGE_SUMMARY_MODEL_PATH/* $MODEL_BACKUP_DIR/$LARGE_TUNED_MODEL 2>/dev/null
# fi

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
# MAGIC
# MAGIC Did not complete on 32gb.
# MAGIC Will take about 15m for 10 records on 5g.4xlarge 64gb, 1 GPU.
# MAGIC So we only ask for 1 record instead, which takes 2.24m.
# MAGIC
# MAGIC We filter on the word hybrid to get a certain class of reviews.

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
  pipe = summarizer_broadcast.value(("summarize: " + reviews).to_list(), batch_size=8, truncation=True)
  return pd.Series([s['summary_text'] for s in pipe])

camera_reviews_df = spark.read.format("delta").load(CLEANED_REVIEWS_PATH)

review_by_product_df = camera_reviews_df.groupBy("product_id").\
  agg(collect_list("review_body").alias("review_array"), count("*").alias("n")).\
  filter("n >= 10").\
  select("product_id", "n", concat_ws(" ", col("review_array")).alias("reviews")).\
  withColumn("summary", summarize_review("reviews"))

# We filter on the word hybrid to get a certain class of reviews
display(review_by_product_df.where("reviews like '% hybrid %'").select("reviews", "summary").limit(1))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Compare with the reviews from the small models
# MAGIC
# MAGIC Compare with the reviews from ../01-small-tuning outputs
# MAGIC
# MAGIC Which do you like better?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task: Look for some other kinds of reviews
# MAGIC
# MAGIC ...and manually compare results from the different model variations.
# MAGIC Filter on something else than "hybrid" in the where clause.

# COMMAND ----------


