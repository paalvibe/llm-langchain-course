# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tuning Billion-Parameter Models with Hugging Face and DeepSpeed
# MAGIC
# MAGIC These notebooks accompany the blog of the same name, with more complete listings and basic commentary about the steps. The blog gives fuller context about what is happening.
# MAGIC
# MAGIC **Note:** Throughout these examples, various temp paths are used to store results, under `/dbfs/tmp/`. Change them to whatever location you desire.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC
# MAGIC This example uses data from the [Amazon Customer Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), or rather just the camera product reviews, as a stand-in for "your" e-commerce site's camera reviews.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get dataset from kaggle, no longer available at aws s3 bucket 
# MAGIC
# MAGIC Start up a normal non-gpu cluster to prep the data.
# MAGIC
# MAGIC No longer available here: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz
# MAGIC
# MAGIC
# MAGIC 1. Find the download link here, after logging in and paste the link into the command below:
# MAGIC
# MAGIC https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?resource=download&select=amazon_reviews_us_Camera_v1_00.tsv
# MAGIC
# MAGIC 2. Unzip the file
# MAGIC 3. Gz the file
# MAGIC 3. Upload the gz file to a new volume under Data, as a managed volume, at this path:
# MAGIC /Volumes/training/awsreviews/awsreviews/amazon_reviews_us_Camera_v1_00.tsv.gz
# MAGIC
# MAGIC see images/upload_reviews_data_set.png for screenshot.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define path as constant for python and bash

# COMMAND ----------

import sys
sys.path.insert(0, '..')
import envsetup
envsetup.setup_env(dbutils, spark)

# COMMAND ----------

# MAGIC %ls /Volumes/training/awsreviews/awsreviews/

# COMMAND ----------

import os

REVIEWSFILE = "/Volumes/training/awsreviews/awsreviews/amazon_reviews_us_Camera_v1_00.tsv.gz"
os.environ['REVIEWSFILE'] = REVIEWSFILE

REVIEWS_DEST_PATH = "/Volumes/training/awsreviews/awsreviews/csvs" 
os.environ['REVIEWS_DEST_PATH'] = REVIEWS_DEST_PATH
TRAINING_CSVS_PATH = f"{REVIEWS_DEST_PATH}/training_csvs"
os.environ['TRAINING_CSVS_PATH'] = TRAINING_CSVS_PATH

# COMMAND ----------

# MAGIC %sh 
# MAGIC mkdir -p $REVIEWS_DEST_PATH
# MAGIC # Check that the file is available
# MAGIC ls $REVIEWSFILE

# COMMAND ----------

camera_reviews_df = spark.read.options(delimiter="\t", header=True).\
  csv(REVIEWSFILE)
display(camera_reviews_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC The data needs a little cleaning because it contains HTML tags, escapes, and other markdown that isn't worth handling further. Simply replace these with spaces in a UDF.
# MAGIC The functions below also limit the number of tokens in the result, and try to truncate the review to end on a sentence boundary. This makes the resulting review more realistic to learn from; it shouldn't end in the middle of a sentence! The result is just saved as a Delta table.

# COMMAND ----------

import re
from pyspark.sql.functions import udf

# Some simple (simplistic) cleaning: remove tags, escapes, newlines
# Also keep only the first N tokens to avoid problems with long reviews
remove_regex = re.compile(r"(&[#0-9]+;|<[^>]+>|\[\[[^\]]+\]\]|[\r\n]+)")
split_regex = re.compile(r"([?!.]\s+)")

def clean_text(text, max_tokens):
  if not text:
    return ""
  text = remove_regex.sub(" ", text.strip()).strip()
  approx_tokens = 0
  cleaned = ""
  for fragment in split_regex.split(text):
    approx_tokens += len(fragment.split(" "))
    if (approx_tokens > max_tokens):
      break
    cleaned += fragment
  return cleaned.strip()

@udf('string')
def clean_review_udf(review):
  return clean_text(review, 100)

@udf('string')
def clean_summary_udf(summary):
  return clean_text(summary, 20)

# Pick examples that have sufficiently long review and headline
camera_reviews_df.select("product_id", "review_body", "review_headline").\
  sample(0.1, seed=42).\
  withColumn("review_body", clean_review_udf("review_body")).\
  withColumn("review_headline", clean_summary_udf("review_headline")).\
  filter("LENGTH(review_body) > 0 AND LENGTH(review_headline) > 0").\
  write.format("delta").save(f"{REVIEWS_DEST_PATH}/cleaned")

# COMMAND ----------

# MAGIC %sh
# MAGIC # Check cleaned path content
# MAGIC ls $REVIEWS_DEST_PATH/cleaned

# COMMAND ----------

camera_reviews_cleaned_df = spark.read.format("delta").load(f"{REVIEWS_DEST_PATH}/cleaned").\
  select("review_body", "review_headline").toDF("text", "summary")
display(camera_reviews_cleaned_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Fine-tuning will need this data as simple CSV files. Split the data into train/validation sets and write as CSV for later

# COMMAND ----------

# MAGIC %sh mkdir -p $REVIEWS_DEST_PATH/training_csvs

# COMMAND ----------

# MAGIC %sh ls $REVIEWS_DEST_PATH/training_csvs

# COMMAND ----------

train_df, val_df = camera_reviews_cleaned_df.randomSplit([0.9, 0.1], seed=42)
train_df.toPandas().to_csv(f"{TRAINING_CSVS_PATH}/camera_reviews_train.csv", index=False)
val_df.toPandas().to_csv(f"{TRAINING_CSVS_PATH}/camera_reviews_val.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### List output files

# COMMAND ----------

# MAGIC %sh ls -lh $TRAINING_CSVS_PATH

# COMMAND ----------


