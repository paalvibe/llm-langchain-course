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
# MAGIC This example uses data from the [Amazon Customer Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), or rather just the camera product reviews, as a stand-in for "your" e-commerce site's camera reviews. Simply download it and display:

# COMMAND ----------

import os
email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
os.environ['UEMAIL'] = email
email

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get dataset from kaggle, no longer available at aws s3 bucket 
# MAGIC
# MAGIC Start up a normal non-gpu cluster to prep the data.
# MAGIC
# MAGIC No longer available here: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz
# MAGIC
# MAGIC Find the download link here, after logging in and paste the link into the command below:
# MAGIC
# MAGIC https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?resource=download&select=amazon_reviews_us_Camera_v1_00.tsv

# COMMAND ----------

# No longer working since dead link
# %sh mkdir -p /dbfs/tmp/$UEMAIL/review ; curl -s https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz | gunzip > /dbfs/tmp/$UEMAIL/review/amazon_reviews_us_Camera_v1_00.tsv

# COMMAND ----------

# MAGIC %sh curl -s "https://storage.googleapis.com/kaggle-data-sets/1412891/2342537/compressed/amazon_reviews_us_Camera_v1_00.tsv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230824%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230824T134819Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4f4bbc1eb2a3a1291ae9911ab1d3f075d5c5ce8c0cad165088491aabec6b1b4256b01ac56a1cad30e8615ea1963bf71a871f10fb13f90d5e7d8c2e116bdf73f987d3365e8e3e7c2c56bdfb274e15d1a45e8c783cfbe91d68e1f17220fe3a3812cbd54ae580017da602ff8ba596909c264282feea63610e0c1d9535fc71979d674b448f6eae05c1c604ebc1bb003af8a9021de1a87693b2c63775820bf315c514e886c07996a27e1ba1fc01b329736c0e9ee91d85e6c0587f1d27470e86a8b11f66ef84ddc8b86457aa515344cd2d60c977cab8750fa1734be1abe321dc8e66988465315c2d44781eb6dd7e9e3021a19de3c170ba777566b373733762190214a3" > /dbfs/tmp/$UEMAIL/review/amazon_reviews_us_Camera_v1_00.tsv.zip

# COMMAND ----------

# MAGIC %sh ls -l /dbfs/tmp/$UEMAIL/review/

# COMMAND ----------

# MAGIC %sh unzip /dbfs/tmp/$UEMAIL/review/amazon_reviews_us_Camera_v1_00.tsv.zip

# COMMAND ----------

camera_reviews_df = spark.read.options(delimiter="\t", header=True).\
  csv(f"/tmp/$email/review/amazon_reviews_us_Camera_v1_00.tsv")
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
  write.format("delta").save(f"/tmp/$email/review/cleaned")

# COMMAND ----------

camera_reviews_cleaned_df = spark.read.format("delta").load(f"/tmp/$email/review/cleaned").\
  select("review_body", "review_headline").toDF("text", "summary")
display(camera_reviews_cleaned_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Fine-tuning will need this data as simple CSV files. Split the data into train/validation sets and write as CSV for later

# COMMAND ----------

train_df, val_df = camera_reviews_cleaned_df.randomSplit([0.9, 0.1], seed=42)
train_df.toPandas().to_csv(f"/dbfs/tmp/email/review/camera_reviews_train.csv", index=False)
val_df.toPandas().to_csv(f"/dbfs/tmp/email/review/camera_reviews_val.csv", index=False)

# COMMAND ----------


