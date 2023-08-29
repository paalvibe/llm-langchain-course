# Databricks notebook source
# MAGIC %md
# MAGIC ## Applying T5 without Fine-Tuning
# MAGIC
# MAGIC To start, these examples show again how to apply T5 for summarization, without fine-tuning, as a baseline.
# MAGIC
# MAGIC Helpful tip: set `HUGGINGFACE_HUB_CACHE` to a consistent location on `/dbfs` across jobs, and you can avoid downloading large models repeatedly.
# MAGIC (It's also possible to set a cache dir for datasets downloaded from Hugging Face, but this isn't relevant in this example.)

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

dbutils.library.restartPython()
from transformers.utils import check_min_version
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.32")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load shared environment setup

# COMMAND ----------

import sys
sys.path.insert(0, '..')
import envsetup
envsetup.setup_env(dbutils, spark)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Produce reviews with off the shelf model
# MAGIC
# MAGIC Applying an off-the-shelf summarization pipeline is just a matter of wrapping it in a UDF and applying it to data in a Spark DataFrame, such as the data read from the Delta table created in the last notebooks.
# MAGIC
# MAGIC - `pandas_udf` makes inference more efficient as it can apply the model to batches of data at a time
# MAGIC - `broadcast`ing the pipeline is optional but makes transfer and reuse of the model in UDFs on the workers faster 
# MAGIC - This won't work for models over 2GB in size, though `pandas_udf` provides another pattern to load the model once and apply it many times in this case (not shown here)

# COMMAND ----------

from transformers import pipeline
from pyspark.sql.functions import pandas_udf
import pandas as pd

summarizer_pipeline = pipeline("summarization", model="t5-large", tokenizer="t5-large", num_beams=10)
summarizer_broadcast = sc.broadcast(summarizer_pipeline)

@pandas_udf('string')
def summarize_review(reviews):
  pipe = summarizer_broadcast.value(("summarize: " + reviews).to_list(), batch_size=8, truncation=True)
  return pd.Series([s['summary_text'] for s in pipe])

camera_reviews_df = spark.read.format("delta").load(f"{envsetup.CLEAN_REVIEWS_PATH}")

display(camera_reviews_df.withColumn("summary", summarize_review("review_body")).select("review_body", "summary").limit(4))

# COMMAND ----------

# MAGIC %md
# MAGIC Summaries of individual reviews are interesting, and seems to produce plausible results. However, perhaps the more interesting application is summarizing _all_ reviews for a product into _one_ review. This is not really harder, as Spark can group the review text per item and apply the same pattern. Below we have:
# MAGIC
# MAGIC - Creating a pipeline based on `t5-large`
# MAGIC - Broadcasting it for more efficient reuse across the cluster in a UDF
# MAGIC - Creating on an efficient pandas UDF for 'vectorized' inference in parallel

# COMMAND ----------

from pyspark.sql.functions import collect_list, concat_ws, col, count

summarizer_broadcast = pipeline("summarization", model="t5-large", tokenizer="t5-large", num_beams=10, min_new_tokens=50)
summarizer_broadcast = sc.broadcast(summarizer_pipeline)

@pandas_udf('string')
def summarize_review(reviews):
  pipe = summarizer_broadcast.value(("summarize: " + reviews).to_list(), batch_size=8, truncation=True)
  return pd.Series([s['summary_text'] for s in pipe])

review_by_product_df = camera_reviews_df.groupBy("product_id").\
  agg(collect_list("review_body").alias("review_array"), count("*").alias("n")).\
  filter("n >= 10").\
  select("product_id", "n", concat_ws(" ", col("review_array")).alias("reviews")).\
  withColumn("summary", summarize_review("reviews"))

# We filter on the word hybrid to get a certain class of review
display(review_by_product_df.where("reviews like '% hybrid %'").select("reviews", "summary").limit(1))

# COMMAND ----------


