# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Check the contents of the models backup folder

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/training/backup

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Model folder sizes
# MAGIC
# MAGIC The models are stored inside these folders:
# MAGIC
# MAGIC /Volumes/training/backup/t5-small-summary
# MAGIC /Volumes/training/backup/t5-large-summary
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC du /Volumes/training/backup

# COMMAND ----------


