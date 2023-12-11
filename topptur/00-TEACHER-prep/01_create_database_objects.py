# Databricks notebook source
# MAGIC %md
# MAGIC # What's in this exercise?
# MAGIC
# MAGIC 1) Show existing databases<BR>

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW DATABASES;

# COMMAND ----------

display(spark.catalog.listDatabases())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create training catalog
# MAGIC * The cluster needs shared access mode, to be able to use unity catalog
# MAGIC   * Notice that you PT need to create the cluster in unrestricted mode, to enable shared access mode. Otherwise, you cannot work with unity catalog.
# MAGIC * The admin user running this needs needs access to create catalogs in the metastore. One way is to make the user metastore admin in https://accounts.cloud.databricks.com/

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS trainingmodels")

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS training")
# Set ownership of table to training group so all training users can recreate these credentials
spark.sql(f"""GRANT USE CATALOG ON CATALOG training TO `account users`;""")


# COMMAND ----------

# MAGIC %md
# MAGIC # Grant select on files to users
# MAGIC
# MAGIC Needed to avoid `java.lang.SecurityException: User does not have permission SELECT on any file.`
# MAGIC when reading data from /mnt/ areas.

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT SELECT ON ANY FILE TO `users`;
# MAGIC GRANT MODIFY ON ANY FILE TO `users`;

# COMMAND ----------


