import os

# MODEL_CACHE_PATH, REVIEWS_BASE_PATH, REVIEWS_DEST_PATH, CLEAN_REVIEWS_PATH, TRAINING_CSVS_PATH, SUMMARIZATION_SCRIPT_PATH

def setup_env(dbutils, spark):
    global EMAIL, MODEL_CACHE_PATH, REVIEWS_BASE_PATH, REVIEWS_DEST_PATH, CLEAN_REVIEWS_PATH, TRAINING_CSVS_PATH, SUMMARIZATION_SCRIPT_PATH
    os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
    EMAIL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    MODEL_CACHE_PATH = f"/dbfs/tmp/{EMAIL}/cache/hf"
    os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_PATH
    # reviews base path
    REVIEWS_BASE_PATH = "/Volumes/training/awsreviews/awsreviews"
    # reviews dest path
    REVIEWS_DEST_PATH = f"{REVIEWS_BASE_PATH}/csvs" 
    os.environ['REVIEWS_DEST_PATH'] = REVIEWS_DEST_PATH
    # clean reviews path
    CLEAN_REVIEWS_PATH = f"{REVIEWS_DEST_PATH}/cleaned"
    os.environ['CLEAN_REVIEWS_PATH'] = CLEAN_REVIEWS_PATH
    # training csvs path
    TRAINING_CSVS_PATH = f"{REVIEWS_DEST_PATH}/training_csvs"
    os.environ['TRAINING_CSVS_PATH'] = TRAINING_CSVS_PATH
    # script path
    SUMMARIZATION_SCRIPT_PATH = f"/Workspace/Repos/{EMAIL}/llm-tuning-course/scripts/summarization"
    os.environ['SUMMARIZATION_SCRIPT_PATH'] = SUMMARIZATION_SCRIPT_PATH