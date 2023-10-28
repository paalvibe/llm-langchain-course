# must be passed in as param as databricks lib not available in UC cluster
# from databricks.sdk.runtime import dbutils
import inspect

def username(dbutils):
    # dbutils must be passed in as param as databricks lib not available in UC cluster
    """Return username, stripped for dots, part of users's email to use as dev db prefix"""
    email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    name = email.split("@")[0].replace(".", "").replace("-", "")
    return name


def ucschema(*,
            db="llmtopptur",
            catalog="trainingmodels",
            env="dev",
            dbutils):
    uname = username(dbutils)
    db_prefix = ""
    if env == "dev":
        uname = username(dbutils)
        db_prefix = f"dev_{uname}_"
    return f"{catalog}.{db_prefix}{db}"


def ucmodel(
*, 
model,
db="llmtopptur",
catalog="trainingmodels",
env="dev",
):
    # Get dbutils from calling module, as databricks lib not available in UC cluster
    dbutils = None
    try:
        dbutils = inspect.stack()[1][0].f_globals['dbutils']
    except KeyError:
        dbutils = inspect.stack()[2][0].f_globals['dbutils']
    if not model:
        raise ValueError("model must be a non-empty string")
    if not db:
        raise ValueError("db must be a non-empty string")
    schema = ucschema(db=db, catalog=catalog, env=env, dbutils=dbutils)
    return f"{schema}.{model}"
    