# Databricks notebook source
# Global constants and defaults

# COMMAND ----------

TRAINING_GROUP = "dataops-workshop"
print(f"TRAINING_GROUP: {TRAINING_GROUP}")

# COMMAND ----------

def create_constants_table(*, server_name):
    constants_table = f"training.llm_langchain_shared.{server_name}_constants"

    schema = "training.llm_langchain_shared"
    # Grant select and modify permissions for the table to all users on the account.
    # This also works for other account-level groups and individual users.
    spark.sql(f"""
        CREATE SCHEMA IF NOT EXISTS {schema};
    """)
    spark.sql(f"""
        GRANT USE SCHEMA
        ON schema {schema}
        TO `account users`""")

    spark.sql(f"""DROP TABLE IF EXISTS {constants_table}""")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {constants_table}
        (
            name STRING,
            var STRING
        )""")

    # Grant select and modify permissions for the table to all users on the account.
    # This also applies to other account-level groups and individual users.
    spark.sql(f"""
        GRANT SELECT
        ON TABLE {constants_table}
        TO `account users`""")

    # Set ownership of table to training group so all training users can recreate these credentials
    spark.sql(f"""
        ALTER TABLE {constants_table} SET OWNER TO `{TRAINING_GROUP}`;""")
    print("Create schema and grant use rights")

    return constants_table
