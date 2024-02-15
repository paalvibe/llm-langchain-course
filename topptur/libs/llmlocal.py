from langchain.llms import Databricks


def llmlocal(*, server_name, server_num, spark):
    server_num = 1 # Use same num as the group you have been given (1-6)
    constants_table = f"training.llm_langchain_shared.{server_name}{server_num}_constants"
    constants_df = spark.read.table(constants_table)
    # display(constants_df)
    raw_dict = constants_df.toPandas().to_dict()
    names = raw_dict['name'].values()
    vars = raw_dict['var'].values()
    constants = dict(zip(names, vars))
    cluster_id = constants['cluster_id']
    port = constants['port']
    host = constants['host']
    api_token = constants['api_token']
    print(f"Connecting to llm on {host}/{cluster_id}/{port}")
    llm = Databricks(host=host, cluster_id=cluster_id, cluster_driver_port=port, api_token=api_token,)
    return llm
