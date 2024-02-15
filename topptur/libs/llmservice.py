from langchain_community.llms import Databricks
from langchain import PromptTemplate, LLMChain


def llmservice(*, host, endpoint_name, max_tokens=1024):
    if endpoint_name.startswith("azure_openai_training"):
        return Databricks(endpoint_name=endpoint_name)
        Databricks(host=host, endpoint_name=endpoint_name, max_tokens=max_tokens)