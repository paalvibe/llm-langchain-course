# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced retrieval from a pdf with langchain on Databricks
# MAGIC
# MAGIC We use a mistral model served from another cluster which has GPU.
# MAGIC
# MAGIC Can be run on a non-gpu cluster.
# MAGIC
# MAGIC ## What is Langchain?
# MAGIC
# MAGIC LangChain is an intuitive open-source Python framework build automation around LLMs), and allows you to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing.
# MAGIC
# MAGIC Examples from here:
# MAGIC
# MAGIC https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%201%20-%20Fundamentals.ipynb

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Advanced Retrieval With LangChain
# MAGIC Let's go over a few more complex and advanced retrieval methods with LangChain.
# MAGIC
# MAGIC There is no one right way to retrieve data - it'll depend on your application so take some time to think about it before you jump in
# MAGIC
# MAGIC Let's have some fun
# MAGIC
# MAGIC * Multi Query - Given a single user query, use an LLM to synthetically generate multiple other queries. Use each one of the new queries to retrieve documents, take the union of those documents for the final context of your prompt
# MAGIC * Contextual Compression - Fluff remover. Normal retrieval but with an extra step of pulling out relevant information from each returned document. This makes each relevant document smaller for your final prompt (which increases information density)
# MAGIC
# MAGIC Maybe later:
# MAGIC
# MAGIC * Parent Document Retriever - Split and embed small chunks (for maximum information density), then return the parent documents (or larger chunks) those small chunks come from
# MAGIC * Ensemble Retriever - Combine multiple retrievers together
# MAGIC * Self-Query - When the retriever infers filters from a users query and applies those filters to the underlying data

# COMMAND ----------

# MAGIC %pip install -q -U langchain
# MAGIC %pip install sentence-transformers unstructured chromadb "unstructured[pdf]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Get llm server constants from constants table

# COMMAND ----------

constants_table = "training.llm_langchain_shared.server1_constants"
constants_df = spark.read.table(constants_table)
raw_dict = constants_df.toPandas().to_dict()
names = raw_dict['name'].values()
vars = raw_dict['var'].values()
constants = dict(zip(names, vars))
cluster_id = constants['cluster_id']
port = constants['port']
host = constants['host']

# COMMAND ----------

# MAGIC %md Create llm object connecting to mistral server

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import Databricks
api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
llm = Databricks(host=host, cluster_id=cluster_id, cluster_driver_port=port, api_token=api_token,)

# COMMAND ----------

# MAGIC %md
# MAGIC ![./images/rag_pipeline.png](Rag pipeline)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Advanced Retrieval With LangChain
# MAGIC Let's go over a few more complex and advanced retrieval methods with LangChain.
# MAGIC
# MAGIC There is no one right way to retrieve data - it'll depend on your application so take some time to think about it before you jump in
# MAGIC
# MAGIC Let's have some fun
# MAGIC
# MAGIC * Multi Query - Given a single user query, use an LLM to synthetically generate multiple other queries. Use each one of the new queries to retrieve documents, take the union of those documents for the final context of your prompt
# MAGIC * Contextual Compression - Fluff remover. Normal retrieval but with an extra step of pulling out relevant information from each returned document. This makes each relevant document smaller for your final prompt (which increases information density)
# MAGIC * Parent Document Retriever - Split and embed small chunks (for maximum information density), then return the parent documents (or larger chunks) those small chunks come from
# MAGIC * Ensemble Retriever - Combine multiple retrievers together
# MAGIC * Self-Query - When the retriever infers filters from a users query and applies those filters to the underlying data

# COMMAND ----------

# MAGIC %md
# MAGIC Load up our texts and documents
# MAGIC Then chunk them, and put them into a vector store

# COMMAND ----------

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# COMMAND ----------

# MAGIC %md Download an Embedding Model
# MAGIC
# MAGIC I am using the best embedding model on Huggingface’s embedding leaderboard. Feel free to use another.
# MAGIC
# MAGIC
# MAGIC # Downloading embedding model 
# MAGIC embedding_model = SentenceTransformerEmbeddings(model_name='BAAI/bge-large-en-v1.5')

# COMMAND ----------

# Downloading embedding model 
embedding_model = SentenceTransformerEmbeddings(model_name='BAAI/bge-large-en-v1.5')

# COMMAND ----------

loader = DirectoryLoader('../../data/Sagas/', glob="**/*.pdf", show_progress=True)

docs = loader.load()

# COMMAND ----------

print(f"You have {len(docs)} essays loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC Then we'll split up our text into smaller sized chunks

# COMMAND ----------

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)

print (f"Your {len(docs)} documents have been split into {len(splits)} chunks")

# COMMAND ----------

# MAGIC %md #### Adding embeddings to Chroma
# MAGIC
# MAGIC This can take some time

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC USE CATALOG training;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS data;
# MAGIC use schema data;
# MAGIC CREATE VOLUME IF NOT EXISTS langchain
# MAGIC     COMMENT 'Volume for langchain example data';

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf ./snorre_vector_db
# MAGIC mkdir -p ./snorre_vector_db

# COMMAND ----------

# if 'vectordb_persisted' in globals(): # If you've already made your vectordb this will delete it so you start fresh
#     vectordb_persisted.delete_collection()

# Didnt get it to write straight to dfgs. Chroma gives IO Error or stores dbfs as a dir name
# persist_path = "dbfs:/Volumes/training/data/langchain/test_vector_db/"
# Cannot write to /tmp either, Chroma DB gives (cannot write to read only db error)
persist_path = "./snorre_vector_db"
# embedding = OpenAIEmbeddings()
vectordb_persisted = Chroma.from_documents(documents=splits, 
                                 embedding=embedding_model,
                                 persist_directory=persist_path)
vectordb_persisted.persist()
# vectordb_persisted = None
vectordb = Chroma(persist_directory=persist_path,
                    embedding_function=embedding_model)

# COMMAND ----------

# MAGIC %sh
# MAGIC # Clean old dir
# MAGIC mkdir -p /Volumes/training/data/langchain
# MAGIC rm -rf /Volumes/training/data/langchain/snorre_vector_db
# MAGIC # Move files to volume, to avoid adding to git
# MAGIC cp -r ./snorre_vector_db /Volumes/training/data/langchain/snorre_vector_db

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Test MultiQuery to evaluate vectordb
# MAGIC This retrieval method will generated 3 additional questions to get a total of 4 queries (with the users included) that will be used to go retrieve documents. This is helpful when you want to retrieve documents which are similar in meaning to your question.
# MAGIC

# COMMAND ----------

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
# Set logging for the queries
import logging

# COMMAND ----------

# MAGIC %md
# MAGIC Doing some logging to see the other questions that were generated. I tried to find a way to get these via a model property but couldn't, lmk if you find a way!

# COMMAND ----------

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# COMMAND ----------

# MAGIC %md
# MAGIC Then we set up the MultiQueryRetriever which will generate other questions for us

# COMMAND ----------

question = "How did Harald Hardrade help the Icelanders?"
# question = "Which year did Harald Hardrade leave Norway?"
#question = "Who was the longest reigning king in Norway, according to Snorre? How long did he reign? How did he die? Who was his father?"
# llm = ChatOpenAI(temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)

# COMMAND ----------

# MAGIC %md
# MAGIC Ok now let's put those docs into a prompt template which we'll use as context

# COMMAND ----------

prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# COMMAND ----------

llm.predict(text=PROMPT.format_prompt(
    context=unique_docs,
    question=question
).text)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Another question with multi-retrieval

# COMMAND ----------

question = "Who was the wife of Hardrade?"
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
ret = llm.predict(text=PROMPT.format_prompt(
    context=unique_docs,
    question=question
).text)
ret

# COMMAND ----------

# MAGIC %md ## Clean up git repo from vector files

# COMMAND ----------

# MAGIC %sh
# MAGIC # rm -rf ./snorre_vector_db

# COMMAND ----------


