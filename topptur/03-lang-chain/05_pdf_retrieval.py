# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced retrieval from a pdf with langchain on Databricks
# MAGIC
# MAGIC ### Source Snorre Saga as PDF
# MAGIC
# MAGIC You can search Snorre Saga online here: [https://www.gutenberg.org/files/598/598-h/598-h.htm](https://www.gutenberg.org/files/598/598-h/598-h.htm)
# MAGIC
# MAGIC ### Model
# MAGIC We use OpenAI.
# MAGIC
# MAGIC Can be run on a non-gpu cluster like UC Shared Cluster 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ![Rag pipeline](https://raw.githubusercontent.com/paalvibe/llm-langchain-course/main/topptur/00-TEACHER-prep/images/rag_pipeline.png)

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

# MAGIC %pip install -qU \
# MAGIC     langchain==0.1.1 \
# MAGIC     langchain-community==0.0.13 \
# MAGIC     datasets==2.14.6 \
# MAGIC     openai==1.6.1 \
# MAGIC     tiktoken==0.5.2 \
# MAGIC     chromadb==0.4.22 \
# MAGIC     mlflow==2.10.2 \
# MAGIC     unstructured \
# MAGIC     sentence_transformers \
# MAGIC     "unstructured[pdf]"
# MAGIC  
# MAGIC #%pip install -q -U langchain
# MAGIC #%pip install sentence-transformers unstructured chromadb mlflow openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import Databricks
import os
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")
llm = Databricks(host=host, endpoint_name="azure_openai_training", max_tokens=1024)

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
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# COMMAND ----------

# MAGIC %md
# MAGIC # Downloading embedding model from HuggingFace
# MAGIC
# MAGIC I am using the best embedding model on Huggingface’s embedding leaderboard. Feel free to use another.

# COMMAND ----------

# Downloading embedding model 
embedding_model = SentenceTransformerEmbeddings(model_name='BAAI/bge-large-en-v1.5')

# COMMAND ----------

# MAGIC %md ## Load vector DB
# MAGIC
# MAGIC A vector db of a Snorre Saga pdf has been prepared in ../00-Teacher-prep/05_pdf_retrieval_prep
# MAGIC
# MAGIC Embeddings vector DB allows us to interact with large texts.

# COMMAND ----------

# MAGIC %ls ../../data/Sagas/

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

# MAGIC %ls -lh /Volumes/training/data/langchain/snorre_vector_db

# COMMAND ----------

# MAGIC %sh
# MAGIC # Make local copy of test_vector_db, prepped by teacher
# MAGIC # Chroma does not know how to access dbfs
# MAGIC rm -rf ./snorre_vector_db
# MAGIC cp -r /Volumes/training/data/langchain/snorre_vector_db snorre_vector_db
# MAGIC ls -lh ./snorre_vector_db/*

# COMMAND ----------

persist_path = "./snorre_vector_db"
vectordb = Chroma(persist_directory=persist_path,
                    embedding_function=embedding_model)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## MultiQuery
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
# llm = ChatOpenAI(temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)
unique_docs = retriever_from_llm.get_relevant_documents(query=question)

# COMMAND ----------

# MAGIC %md
# MAGIC Check out how there are other questions which are related to but slightly different than the question I asked.
# MAGIC
# MAGIC Let's see how many docs were actually returned

# COMMAND ----------

# Should be more than 0
len(unique_docs)

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

# MAGIC %md
# MAGIC # ChatGPT 3.5 cannot process enough tokens
# MAGIC ```
# MAGIC llm.predict(text=PROMPT.format_prompt(
# MAGIC     context=unique_docs,
# MAGIC     question=question
# MAGIC ).text)
# MAGIC ...
# MAGIC Error code: 400 - {'error': {'message': "This model's maximum context length is 4097 tokens, however you requested 4606 tokens (4350 in your prompt; 256 for the completion). Please reduce your prompt; or completion length.", 'type': 'invalid_request_error', 'param': None, 'code': None}}
# MAGIC
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # Get a more powerful model
# MAGIC
# MAGIC GPT-4 does support completions API, and we were not able to get sensible replies from the context,
# MAGIC so instead we use open source mistral model, served from a GPU VM.
# MAGIC
# MAGIC OpenAI Models: https://platform.openai.com/docs/models/gpt-3-5-turbo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get locally served mistral model

# COMMAND ----------

# Enable live reloading of libs
%load_ext autoreload
%autoreload 2

# COMMAND ----------

from topptur.libs import llmlocal

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Select a server number between 1-4
# MAGIC
# MAGIC We should spread out across GPUs

# COMMAND ----------

server_num = 3
power_llm = llmlocal.llmlocal(server_name="server", server_num=server_num, spark=spark)

# COMMAND ----------

# Test invocation with context
power_llm.predict(text=PROMPT.format_prompt(
    context=unique_docs,
    question=question
).text)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Another question with multi-retrieval

# COMMAND ----------

question = "Where did Gyda, daughter of King Eirik of Hordaland, grow up?"
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
ret = power_llm.predict(text=PROMPT.format_prompt(
    context=unique_docs,
    question=question
).text)
ret

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task: By changing the question, try to get good answers to: 
# MAGIC 1. When Harald Hardrade and his wife Ellisif got married?
# MAGIC 2. When did Olav Tryggvason rule Norway?
# MAGIC 3. What position did Harald hold in constantinople?

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ## NOT READY TODO Contextual Compression
# MAGIC Then we'll move onto contextual compression. This will take the chunk that you've made (above) and compress it's information down to the parts relevant to your query.
# MAGIC
# MAGIC Say that you have a chunk that has 3 topics within it, you only really care about one of them though, this compressor will look at your query, see that you only need one of the 3 topics, then extract & return that one topic.
# MAGIC
# MAGIC This one is a bit more expensive because each doc returned will get processed an additional time (to pull out the relevant data)

# COMMAND ----------

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# COMMAND ----------

# MAGIC %md We first need to set up our compressor, it's cool that it's a separate object because that means you can use it elsewhere outside this retriever as well.

# COMMAND ----------

# llm = ChatOpenAI(temperature=0, model='gpt-4')

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                       base_retriever=vectordb.as_retriever())

# COMMAND ----------

# MAGIC %md
# MAGIC First, an example of compression. Below we have one of our splits that we made above

# COMMAND ----------

splits[0].page_content

# COMMAND ----------

# MAGIC %md
# MAGIC Now we are going to pass a question to it and with that question we will compress the doc. The cool part is this doc will be contextually compressed, meaning the resulting file will only have the information relevant to the question.

# COMMAND ----------

compressor.compress_documents(documents=[splits[0]], query="test for what you like to do")

# COMMAND ----------


