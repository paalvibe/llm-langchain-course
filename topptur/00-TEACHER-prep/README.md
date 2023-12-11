# LLM Langchain teacher prep

Run through notebooks 00-06. You might not need all the 02 notebooks.

* The 02 notebooks will setup proxy servers. You need approximately one server per 4 students. Each server needs its own GPU.
Only start those you need.
* Notice that there is sometimes query mixup between the clients. This possibly happens inside the mistral serving.
* In the future, the model serving could be rewritten to use Databricks Model serving instead.
  * One advantage with the current setup is that the code is more portable.
* The shared coded between the 02 (and 06) notebooks could be refactored for DRYness.