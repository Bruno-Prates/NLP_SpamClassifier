# Databricks notebook source
# MAGIC %md
# MAGIC Esse projeto foi elaborado durante o curso disponível pelo seguinte link:
# MAGIC https://www.udemy.com/course/spark-and-python-for-big-data-with-pyspark/
# MAGIC 
# MAGIC Utilizamos o seguinte dataset - Repository SMS Spam Detection: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# MAGIC 
# MAGIC Futuramente pretendo dar uma incrementada nesse projeto. Por enquanto, o último semestre da graduação em Física está me tomando bastante tempo. Assim que eu conseguir, vou revisitar este código.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #1. Importando dados

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('SpamClassifier').getOrCreate()

# COMMAND ----------

df = spark.read.table("smsspamcollection")
df.show(n=2)

# COMMAND ----------

df = df.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
df.show(n=2)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Tratamento dos Dados

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Criando uma feature chamada length

# COMMAND ----------

from pyspark.sql.functions import length
df = df.withColumn('length',length(df['text']))

# COMMAND ----------

df.show(n=2)

# COMMAND ----------

df.groupby('class').mean().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##2.2  Transformação de Features

# COMMAND ----------

from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

# COMMAND ----------

clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3 Construindo modelo
# MAGIC 
# MAGIC Futuramente pretendo incrementar essa etapa para deixar ele mais preciso.

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Pipeline

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
cleaner = data_prep_pipe.fit(df)
clean_data = cleaner.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.2 Treinando modelo

# COMMAND ----------

clean_data = clean_data.select(['label','features'])
clean_data.show(n=5)

# COMMAND ----------

(training,testing) = clean_data.randomSplit([0.7,0.3])
spam_predictor = nb.fit(training)

# COMMAND ----------

test_results = spam_predictor.transform(testing)
test_results.show(n=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.3 Evaluation

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Acurácia do modelo: {}".format(acc))

# COMMAND ----------


