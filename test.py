#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
data = spark.read.csv('poverty.csv', header=True, inferSchema=True)
data = data['Location', 'PovPct', 'Brth15to17', 'Brth18to19', 'ViolCrime', 'TeenBrth']

print("Data: ")
data.show()

feature_columns = ["PovPct"]

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")

data_2 = assembler.transform(data)
print("Data & Feature Vectors: ")
data_2.show()

train, test = data_2.randomSplit([0.7, 0.3])

from pyspark.ml.regression import LinearRegression
algo = LinearRegression(featuresCol="features", labelCol="TeenBrth")
model = algo.fit(train)

model.evaluate(test)
evaluation_summary = model.evaluate(test)
print("MeanAbsolute: ", evaluation_summary.meanAbsoluteError)
print("RootMeanSqr: ",evaluation_summary.rootMeanSquaredError)
print("RSquared: ",evaluation_summary.r2)

print("PREDICTIONS: ")
predictions = model.transform(test)
predictions.select(['PovPct', 'TeenBrth', 'prediction']).show()

