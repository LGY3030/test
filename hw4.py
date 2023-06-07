#!/usr/bin/env python
# coding: utf-8

# Q1-2

# In[ ]:


from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    conf = SparkConf().setAppName("Spark Count")
    sc = SparkContext(conf=conf)
    
    input_file = sc.textFile("input.txt")
    map = input_file.flatMap (lambda line:\
            line.split(" ")).map(lambda word:(word,1))
    counts= map.reduceByKey(lambda a,b: a+b)
    counts.saveAsTextFile("output")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




