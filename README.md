# **Features selection in spark**
How to use the tools provided by feature selector in spark

# *Feature selector*

## Vector slicer

An important thing to remember is that most models in the ml package from spark use a format based on two columns: one with a label and another with a vector of features (variables):


<img src="input_for_models.png" alt="alt text" width="300" >

Vector slicer is a tool to choose the useful elements of the vector in the features column. It is useful If and only if you already know which are the useful features.

The following example comes from: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

```python
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row

df = spark.createDataFrame([
    Row(userFeatures=Vectors.sparse(3, {0: -2.0, 1: 2.3})),
    Row(userFeatures=Vectors.dense([-2.0, 2.3, 0.0]))])

slicer = VectorSlicer(inputCol="userFeatures", outputCol="features", indices=[1])

output = slicer.transform(df)

output.select("userFeatures", "features").show()
```

## RFormula
Allows us to build the features for a model using an R formula. This makes authomatic binary categories which may not be useful if you know your data (seems very useful to automatize processes and make libraries but not for our current work.

## **ChiSqSelector** (may be useful but I sill have a lot of questions about it)
Selects categorical features to use for predicting a categorical label.
(https://spark.apache.org/docs/2.2.0/api/java/org/apache/spark/ml/feature/ChiSqSelector.html).

**It seems to only work for categorica data!!! O.o Check this!** ¿Es sólo un problema de versiones anteriores?
https://stackoverflow.com/questions/39076943/spark-ml-how-to-find-feature-importance/39081505

There are previous attempts where it crashes when using the output in trees based models: https://q-a-assistant.com/computer-internet-technology/312712_spark-ml-issue-in-training-after-using-chisqselector-for-feature-selection.html 

It supports five selection methods: numTopFeatures, percentile, fpr, fdr, fwe
* **numTopFeatures:** chooses a fixed number of top features according to a chi-squared test. This is akin to yielding the features with the most predictive power. 
* **percentile:** is similar to numTopFeatures but chooses a fraction of all features instead of a fixed number. 
* **fpr** chooses all features whose p-values are below a threshold, thus controlling the false positive rate of selection. 
* **fdr** uses the Benjamini-Hochberg procedure to choose all features whose false discovery rate is below a threshold. 
* **fwe** chooses all features whose p-values are below a threshold. The threshold is scaled by 1/numFeatures, thus controlling the family-wise error rate of selection. 

By default, the selection method is numTopFeatures, with the default number of top features set to 50. The user can choose a selection method using setSelectorType.

```python
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
    (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
    (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["id", "features", "clicked"])

selector = ChiSqSelector(numTopFeatures=1, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="clicked")

result = selector.fit(df).transform(df)

print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
result.show()
```

Si sí hay problemas con datos no categoricos aplicar:


# Other methods:
https://stackoverflow.com/questions/39076943/spark-ml-how-to-find-feature-importance/39081505

## **Information gain based feature selection**
