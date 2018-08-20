package com.intel.analytics.zoo.examples.sagemaker

import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.amazonaws.services.sagemaker.AmazonSageMakerClientBuilder
import com.amazonaws.services.sagemaker.sparksdk.{IAMRole, SageMakerEstimator}
import com.amazonaws.services.sagemaker.sparksdk.algorithms.{KMeansSageMakerEstimator, PCASageMakerEstimator}
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.KMeansProtobufResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer
import com.amazonaws.services.securitytoken.AWSSecurityTokenServiceClientBuilder
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, Module}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.pipeline.nnframes.{NNEstimator, NNImageReader, NNModel}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.SparkSession

import scala.util.Random

object SageMakerExample {

  def main(args: Array[String]): Unit = {
    example4()
//    val conf = Engine.createSparkConf().setAppName("Test NNEstimator").setMaster("local[2]")
//    val sc = NNContext.initNNContext(conf)
//    val sqlContext = new SQLContext(sc)
//
//    val nRecords = 100
//    val smallData = generateTestInput(
//      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), -1.0, 42L)
//
//
//    val data = sc.parallelize(smallData)
//    val df = sqlContext.createDataFrame(data).toDF("features", "label")
//
//    val model = Linear[Float](8, 4)
//    val criterion = ClassNLLCriterion[Float]()
//    val lrEstimator = NNEstimator(model, criterion, Array(8), Array(1))
//
//    val roleArn = "arn:aws:iam::account-id:role/rolename"
//    val pca = new PCASageMakerEstimator(
//      sagemakerRole = IAMRole(roleArn),
//      requestRowSerializer =
//        new ProtobufRequestRowSerializer(featuresColumnName = "projectedFeatures"),
//      trainingSparkDataFormatOptions = Map("featuresColumnName" -> "projectedFeatures"),
//      trainingInstanceType = "ml.p2.xlarge",
//      trainingInstanceCount = 1,
//      endpointInstanceType = "ml.c4.xlarge",
//      endpointInitialInstanceCount = 1)
//      .setK(10).setFeatureDim(50)
  }

  def example1(): Unit = {
    val conf = Engine.createSparkConf().setAppName("Test NNEstimator")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = new SQLContext(sc)

    val nRecords = 100
    val smallData = generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), -1.0, 42L)


    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val model = Linear[Float](8, 4)
    val criterion = ClassNLLCriterion[Float]()
    val lrEstimator = NNEstimator(model, criterion, Array(8), Array(1))

    val roleArn = "arn:aws:iam::account-id:role/rolename"
    val kmeans = new KMeansSageMakerEstimator(
      sagemakerRole = IAMRole(roleArn),
      requestRowSerializer =
        new ProtobufRequestRowSerializer(featuresColumnName = "projectedFeatures"),
      trainingSparkDataFormatOptions = Map("featuresColumnName" -> "projectedFeatures"),
      trainingInstanceType = "ml.p2.xlarge",
      trainingInstanceCount = 1,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1)
      .setK(10).setFeatureDim(50)
  }

  def example2(): Unit = {
    val conf = Engine.createSparkConf().setAppName("Test NNEstimator")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = new SQLContext(sc)

    val createLabel = udf { row: Row =>
      if (row.getString(0).contains("demo/cats")) 1.0 else 2.0
    }
    val imagesDF: DataFrame = NNImageReader.readImages("/data/zoo/small" + "/*/*", sc)
      .withColumn("label", createLabel(col("image")))
    val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.1, 0.9), seed = 42L)

    val transformer = RowToImageFeature() -> ImageResize(256, 256) -> ImageCenterCrop(224, 224) ->
      ImageChannelNormalize(123, 117, 104) -> ImageMatToTensor() -> ImageFeatureToTensor()

    val caffeDefPath = "/home/jwang/model/googlenet_caffe/deploy.prototxt"
    val modelPath = "/home/jwang/model/googlenet_caffe/bvlc_googlenet.caffemodel"
    val loadedModel = Module.loadCaffeModel[Float](caffeDefPath, modelPath)
    val batchSize = 4
    val featurizer = NNModel(loadedModel, transformer)
      .setBatchSize(batchSize)
      .setFeaturesCol("image")
      .setPredictionCol("embedding")

    println("finish create nnmodels")

    val kMeansSageMakerEstimator = new KMeansSageMakerEstimator(
      sagemakerRole = IAMRole(roleArn),
      requestRowSerializer =
        new ProtobufRequestRowSerializer(featuresColumnName = "embedding"),
      trainingSparkDataFormatOptions = Map("featuresColumnName" -> "embedding"),
      trainingInstanceType = "ml.p2.xlarge",
      trainingInstanceCount = 1,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1)
      .setK(10).setFeatureDim(1000)

    val pipeline = new Pipeline().setStages(Array(featurizer, kMeansSageMakerEstimator))

    // train
    val pipelineModel = pipeline.fit(trainingDF)

    val transformedData = pipelineModel.transform(validationDF)
    transformedData.show()

  }

  def example3(): Unit = {
    val spark = SparkSession.builder.getOrCreate
    // load mnist data as a dataframe from libsvm. replace this region with your own.
    val region = "us-east-1"
    val trainingData = spark.read.format("libsvm")
      .option("numFeatures", "784")
      .load(s"s3a://sagemaker-sample-data-$region/spark/mnist/train/")

    val testData = spark.read.format("libsvm")
      .option("numFeatures", "784")
      .load(s"s3a://sagemaker-sample-data-$region/spark/mnist/test/")

    // Replace this IAM Role ARN with your own.

    val estimator = new KMeansSageMakerEstimator(
      sagemakerRole = IAMRole(roleArn),
      trainingInstanceType = "ml.p2.xlarge",
      trainingInstanceCount = 1,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1)
      .setK(10).setFeatureDim(784)

    val model = estimator.fit(trainingData)
    val transformedData = model.transform(testData)
    transformedData.show
  }

  def example4(): Unit = {
    val conf = Engine.createSparkConf().setAppName("Test NNEstimator")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = new SQLContext(sc)

    val nRecords = 1000
    val smallData = generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), -1.0, 42L)


    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "class")
    df.show()

    val estimator = new SageMakerEstimator(
      trainingImage =
        "174872318107.dkr.ecr.us-west-2.amazonaws.com/kmeans:1",
      modelImage =
        "174872318107.dkr.ecr.us-west-2.amazonaws.com/kmeans:1",
      requestRowSerializer = new ProtobufRequestRowSerializer(),
      responseRowDeserializer = new KMeansProtobufResponseRowDeserializer(),
      hyperParameters = Map("k" -> "5", "feature_dim" -> "8"),
      sagemakerRole = IAMRole(roleArn),
      trainingInstanceType = "ml.p2.xlarge",
      trainingInstanceCount = 1,
      endpointInstanceType = "ml.c4.xlarge",
      endpointInitialInstanceCount = 1,
      trainingSparkDataFormat = "sagemaker",
      sagemakerClient = AmazonSageMakerClientBuilder.standard().withRegion("us-west-2").build(),
      s3Client = AmazonS3ClientBuilder.standard().withRegion("us-west-2").build(),
      stsClient = AWSSecurityTokenServiceClientBuilder.standard().withRegion("us-west-2").build())
//    val kmeans = new KMeansSageMakerEstimator(
//      sagemakerRole = IAMRole(roleArn),
//      sagemakerClient = AmazonSageMakerClientBuilder.standard().withRegion("us-west-2").build(),
//      s3Client = AmazonS3ClientBuilder.standard().withRegion("us-west-2").build(),
//      stsClient = AWSSecurityTokenServiceClientBuilder.standard().withRegion("us-west-2").build(),
//      requestRowSerializer =
//        new ProtobufRequestRowSerializer(featuresColumnName = "features"),
//      trainingSparkDataFormatOptions = Map("featuresColumnName" -> "features"),
//      trainingInstanceType = "ml.p2.xlarge",
//      trainingInstanceCount = 1,
//      endpointInstanceType = "ml.c4.xlarge",
//      endpointInitialInstanceCount = 1)
//      .setK(5).setFeatureDim(8)
    val model = estimator.fit(df)
    val transformedData = model.transform(df)
    transformedData.show
  }

  def generateTestInput(
                         numRecords: Int,
                         weight: Array[Double],
                         intercept: Double,
                         seed: Long): Seq[(Array[Double], Double)] = {
    val rnd = new Random(seed)
    val data = (1 to numRecords)
      .map( i => Array.tabulate(weight.length)(index => rnd.nextDouble() * 2 - 1))
      .map { record =>
        val y = record.zip(weight).map(t => t._1 * t._2).sum
        +intercept + 0.01 * rnd.nextGaussian()
        val label = if (y > 0) 2.0 else 1.0
        (record, label)
      }
    data
  }

  def generateTestInput2(
                         numRecords: Int,
                         weight: Array[Double],
                         intercept: Double,
                         seed: Long): Seq[(Array[Double])] = {
    val rnd = new Random(seed)
    val data = (1 to numRecords)
      .map { i => (Array.tabulate(weight.length)(index => rnd.nextDouble() * 2 - 1))
      }
    data
  }
}
