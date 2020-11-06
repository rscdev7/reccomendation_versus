import scala.math._
import scala.sys._
import scala.io._

import scala.collection.mutable._

import scala.util.matching._

import scala.collection.parallel._
import scala.concurrent._

import scala.reflect._

import org.apache.spark.sql._


import org.apache.spark.ml.evaluation.RegressionEvaluator

import lib.MovieLensLoader._
import lib.ALSWrangler._
import lib.ALSTrainer._

object Main {

  def main(args: Array[String]): Unit = {

    //Creazione Spark Session
    val spark = SparkSession.builder.appName("Spark App Name").getOrCreate()
    import spark.implicits._


    //Test classe MovieLensLoader
    val DATA_PATH = "hdfs://localhost:9000/dataset/ml-1m-csv/ratings.csv"

    val loader = new MovieLensLoader (DATA_PATH, spark)

    loader.printData
    loader.loadData
    loader.checkDataIntegrity

    loader._dataLake.get.printSchema()


    //Test classe ALSWrangler
    val wrg = new ALSWrangler ();

    wrg.wrangle(loader._dataLake.get)
    wrg.printDataSplitsProperties


    //Test classe ALSTrainer
    val als = new ALSTrainer(Array(5,10,20), Array(0.99,0.5,0.2,0.3), "user_id", "movie_id","label")

    als.train( wrg._trainingSet.get )



    val best = als._model.get.bestModel

    val res  = best.transform (wrg._testSet.get)

    val evaluator = new RegressionEvaluator()
                        .setMetricName("rmse")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")

    

    val rmse = evaluator.evaluate(res)
    println(s"Root-mean-square error = $rmse")


    
  }

}