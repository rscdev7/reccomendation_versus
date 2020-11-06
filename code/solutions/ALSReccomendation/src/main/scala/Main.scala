/*
@author    :     rscalia
@date      :     Fri 06/11/2020

Questo componente rappresenta il driver per la soluzione ALSReccomendation.
*/

import scala.math._
import scala.sys._
import scala.io._

import scala.collection.mutable._

import scala.util.matching._

import scala.collection.parallel._
import scala.concurrent._

import scala.reflect._

import org.apache.spark.sql._


import lib.MovieLensLoader._
import lib.ALSWrangler._
import lib.ALSTrainer._
import lib.ALSDataWriter._


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
    val als = new ALSTrainer(Array(5,10), Array(0.99), "user_id", "movie_id","label")
    als.train( wrg._trainingSet.get )


    //Test Classe ALSDataWriter
    val wr                        = new ALSDataWriter (spark)
    val VALIDATION_ANALYSIS_PATH  = "hdfs://localhost:9000/cp_output/als_rec_val"
    val TRAINING_ANALYSIS_PATH    = "hdfs://localhost:9000/cp_output/als_rec_test"

    wr.fillObject(VALIDATION_ANALYSIS_PATH , TRAINING_ANALYSIS_PATH)
    wr.writeData(als._model.get, wrg._testSet.get)

  }

}