/*
@author    :     rscalia
@date      :     Sat 07/11/2020

Questo componente rappresenta il driver per la soluzione SVDReccomendation.
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
import lib.SVDDataWrangler._
import lib.SVD._
import lib.SVDDataWriter._


object Main {

  def main(args: Array[String]): Unit = {

    //Creazione Spark Session
    val spark = SparkSession.builder.appName("SVDReccomendation").getOrCreate()
    import spark.implicits._


    //Caricamento dati
    val DATA_PATH                 = "hdfs://localhost:9000/dataset/ml-1m-csv/ratings.csv"

    val loader                    = new MovieLensLoader (DATA_PATH, spark)
    loader.printDataPath
    loader.loadData

    val result                    = loader.checkDataIntegrity

    result match {
      case false => {
                        println ("[!] Your data is corrupted \n")
                        return
                    }
            
      case true =>  {
                      println ("[!] Your data is Correct \n")
                    }   
    }
    
    loader._dataLake.get.printSchema()


    //Wrangling Dati
    val N_FILM = 3952

    val wrg = new SVDDataWrangler(spark, N_FILM)

    wrg.wrangle(loader._dataLake.get)
    wrg.split
    wrg.printWranglingSpecs


    //Training e Validation
    val LATENT_FACTORS  = Array(3,5,10,20)
    val svdTr           = new SVDTrainer ()
    val eval            = new SVDEvaluator(spark, N_FILM)

    svdTr.fit(wrg._trainingSet.get, wrg._validationSet.get, LATENT_FACTORS, eval)


    //Writing dati sul file system distribuito
    val VALIDATION_ANALYSIS_PATH  = "hdfs://localhost:9000/cp_output/svd_rec_val"
    val EVALUATION_ANALYSIS_PATH  = "hdfs://localhost:9000/cp_output/svd_rec_eval"
    val writer        = new SVDDataWriter (spark, N_FILM, VALIDATION_ANALYSIS_PATH, EVALUATION_ANALYSIS_PATH)
    
    writer.write(svdTr._model.get,wrg._testSet.get,svdTr._validationInfo)


  }

}