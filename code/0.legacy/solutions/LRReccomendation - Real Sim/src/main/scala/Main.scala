/*
@author    :     rscalia
@date      :     Sun 01/11/2020

Questa classe serve per testare la soluzione di raccomandazione basata su Regressione Lineare.
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
import lib.LRDataWrangler._


object Main {

  def main(args: Array[String]): Unit = {

    //Creazione Spark Session
    val spark = SparkSession.builder.appName("LRReccomendation").getOrCreate()
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
    val N_FILM                  = 3952
    val K_NN:Long               = 150
    val SAVE_PATH               = "hdfs://localhost:9000/dataset/ml-1m-csv/lin_regr_dataset__nn_" + K_NN.toString +"/"
    val N_SPLITS                = 10

    val wrg = new LRDataWrangler(spark, N_FILM, K_NN, SAVE_PATH, N_SPLITS)


    wrg.wrangle(loader._dataLake.get)

    
  }

}