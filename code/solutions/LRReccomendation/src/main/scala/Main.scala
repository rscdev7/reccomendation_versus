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
    val N_USER                  = 6040
    val K_NN                    = 2
    val SAVE_PATH               = "hdfs://localhost:9000/dataset/ml-1m-csv/lin_regr_dataset__nn_" + K_NN.toString +"/"
    val BUCKET_LENGTH:Double    = 2.0
    val N_HASH_TABLES           = 3
    val N_SPLITS                = 10000

    println (s"\n\n-> Dataset Save Path: ${SAVE_PATH} \n\n")


    val wrg = new LRDataWrangler(spark, N_USER, K_NN, SAVE_PATH, BUCKET_LENGTH, N_HASH_TABLES, N_SPLITS)

    wrg.wrangle(loader._dataLake.get)

    
  }

}