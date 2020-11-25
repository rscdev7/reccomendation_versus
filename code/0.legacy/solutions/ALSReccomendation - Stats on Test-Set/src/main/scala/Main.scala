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
    val spark                     = SparkSession.builder.appName("ALSReccomendation").getOrCreate()
    import spark.implicits._



    //Caricamento dataset "in memoria"
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


    //Wrangling dei Dati
    val wrg                       = new ALSWrangler ();
    wrg.wrangle(loader._dataLake.get)
    wrg.printDataSplitsProperties

    
    //Training Modello LFM sfruttando l'algoritmo ALS
    val SEED:Long                 = 12345678
    val MAX_ITER                  = Array (5,10,20)
    val REG_PARAM                 = Array (0.99,0.90,0.80)
    val RANK                      = Array (5,10,20)
    val FOLD                      = 10
    
    val als                       = new ALSTrainer(MAX_ITER, REG_PARAM, RANK, FOLD, SEED ,"user_id", "movie_id","label")
    als.train( wrg._trainingSet.get )


    //Writing del risultato dell'esperimento sul file system ditribuito
    val VALIDATION_ANALYSIS_PATH  = "hdfs://localhost:9000/cp_output/als_rec_val"
    val EVALUATION_ANALYSIS_PATH  = "hdfs://localhost:9000/cp_output/als_rec_eval"

    val wr                        = new ALSDataWriter (spark)
    wr.fillObject(VALIDATION_ANALYSIS_PATH , EVALUATION_ANALYSIS_PATH)
    wr.writeData(als._model.get, wrg._testSet.get)

  }

}