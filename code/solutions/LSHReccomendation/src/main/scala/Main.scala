/*
@author    :     rscalia
@date      :     Sun 01/11/2020

Questa classe serve per testare la soluzione di raccomandazione basata su LSH.
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
import lib.LSH._
import lib.LSHEvaluator._

object Main {

  def main(args: Array[String]): Unit = {

    //Creazione Spark Session
    val spark                     = SparkSession.builder.appName("LSHReccomendation").getOrCreate()
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



    //Fitting LSH Model
    val N_USER                  = 6040
    val DIST_THR                = 40
    val BUCKET_LENGTH:Double    = 2.0
    val N_HASH_TABLES           = 3

    val lsh = new LSH(spark, N_USER, DIST_THR, BUCKET_LENGTH, N_HASH_TABLES)

    lsh.fit(loader._dataLake.get)

    

    //Evaluate and Record LSH Model Performance
    val SPLITS                    = 10
    val SAVE_PATH                 = "hdfs://localhost:9000/cp_output/lsh_rec__bucket_len_"+BUCKET_LENGTH+"__n_hash_tables_"+N_HASH_TABLES+"__dist_thr_"+DIST_THR

    println (s"\n\n-> Dataset Save Path: ${SAVE_PATH} \n\n")
    

    val evaluator                 = new LSHEvaluator(spark, SAVE_PATH, SPLITS)


    //evaluator.evaluate(loader._dataLake.get, lsh._model.get)
    evaluator.parallelEvaluate(loader._dataLake.get, lsh._model.get)
    evaluator.write (BUCKET_LENGTH, N_HASH_TABLES, DIST_THR)
    

    
  }

}