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


object Main {

  def main(args: Array[String]): Unit = {

    //Creazione Spark Session
    val spark                     = SparkSession.builder.appName("ALSReccomendation").getOrCreate()
    import spark.implicits._



    //Test Classe MovieLensLoader
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

  }

}