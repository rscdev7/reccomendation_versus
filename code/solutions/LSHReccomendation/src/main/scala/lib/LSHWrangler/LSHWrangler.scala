/*
@author    :     rscalia
@date      :     Wed 04/11/2020

Questa classe serve per effettuare il Wrangling dei Dati necessario all'algoritmo LSHReccomendation.
*/


package lib.LSHWrangler

import scala.math._
import scala.sys._
import scala.io._

import scala.collection.mutable._

import scala.util.matching._

import scala.collection.parallel._
import scala.concurrent._

import scala.reflect._

import org.apache.spark.sql._

class LSHWrangler  {

    var _trainingSet:Option[Dataset[Row]]           = None
    var _testSet:Option[Dataset[Row]]               = None
    var _originalDatasetLen:Long                    = -1
    

    def wrangle (pDataLake:Dataset[Row]): Unit = {
        val trainTest:Array[Dataset[Row]] = pDataLake.randomSplit(Array(0.8, 0.2), seed=2014554)

        _trainingSet                      = Some(trainTest(0))
        _testSet                          = Some(trainTest(1))

        _originalDatasetLen               = pDataLake.count
    }

    def printDataSplitsProperties = {
        println (s"""\n***************** DATASET SPLITS ************************ \n
        |-> Original Dataset Len: ${_originalDatasetLen} \n 
        |-> Training-Set Len: ${_trainingSet.get.count} \n 
        |-> Test-Set Len: ${_testSet.get.count} 
        |\n******************************************************** """.stripMargin)
    }
}