/*
@author    :     rscalia
@date      :     Fri 06/11/2020

Questa classe serve per scrivere sul file system distribuito un report sul  training del modello LFM basato su l'algoritmo ALS.
*/

package lib.ALSDataWriter

import scala.math._
import scala.sys._
import scala.io._

import scala.collection.mutable._

import scala.util.matching._

import scala.collection.parallel._
import scala.concurrent._

import scala.reflect._

import org.apache.spark.sql._
import org.apache.spark.ml.tuning._
import org.apache.spark.ml.evaluation.RegressionEvaluator


case class ValidationRecord (val NumIterations:Int, val Regularization:Double, val Rank:Int, val RMSE:Double)
case class EvaluationRecord (val NumIterations:Int, val Regularization:Double, val Rank:Int, val RMSE:Double)


class ALSDataWriter (val _spark:SparkSession) {


    import _spark.implicits._

    var _validationDataPath:Option[String]                              = None
    var _evaluationDataPath:Option[String]                              = None

    var _validationData:Option[Dataset[ValidationRecord]]               = None
    var _evaluationData:Option[Dataset[EvaluationRecord]]               = None


    //Filling Object
    def fillObject (pValidationPath:String, pEvaluationPath:String) =  {
        _validationDataPath             = Some(pValidationPath)
        _evaluationDataPath             = Some(pEvaluationPath)
    }

    
    def writeData (pModelValidator: TrainValidationSplitModel, pTestSet:Dataset[Row], pValidation:Dataset[Row]) = {

        //Writing Validation Info
        var validation_tmp_data         = Seq.empty[ValidationRecord].toDS()
        val models                      = pModelValidator.subModels
        

            
        models.foreach ( (m) => {

            val pairs               = m.parent.extractParamMap.toSeq

            
            val maxIterBuffer       = pairs.filter ( (pair) => pair.param.name == "maxIter").map ((x) => x.value)
            val maxIter             = maxIterBuffer(0).asInstanceOf[Int]

            
            val regParamBuffer      = pairs.filter ( (pair) => pair.param.name == "regParam").map ((x) => x.value)
            val regParam            = regParamBuffer(0).asInstanceOf[Double]

            val rankBuffer          = pairs.filter ( (pair) => pair.param.name == "rank").map ((x) => x.value)
            val rank                = rankBuffer(0).asInstanceOf[Int]

        
            val res                 = m.transform (pValidation)

            val evaluator           = new RegressionEvaluator()
                                            .setMetricName("rmse")
                                            .setLabelCol("label")
                                            .setPredictionCol("prediction")

            val rmse                = evaluator.evaluate(res)


            val new_record          = Seq(ValidationRecord(maxIter, regParam, rank, rmse)).toDS()
            validation_tmp_data     = validation_tmp_data.union(new_record)
            
        } )
        
        
        _validationData             =  Some(validation_tmp_data)    
        _validationData.get.write.format("csv").option("header",true).save(_validationDataPath.get)



        //Writing Test Info
        val bestModel           = pModelValidator.bestModel
        val pairs               = bestModel.parent.extractParamMap.toSeq

            
        val maxIterBuffer       = pairs.filter ( (pair) => pair.param.name == "maxIter").map ((x) => x.value)
        val maxIter             = maxIterBuffer(0).asInstanceOf[Int]

        val regParamBuffer      = pairs.filter ( (pair) => pair.param.name == "regParam").map ((x) => x.value)
        val regParam            = regParamBuffer(0).asInstanceOf[Double]

        val rankBuffer          = pairs.filter ( (pair) => pair.param.name == "rank").map ((x) => x.value)
        val rank                = rankBuffer(0).asInstanceOf[Int]

    
        val res                 = bestModel.transform (pTestSet)

        val evaluator           = new RegressionEvaluator()
                                                            .setMetricName("rmse")
                                                            .setLabelCol("label")
                                                            .setPredictionCol("prediction")

        val rmse                = evaluator.evaluate(res)


        val new_record          = Seq(EvaluationRecord(maxIter, regParam, rank, rmse)).toDS()
        _evaluationData         = Some(new_record)

        _evaluationData.get.write.format("csv").option("header",true).save(_evaluationDataPath.get)
        

    }


        
} 