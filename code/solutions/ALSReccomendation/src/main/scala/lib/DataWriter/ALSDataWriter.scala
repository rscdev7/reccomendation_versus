/*
@author    :     rscalia
@date      :     Fri 06/11/2020

Questa classe serve per scrivere un log dei risultati del training sul file system distribuito.
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
import org.apache.spark.ml.tuning.{CrossValidatorModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator


case class ValidationRecord (val NumIteration:Int, val Regularization:Double, val Fold:Int, val RMSE:Double)
case class TestRecord (val NumIteration:Int, val Regularization:Double, val RMSE:Double)


class ALSDataWriter (val _spark:SparkSession) {


    import _spark.implicits._

    var _validationDataPath:Option[String]                          = None
    var _testDataPath:Option[String]                                = None

    var _validationData:Option[Dataset[ValidationRecord]]           = None
    var _testData:Option[Dataset[TestRecord]]                       = None


    //Filling Object
    def fillObject (pValidationPath:String, pTestPath:String) =  {
        _validationDataPath     = Some(pValidationPath)
        _testDataPath           = Some(pTestPath)
    }

    
    def writeData (_modelValidator: CrossValidatorModel, _testSet:Dataset[Row]) = {

        //Writing Validation Info
        var testData    = Seq.empty[ValidationRecord].toDF()
        val models      = _modelValidator.subModels
        

        for ( fold <- List.range(0,models.length) ) {
            
            models(fold).map ( (m) => {
                val pairs       = m.parent.extractParamMap.toSeq

            
                val maxIterBuffer     = pairs.filter ( (pair) => pair.param.name == "maxIter").map ((x) => x.value)
                val maxIter           = maxIterBuffer(0).asInstanceOf[Int]

                

                val regParamBuffer     = pairs.filter ( (pair) => pair.param.name == "regParam").map ((x) => x.value)
                val regParam           = regParamBuffer(0).asInstanceOf[Double]

            
                val res                 = m.transform (_testSet)

                val evaluator           = new RegressionEvaluator()
                                                .setMetricName("rmse")
                                                .setLabelCol("label")
                                                .setPredictionCol("prediction")

                val rmse                = evaluator.evaluate(res)


                val new_record          = Seq(ValidationRecord(maxIter, regParam, fold, rmse)).toDF()
                testData = testData.union(new_record)
                
            } )
        }

        testData.write.format("csv").option("header",true).save(_validationDataPath.get)


        //Writing Test Info
        val bestModel           = _modelValidator.bestModel
        val pairs               = bestModel.parent.extractParamMap.toSeq

            
        val maxIterBuffer       = pairs.filter ( (pair) => pair.param.name == "maxIter").map ((x) => x.value)
        val maxIter             = maxIterBuffer(0).asInstanceOf[Int]

        val regParamBuffer      = pairs.filter ( (pair) => pair.param.name == "regParam").map ((x) => x.value)
        val regParam            = regParamBuffer(0).asInstanceOf[Double]

    
        val res                 = bestModel.transform (_testSet)

        val evaluator           = new RegressionEvaluator()
                                                            .setMetricName("rmse")
                                                            .setLabelCol("label")
                                                            .setPredictionCol("prediction")

        val rmse                = evaluator.evaluate(res)


        val new_record          = Seq(TestRecord(maxIter, regParam, rmse)).toDF()
        new_record.write.format("csv").option("header",true).save(_testDataPath.get)
        

    }


        
} 