/*
@author    :     rscalia
@date      :     Wed 04/11/2020

Questa classe serve per addestrare un Modello LFM usando l'algoritmo ALS.

In tale procedura, è prevista anche la fase di Tuning degli Iperparametri.
*/

package lib.ALSTrainer

import scala.math._
import scala.sys._
import scala.io._

import scala.collection.mutable._

import scala.util.matching._

import scala.collection.parallel._
import scala.concurrent._

import scala.reflect._

import org.apache.spark.sql._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, CrossValidatorModel}


class ALSTrainer (val _iterations:Array[Int], val _regularization:Array[Double], val _userCol:String, 
                  val _itemCol:String      , val _ratingCol:String 
                 ) {

    var _model:Option[CrossValidatorModel] = None


    def train (pDataset:Dataset[Row]) = {
        val model     = new ALS ()

        model
        .setUserCol(_userCol)
        .setItemCol(_itemCol)
        .setRatingCol(_ratingCol)
        .setColdStartStrategy("drop")


        val paramGrid = new ParamGridBuilder()
                            .addGrid(model.maxIter, _iterations)
                            .addGrid(model.regParam, _regularization)
                            .build()

        val ev        = new RegressionEvaluator ()
        ev.setMetricName("rmse")
 

        val cv        = new CrossValidator()
                            .setCollectSubModels(true)
                            .setEstimator(model)
                            .setEvaluator(ev)
                            .setEstimatorParamMaps(paramGrid)
                            .setNumFolds(2)  
                            .setParallelism(8) 
                           


        _model        = Some( cv.fit(pDataset) )
    }


}
