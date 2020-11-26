/*
@author    :     rscalia
@date      :     Wed 04/11/2020

Questa classe serve per addestrare un Modello LFM usando l'algoritmo ALS.

In tale procedura, è prevista anche la fase di Tuning degli Iperparametri (Validation).
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
import org.apache.spark.ml.tuning._


class ALSTrainer (val _iterations:Array[Int]    ,  val _regularization:Array[Double] , 
                  val _rank:Array[Int]          ,  val _seed:Long                   , 
                  val _userCol:String           ,  val _itemCol:String        , 
                  val _ratingCol:String 
                 ) {


    var _model:Option[TrainValidationSplitModel] = None


    def train (pDataset:Dataset[Row]) = {
        val model     = new ALS ()

        model
        .setUserCol(_userCol)
        .setItemCol(_itemCol)
        .setRatingCol(_ratingCol)
        .setSeed(_seed)
        .setColdStartStrategy("drop")


        val paramGrid = new ParamGridBuilder()
                            .addGrid ( model.maxIter  , _iterations     )
                            .addGrid ( model.regParam , _regularization )
                            .addGrid ( model.rank     , _rank           )
                            .build()

        val ev        = new RegressionEvaluator ()
        ev.setMetricName("rmse")
 

        val eng        = new TrainValidationSplit()
                            .setCollectSubModels(true)
                            .setEstimator(model)
                            .setEvaluator(ev)
                            .setEstimatorParamMaps(paramGrid)
                            .setParallelism(8) 
                            .setSeed(_seed)
                            .setTrainRatio(0.80)
                           


        _model        = Some( eng.fit(pDataset) )
    }


}
