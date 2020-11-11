/*
@author    :     rscalia
@date      :     Sun 01/11/2020

Questa classe serve per effettuare il wrangling dei dati necessario al calcolo della Singular Value Decomposition.
*/

package lib.SVDDataWrangler

import scala.reflect._
import scala.collection._

import org.apache.spark.sql._
import org.apache.spark.rdd._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix



class SVDDataWrangler (val _spark:SparkSession , val _nFilm:Int) extends Serializable{

    import _spark.implicits._

    var _wrangledSet:Option[RDD[Vector]]       = None

    var _trainingSet:Option[RDD[Vector]]       = None
    var _validationSet:Option[RDD[Vector]]     = None
    var _testSet:Option[RDD[Vector]]           = None


    def wrangle (pDataset:Dataset[Row]) = {

        
        val augmented_dataset = pDataset.withColumn ( "film_&_rating", struct ( pDataset("movie_id") - 1 , pDataset("label")  ) )


        
        val film_per_user = augmented_dataset.groupBy (pDataset.col("user_id")).agg(collect_list( "film_&_rating" ))

    
        
        val svdReadyDataset = film_per_user.rdd.map ( (record) => {
                                                val pairsList = record.getSeq[GenericRowWithSchema](1)
                                                
                                                
                                                val castedPairsList = pairsList.map ( (x) => Tuple2( x.asInstanceOf[Row].getInt(0), x.asInstanceOf[Row].getInt(1).toDouble ))

                                                val v:Vector = Vectors.sparse(_nFilm, castedPairsList)

                                                v
                                            })

        _wrangledSet = Some(svdReadyDataset)

    }

    def split = {
        val trainTestSplit                 = _wrangledSet.get.randomSplit (Array(0.8,0.2) , seed=2014554)
        val trainValidationSplit           = trainTestSplit(0).randomSplit (Array(0.8,0.2) , seed=2014554)


        _trainingSet    = Some(trainValidationSplit(0))
        _validationSet  = Some(trainValidationSplit(1))
        _testSet        = Some(trainTestSplit(1))
    }

    def printWranglingSpecs = {
        println (s"""-> N Record Wrangled Dataset: ${_wrangledSet.get.count} \n
                |-> N Record Training-Set: ${_trainingSet.get.count} \n 
                |-> N Record Validation-Set: ${_validationSet.get.count} \n
                |-> N Record Test-Set: ${_testSet.get.count} \n""".stripMargin)
    }

}

