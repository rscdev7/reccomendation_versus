/*
@author    :     rscalia
@date      :     Sun 01/11/2020

Questa classe serve per calcolare un modello LSH basato sulla distanza Euclidea a partire da una Dataset.
*/

package lib.LSH

import scala.reflect._
import scala.collection._

import org.apache.spark.sql._
import org.apache.spark.storage._
import org.apache.spark.rdd._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import scala.collection.JavaConversions._
import org.apache.spark.ml.stat._


case class UtilityMatrixRec (val movie_id:Int  , val _dataPoint:Vector)
case class LSHRecord        (val _keyMovie:Int , val _valueMovie:Seq[Int])

class LSH (     val _spark:SparkSession             ,   val _nUser:Int, 
                val _distThr:Int                    ,   val _bucketLen:Double               ,  
                val _nHashTables:Int            ) extends Serializable{


    import _spark.implicits._


    var _model:Option[Dataset[Row]] = None


    def fit (pDataset:Dataset[Row]) = {


        //Generazione Utility-Matrix
        val augmented_dataset   = pDataset.withColumn ( "user_&_rating", struct ( pDataset("user_id") - 1 , pDataset("label")  ) )


        
        val user_per_film       = augmented_dataset.groupBy (pDataset.col("movie_id")).agg(collect_list( "user_&_rating" ))

    
        
        val utilityMatrix       = user_per_film.map ( (record) => {
                                                val pairsList = record.getSeq[GenericRowWithSchema](1)
                                                
                                                
                                                val castedPairsList = pairsList.map ( (x) => Tuple2( x.asInstanceOf[Row].getInt(0), x.asInstanceOf[Row].getInt(1).toDouble ))

                                                val v:Vector = Vectors.sparse(_nUser, castedPairsList)

                                                UtilityMatrixRec (record.getInt(0) , v)
                                            })


        //Preparazione Strutture md.show()dati necessarie a LSH
        val brp                 = new BucketedRandomProjectionLSH()
                                    .setBucketLength(_bucketLen)
                                    .setNumHashTables(_nHashTables)
                                    .setInputCol("_dataPoint")
                                    .setOutputCol("hashes")

        
        //Fitting LSH Model
        val model               = brp.fit(utilityMatrix)
        model.transform(utilityMatrix)


        //Evaluate LSH-Model on Data
        val md = model.approxSimilarityJoin(utilityMatrix, utilityMatrix, _distThr, "EuclideanDistance")
                                    .select( col("datasetA.movie_id").alias("idA") , col("datasetB.movie_id").alias("idB"))

        _model = Some(md)

    }
    

}

