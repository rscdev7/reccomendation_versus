/*
@author    :     rscalia
@date      :     Sun 01/11/2020

Questa classe serve per effettuare il wrangling dei dati necessario al training di un Regressore Lineare per il task di Raccomandazione.
*/

package lib.LRDataWrangler

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


case class UtilityMatrixRec (val movie_id:Int, val _dataPoint:Vector)
case class LinRegrDatasetRec (val features:Vector, val label:Double)


class LRDataWrangler ( val _spark:SparkSession      ,   val _nUser:Int, 
                       val _kNN:Int                 ,   val _datasetSavePath:String     ,
                       val _bucketLen:Double        ,   val _nHashTables:Int            ,
                       val _nSplits:Int)                extends Serializable{


    import _spark.implicits._

    var _wrangleSet:Option[Dataset[LinRegrDatasetRec]]                      = None

    var _trainingSet:Option[Dataset[LinRegrDatasetRec]]                     = None
    var _validationSet:Option[Dataset[LinRegrDatasetRec]]                   = None
    var _testSet:Option[Dataset[LinRegrDatasetRec]]                         = None


    def wrangle (pDataset:Dataset[Row]) = {


        //Generazione Utility-Matrix
        val augmented_dataset   = pDataset.withColumn ( "user_&_rating", struct ( pDataset("user_id") - 1 , pDataset("label")  ) )


        
        val user_per_film       = augmented_dataset.groupBy (pDataset.col("movie_id")).agg(collect_list( "user_&_rating" ))

    
        
        val utilityMatrix       = user_per_film.map ( (record) => {
                                                val pairsList = record.getSeq[GenericRowWithSchema](1)
                                                
                                                
                                                val castedPairsList = pairsList.map ( (x) => Tuple2( x.asInstanceOf[Row].getInt(0), x.asInstanceOf[Row].getInt(1).toDouble ))

                                                val v:Vector = Vectors.sparse(_nUser, castedPairsList)

                                                UtilityMatrixRec (record.getInt(0) , v)
                                            })


        //Preparazione Strutture dati per il Wrangling
        val selectedDataset     = pDataset.select("user_id", "movie_id", "label")
        val augMentedDataset    = selectedDataset.join (utilityMatrix, "movie_id")
        val reSelDataset        = augMentedDataset.select ("user_id", "movie_id", "label", "_dataPoint")
        val shuffledDataset     = reSelDataset.repartition(_nSplits)

        
  
        val brp                 = new BucketedRandomProjectionLSH()
                                    .setBucketLength(_bucketLen)
                                    .setNumHashTables(_nHashTables)
                                    .setInputCol("_dataPoint")
                                    .setOutputCol("hashes")

        
        val datasetIterator     = shuffledDataset.toLocalIterator
        var linRegreDataset     = Seq.empty[LinRegrDatasetRec].toDS()



        //Wrangling - a.k.a. costruzione dataset per il regressore lineare
        for ( record <- datasetIterator) {

            //Prelievo dati dell'i-esimo record
            val currentUser             = record.getInt(0)
            val currentFilm             = record.getInt(1)
            val currentLabel            = record.getInt(2)
            val currentKey              = record.getAs[Vector](3)


            //Prelievo film visti dall'utente diversi da quello in esame
            val filteredDataset         = shuffledDataset.filter ( (x) => (x.getInt(0) == currentUser && x.getInt(1) != currentFilm) )

            val osservationMatrixSketch = filteredDataset.select ( "movie_id" , "label" )


            //Calcolo Approssimato del K-NN
            val searchRegion            = utilityMatrix.join(osservationMatrixSketch, utilityMatrix("movie_id") === osservationMatrixSketch("movie_id"), "leftsemi")

            val model = brp.fit(searchRegion)
            model.transform(searchRegion)

            val approxNN                = model.approxNearestNeighbors(searchRegion, currentKey, _kNN)
            

            //Estrazione feature per l'i-esimo record
            val linearRegressionFeatures = osservationMatrixSketch.join(approxNN, osservationMatrixSketch("movie_id") === approxNN("movie_id"), "leftsemi")

            val features                = linearRegressionFeatures.select("label")

            val featuresArray           = features.collect

            val unWrapFeatureArray      = featuresArray.map ( (x) => x.getInt(0).toDouble )

            val featureVector           = Vectors.dense(unWrapFeatureArray)

            val finalRec                = LinRegrDatasetRec (featureVector, currentLabel)


            //Memorizzazione record nel dataset finale
            val recDataset              = Seq(finalRec).toDS()
            linRegreDataset             = linRegreDataset.union(recDataset)
                  
        }


        //Scrittura dataset sul file-system distribuito
        linRegreDataset.write.parquet(_datasetSavePath)

        //Memorizzazione del dataset all'interno dell'oggetto corrente
        _wrangleSet                     = Some(linRegreDataset)

    }

    
    def split = {
        val trainTestSplit                  = _wrangleSet.get.randomSplit (Array(0.8,0.2) , seed=2014554)
        val trainValidationSplit            = trainTestSplit(0).randomSplit (Array(0.8,0.2) , seed=2014554)


        _trainingSet                        = Some(trainValidationSplit(0))
        _validationSet                      = Some(trainValidationSplit(1))
        _testSet                            = Some(trainTestSplit(1))
    }

    
    def printWranglingSpecs = {
        println (s"""-> N Record Wrangled Dataset: ${_wrangleSet.get.count} \n
                |-> N Record Training-Set: ${_trainingSet.get.count} \n 
                |-> N Record Validation-Set: ${_validationSet.get.count} \n
                |-> N Record Test-Set: ${_testSet.get.count} \n""".stripMargin)
    }
    

}

