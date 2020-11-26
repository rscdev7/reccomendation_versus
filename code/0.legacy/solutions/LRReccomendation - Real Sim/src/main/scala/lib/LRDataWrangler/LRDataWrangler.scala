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
import org.apache.spark.ml.linalg.Matrix
import scala.collection.JavaConversions._
import org.apache.spark.ml.stat._


case class LinRegrDatasetRec (val features:Vector   , val label:Double)
case class SimVersusRec      (val _movieId:Int      , val _label:Int, val _sim:Double)

class LRDataWrangler ( val _spark:SparkSession      ,   val _nFilm:Int, 
                       val _kNN:Long                ,   val _datasetSavePath:String     ,
                       val _nSplits:Int)                extends Serializable{


    import _spark.implicits._

    var _wrangleSet:Option[Dataset[LinRegrDatasetRec]]                      = None

    var _trainingSet:Option[Dataset[LinRegrDatasetRec]]                     = None
    var _validationSet:Option[Dataset[LinRegrDatasetRec]]                   = None
    var _testSet:Option[Dataset[LinRegrDatasetRec]]                         = None


    def wrangle (pDataset:Dataset[Row]) = {


        //Generazione Utility-Matrix
        val augmented_dataset   = pDataset.withColumn ( "film_&_rating", struct ( pDataset("movie_id") - 1 , pDataset("label")  ) )


        
        val user_per_film       = augmented_dataset.groupBy (augmented_dataset.col("user_id")).agg(collect_list( "film_&_rating" ))

    
        
        val utilityMatrixRDD    = user_per_film.rdd.map ( (record) => {
                                                val pairsList       = record.getSeq[GenericRowWithSchema](1)
                                                
                                                val castedPairsList = pairsList.map ( (x) => Tuple2( x.asInstanceOf[Row].getInt(0), x.asInstanceOf[Row].getInt(1).toDouble ))

                                                val v:Vector = Vectors.sparse(_nFilm, castedPairsList)

                                                v
                                            })

        val utilityMatrix       = utilityMatrixRDD.map(Tuple1.apply).toDF("features")



        //Calcolo Matrice di Correlazione
        val Row(coeff1: Matrix) = Correlation.corr(utilityMatrix, "features").head



        //Preparazione Strutture dati per il Wrangling
        val selectedDataset     = augmented_dataset.select("user_id", "movie_id", "label")
        val shuffledDataset     = selectedDataset.repartition(_nSplits)
        val datasetIterator     = shuffledDataset.toLocalIterator
        //val record            = datasetIterator.next
        var linRegreDataset     = Seq.empty[LinRegrDatasetRec].toDS()
        


        //Wrangling - a.k.a. costruzione dataset per il regressore lineare
        for ( record <- datasetIterator) {

            //Prelievo dati dell'i-esimo record
            val currentUser             = record.getInt(0)
            val currentFilm             = record.getInt(1)
            val currentLabel            = record.getInt(2)


            //Prelievo film visti dall'utente diversi da quello in esame
            val filteredDataset         = shuffledDataset.filter ( (x) => (x.getInt(0) == currentUser && x.getInt(1) != currentFilm) )

            val nRecord                 = filteredDataset.count


            nRecord match {
                case i if i >= _kNN => {

                                //Selezione Campi Utili dal dataset filtrato
                                val osservationMatrixSketch                 = filteredDataset.select ( "movie_id" , "label" )

            
                                //Calcolo Approssimato del K-NN
                                val key                                     = currentFilm-1
                                val augMatrixSketch                         = osservationMatrixSketch.withColumn ("sim", lit(0.0))

            
                                val simMatrixSkecth                         = augMatrixSketch.map ( (x) => SimVersusRec (x.getInt(0), x.getInt(1), coeff1(key, x.getInt(0)-1 ) ) )


                                val notNaNsimMatrix                         = simMatrixSkecth.na.drop


                                val nRecSim                                 = notNaNsimMatrix.count



                                nRecSim match {
                                    case i if i>=_kNN => {
                                                    val orderedSketch = notNaNsimMatrix.sort($"_sim".desc)
                                                    val labelSketch   = orderedSketch.select(orderedSketch.col("_label"))
                                                    val knn           = labelSketch.take(_kNN.toInt)


                                                    //Estrazione feature per l'i-esimo record
                                                    val unWrapFeatureArray      = knn.map ( (x) => x.getInt(0).toDouble )

                                                    val featureVector           = Vectors.dense(unWrapFeatureArray)

                                                    val finalRec                = LinRegrDatasetRec (featureVector, currentLabel)


                                                    //Memorizzazione record nel dataset finale
                                                    val recDataset              = Seq(finalRec).toDS()
                                                    linRegreDataset             = linRegreDataset.union(recDataset)

                                            }

                                    case _ => println ("\n\n\n[!] Skipped Record for NaN\n\n\n")
                                }

                                
                            }
                case _ => println ("\n\n\n[!] Skipped Record for low Viewed Film\n\n\n")
            }

            
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

