/*
@author    :     rscalia
@date      :     Thu 26/11/2020

Questa classe serve per valutare le performance di un modello LSH su di un dataset.
*/

package lib.LSHEvaluator

import scala.reflect._
import scala.collection._
import scala.math

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


case class SaveRecord (val BucketLen:Double, val N_HashTables:Int, val DistThr:Int, val RMSE:Double )


class LSHEvaluator (val _spark:SparkSession, val _savePath:String, val _splits:Int) extends Serializable {

    import _spark.implicits._
    

    var _errors:Option[Dataset[Double]] = None



    def parallelEvaluate (pDataLake:Dataset[Row], pTestSet:Dataset[Row] ,pModel:Dataset[Row]) = {
        import scala.collection.mutable._

        //Per ogni Tripla (U,F,R) del Test-Set rendo esplicito i film valutati dall'Utente presente nell'n-esimo record
        // (U,F,R) ---> (F,R) , ...., (F,R)
        val augmented_dataset               = pDataLake.withColumn ( "film_&_rating", struct ( pDataLake("movie_id") , pDataLake("label")  ) )

        val groupedDtUser                   = augmented_dataset.groupBy (augmented_dataset.col("user_id")).agg(collect_list( "film_&_rating" ).as("candidateSim"))

        val res                             = pTestSet.join (groupedDtUser, "user_id").select("user_id","movie_id","label","candidateSim")


        //Per ogni Tripla (U,F,R) del Test-Set assegno il corrispettivo cluster LSH di Film
        // (U,F,R) ---> (F,R) , ...., (F,R) | F, ...., F
        val groupedModel                    = pModel.groupBy (col("idA")).agg (collect_list("idB").as("candidateLSH"))
        val renamedGroupedModel             = groupedModel.select ( col("idA").as("movie_id") , col("candidateLSH") )

        val parallelDataset                 = res.join (renamedGroupedModel, Seq("movie_id"), "left")



        val errors = parallelDataset.map ( (x) => {
                                                //Retrieving Record Information
                                                val user                    = x.getInt(0)
                                                val film                    = x.getInt(1)
                                                val rating                  = x.getInt(2)

                                                //Retrieving User Rated Films except current ----> (M,R) , (M,R)
                                                val rowCandidateSim         = x.getSeq[GenericRowWithSchema](3)
                                                val candidateSim            = rowCandidateSim.map ( (x) => Tuple2( x.asInstanceOf[Row].getInt(0), x.asInstanceOf[Row].getInt(1) ) ).par
                                                val filteredCandidateSim    = candidateSim.filter ( (x) => x._1 != film)


                                                //Controllo presenza film nel bucket associato al film attuale
                                                val candidateLSH            = x.isNullAt(4) match {
                                                    case true  => Array[Int]().par
                                                    case false => x.getAs[WrappedArray[Int]](4).par
                                                }



                                                //Prelevo i Film che voteranno per l'inferenza
                                                val voters                  = filteredCandidateSim.map((x) => x._1).intersect (candidateLSH)

                                                


                                                //Calcolo Numero Voti
                                                val n_votes                     = voters.length
                                                var wrappedMean:Option[Double]  = None



                                                //[?] Check if there are LSH Voters 
                                                n_votes match {
                                                    case i if i>0 => {

                                                        val wrappedVoters           = voters.toSet

                                                        val concreteVoters          = filteredCandidateSim.filter ( (x) => wrappedVoters(x._1) != false )

                                                        val concreteNumericVoters   = concreteVoters.map ((x) => x._2)


                                                        val sum                      = concreteNumericVoters.sum
                                                        val len                      = concreteNumericVoters.length


                                                        //Retrieving Mean of Voters
                                                        wrappedMean                  = Some(sum/len)
                                                        
                                                    }

                                                    case i if i<=0 => {
                                                        val sum                       = filteredCandidateSim.map((x) => x._2).sum
                                                        val len                       = filteredCandidateSim.length
                                                        wrappedMean                   = Some( sum/len )
                                                    }
                                                }


                                                //Compunting Squared Error
                                                val inferenceRating                 = wrappedMean.get
                                                val diff                            = rating - inferenceRating
                                                val sq_diff                         = math.pow(diff,2)

                                                sq_diff
                                                

        } )

        _errors = Some(errors)
    }


    def write (pBucketLen:Double, pNHashTables:Int, pDistThr:Int) = {
        val rmse        = retrieveRMSE

        val save_blob   = SaveRecord(pBucketLen, pNHashTables, pDistThr, rmse)

        val dtBlob      = Seq(save_blob).toDS()

        dtBlob.write.format("csv").option("header",true).save(_savePath)
    }


    def retrieveRMSE:Double = {
        val errors      = _errors.get

        val sum         = errors.reduce ( (a,b) => a+b )

        val mean        = sum / errors.count

        val rmse        = math.sqrt (mean)

        rmse
    } 

}