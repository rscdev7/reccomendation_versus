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


    def evaluate (pDataset:Dataset[Row], pModel:Dataset[Row]) = {
        
        println ("\n\n\n\n\n\n\n ************ START EVAL ************* \n\n\n\n\n\n\n ")

        val dtIterator  = pDataset.toLocalIterator
        val record      = dtIterator.next
        var err         = Seq.empty[Double].toDS()


        for (record <- dtIterator) {

            println ("\n\n\n\n RECORD PROCESSING \n\n\n\n")

            //Retrieve Information from Record
            val userId                          = record.getInt(0)
            val filmId                          = record.getInt(1)
            val rating                          = record.getInt(2)


            //Guardo al Vicinato della coppia (Utente_x, Film_x)
            val userRatedFilms                  = pDataset.filter ( (x) => x.getInt(0) == userId && x.getInt(1) != filmId ).select (col("movie_id"), col("label") )
            val userRatedLSH                    = pModel.filter ((x) => x.getInt(0) == filmId)

            //Prelevo i Film che voteranno per l'inferenza
            val inferenceSource                 = userRatedFilms.join(userRatedLSH, userRatedFilms("movie_id") === userRatedLSH("idB"), "leftsemi")


            val numberOfVotes                   = inferenceSource.count
            var wrappedMean:Option[Array[Row]]  = None


            //[?] Check if there are LSH Voters 
            numberOfVotes match {
                case i if i>0 => {

                    //Retrieving Mean of Voters
                    wrappedMean                 = Some(inferenceSource.agg(avg ("label")).collect)
                     
                }

                case i if i<=0 => {
                    wrappedMean                 = Some(userRatedFilms.agg(avg ("label")).collect)
                }
            }


            //Compunting Squared Error
            val inferenceRating                 = wrappedMean.get(0).getDouble(0)
            val diff                            = rating - inferenceRating
            val sq_diff                         = math.pow(diff,2)


            //Mergin on Accumulator
            val dtDiff                          = Seq(sq_diff).toDS()
            err                                 = err.union (dtDiff)

        }


        _errors                                 = Some(err)
                                                          
    }


    def parallelEvaluate (pDataset:Dataset[Row], pModel:Dataset[Row]) = {
        import scala.collection.mutable._

        //Per ogni Tripla (U,F,R) rendo esplicito i film valutati dall'Utente presente nell'n-esimo record
        // (U,F,R) ---> (F,R) , ...., (F,R)
        val augmented_dataset               = pDataset.withColumn ( "film_&_rating", struct ( pDataset("movie_id") , pDataset("label")  ) )

        val groupedDtUser                   = augmented_dataset.groupBy (augmented_dataset.col("user_id")).agg(collect_list( "film_&_rating" ).as("candidateSim"))

        val res                             = pDataset.join (groupedDtUser, "user_id").select("user_id","movie_id","label","candidateSim")


        //Per ogni Tripla (U,F,R) assegno il corrispettivo cluster LSH
        // (U,F,R) ---> (F,R) , ...., (F,R) | F, ...., F
        val groupedModel                    = pModel.groupBy (col("idA")).agg (collect_list("idB").as("candidateLSH"))
        val renamedGroupedModel             = groupedModel.select ( col("idA").as("movie_id") , col("candidateLSH") )

        val parallelDataset                 = res.join (renamedGroupedModel, "movie_id")



        val errors = parallelDataset.map ( (x) => {
                                                //Retrieving Record Information
                                                val user                    = x.getInt(0)
                                                val film                    = x.getInt(1)
                                                val rating                  = x.getInt(2)

                                                val rowCandidateSim         = x.getSeq[GenericRowWithSchema](3)
                                                val candidateSim            = rowCandidateSim.map ( (x) => Tuple2( x.asInstanceOf[Row].getInt(0), x.asInstanceOf[Row].getInt(1) ) ).par

                                                val candidateLSH            = x.getAs[WrappedArray[Int]](4).par



                                                //Guardo al Vicinato della coppia (Utente_x, Film_x)
                                                val filteredCandidateSim    = candidateSim.filter ( (x) => x._1 != film)



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


                                                        val sum                          = concreteNumericVoters.sum
                                                        val len                          = concreteNumericVoters.length


                                                        //Retrieving Mean of Voters
                                                        wrappedMean                     = Some(sum/len)
                                                        
                                                    }

                                                    case i if i<=0 => {
                                                        val sum                          = filteredCandidateSim.map((x) => x._2).sum
                                                        val len                          = filteredCandidateSim.length
                                                        wrappedMean                      = Some( sum/len )
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