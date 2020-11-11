/*
@author    :     rscalia
@date      :     Wed 11/11/2020

Questa classe serve per valutare un modello di raccomandazione basato sulla Singular Value Decomposition.
*/

package lib.SVD

import org.apache.spark.sql._
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.commons.lang.ArrayUtils
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.Matrix
import scala.math._



class SVDEvaluator (_spark:SparkSession, _dimensions:Int) extends Serializable{
    import _spark.implicits._

    var _errorsRDD:Option[RDD[Double]]  = None 


    def computeErrors (pModel:SingularValueDecomposition[RowMatrix, Matrix], pTestData:RDD[Vector]) = {

        val engine = new SVDEngine

        val errorsLC = pTestData.flatMap ( (usr_vector) => {

                                            //Estraggo vettori indici e valori
                                            val index  = usr_vector.toSparse.indices
                                            val iter_idx = index.clone
                                            val values = usr_vector.toSparse.values
                                            
                                            var erros_per_user = ArrayBuffer[Double]()
                                            
                                            iter_idx.foreach ( (idx) => {

                                                                    //Preparo il vettore di test
                                                                    var testIndex = index.clone
                                                                    val idxDel    = ArrayUtils.indexOf(testIndex,idx)
                                                                    testIndex = ArrayUtils.remove(testIndex, idxDel)

                                                                    var testArray = values.clone
                                                                    testArray = ArrayUtils.remove(testArray, idxDel)


                                                                    //Estraggo la ground-truth
                                                                    val gt      = values(idxDel)

                                                                    
                                                                    
                                                                    val testVec = Vectors.sparse(_dimensions, testIndex, testArray)


                                                                    
                                                                    //Calcolo l'inferenza
                                                                    val resVec  = engine.compute(pModel, testVec)


                                                                    //Prelevo il risultato
                                                                    val res_values = resVec.toDense.values
                                                                    val inference_res  = res_values(idx)

                                                                    
                                                                    //Calcolo l'errore al quadrato
                                                                    val error   = gt - inference_res
                                                                    val sq_error = pow (error, 2)

                                                                    
                                                                    //Aggiungo l'errore al cumulatore
                                                                    erros_per_user += sq_error

                                            } )

                                            erros_per_user
                                        

        } )

        _errorsRDD = Some(errorsLC)
        
    }

    def retrieveRMSE:Double = {
        val errors = _errorsRDD.get

        val sum = errors.reduce ( (a,b) => a+b )

        val mean = sum / errors.count

        val rmse = math.sqrt (mean)

        rmse
    } 
}