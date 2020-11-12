/*
@author    :     rscalia
@date      :     Thu 12/11/2020

Questa classe serve per scrivere su disco i risulati del training di un modello basato sulla raccomandazione SVD.
*/

package lib.SVDDataWriter

import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.sql._
import org.apache.spark.rdd._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.Vector
import lib.SVD._

case class modelWriteData (val _latentFactors:Int, val _rmse:Double)

class SVDDataWriter (val _spark:SparkSession, val _dimensions:Int, val _validationPath:String, 
                     val _evaluationPath:String) {

    import _spark.implicits._

    def write (pBestModel:SingularValueDecomposition[RowMatrix, Matrix], pTestSet:RDD[Vector], pValidationInfo:ArrayBuffer[Tuple2[Int, Double]]) = {

        //Computing errors on Test-Set using Best Model
        val evaluator = new SVDEvaluator (_spark, _dimensions)

        evaluator.computeErrors(pBestModel ,pTestSet)
        val rmse = evaluator.retrieveRMSE


        //Preparing data for Writing
        val best_model_csv_dataset          = Seq( modelWriteData( pBestModel.s.size, rmse ) ).toDF ("LFC", "RMSE")
        val validation_models_csv_list      = for (info <- pValidationInfo) yield modelWriteData(info._1, info._2)
        val validation_models_csv_dataset   = validation_models_csv_list.toDF("LFC","RMSE")


        //Writing data on the distribuited file system
        best_model_csv_dataset.write.format("csv").option("header",true).save(_evaluationPath)
        validation_models_csv_dataset.write.format("csv").option("header",true).save(_validationPath)

    }
}