/*
@author    :     rscalia
@date      :     Wed 11/11/2020

Questa classe serve per calcolare la Singular Value Decomposition dell'Utility-Matrix.
*/

package lib.SVD

import org.apache.spark.sql._
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.Matrix


class SVDTrainer extends Serializable{

    var _model:Option[SingularValueDecomposition[RowMatrix, Matrix]] = None
    var _subModels:Option[ArrayBuffer[SingularValueDecomposition[RowMatrix, Matrix]]] = None
    var _validationInfo:ArrayBuffer[Tuple2[Int, Double]] = ArrayBuffer[Tuple2[Int, Double]] ()
    

    def fit (pTraining:RDD[Vector], pValidation:RDD[Vector], pLatentFactors:Array[Int], pEvaluator:SVDEvaluator) = {
        val mat: RowMatrix = new RowMatrix(pTraining)

        val subModels = ArrayBuffer[SingularValueDecomposition[RowMatrix, Matrix]] ()


        for (lfc <- pLatentFactors) {
            val svd = mat.computeSVD(lfc, computeU = true)
            
            subModels += svd

            pEvaluator.computeErrors(svd,pValidation)


            val rmse = pEvaluator.retrieveRMSE

            _validationInfo += Tuple2(lfc, rmse)
        }

        val bestModelIdx    = _validationInfo.reduce ( (x,y) => { if (x._2 < y._2) x else y } )

        _subModels          = Some(subModels)
        _model              = Some ( subModels ( _validationInfo.indexOf( bestModelIdx ) ) )


    }
}