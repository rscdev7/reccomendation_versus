/*
@author    :     rscalia
@date      :     Wed 11/11/2020

Questa classe serve per fare raccomandazione basandosi sull'SVD.
*/

package lib.SVD

import org.apache.spark.sql._
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrix


class SVDEngine extends Serializable{

    def compute (pModel:SingularValueDecomposition[RowMatrix, Matrix], pArgument:Vector):Vector = {
        val transpose_V         = pModel.V.transpose
        val film_2_factor       = transpose_V.multiply(pArgument)

        val computed_vector     = pModel.V.multiply(film_2_factor)

        computed_vector
    }
}