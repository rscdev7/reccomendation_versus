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
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrix


class SVDTrainer {

    var _model:Option[Matrix] = None

    def fit (pTraining:RDD[Vector], pValidation:RDD[Vector], pLatentFactor:Int) = {

    }
}