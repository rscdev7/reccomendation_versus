/*
@author    :     rscalia
@date      :     Sun 01/11/2020

Questa classe serve per leggere il dataset MovieLens dal file system distribuito.
*/

package lib.MovieLensLoader 

import org.apache.spark.sql._


class MovieLensLoader (val _dataPath:String, val _spark:SparkSession) {

    import _spark.implicits._
    var _dataLake:Option[Dataset[Row]] = None


    def loadData: Unit = {
        val tmp_data_lake       = _spark.read.option("header",false).option("delimiter","_").csv(_dataPath).toDF("user_id","movie_id","label","timestamp")
        
        val cast_tmp_data_lake  = tmp_data_lake.select (tmp_data_lake("user_id").cast("int"), tmp_data_lake("movie_id").cast("int"), tmp_data_lake ("label").cast("int"), tmp_data_lake("timestamp").cast("long"))

        _dataLake               = Some(cast_tmp_data_lake)
    }


    def printData:Unit = {
        println ("\n******************** DATA LOADER CONFIG ***********************\n")
        println (s"-> DataPath: ${_dataPath} ")   
        println ("\n***************************************************************\n")
    }


    def checkDataIntegrity : Unit = {
        
        val integrity   = _dataLake.get.filter ( (row) => row.getInt(0) == null || row.getInt(1) == null || row.getInt(2) == null|| row.getLong(3) == null || row.getInt(2) < 0 || row.getInt(2) > 5 || row.getLong(3) <0) 

        val n_record    = integrity.count()

        n_record match {
            case i if i == 0 => println ("[!] Dataset is ok")
            case i if i > 0  => println (s"[!] Error, there is some thing wrong on your data <-> N_Rec: ${n_record}")
        }
    }
    
}

