name := "ALSReccomendation"

version := "1.0.0"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.0" ,
  "org.apache.spark" %% "spark-sql" % "2.3.0" ,
  "org.apache.spark" %% "spark-mllib" %  "2.3.0" ,
  "org.apache.spark" %% "spark-streaming" %  "2.3.0",
  "org.scalatest" %% "scalatest" % "3.0.8" % Test 
)