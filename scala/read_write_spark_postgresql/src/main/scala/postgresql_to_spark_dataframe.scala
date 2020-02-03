/* SimpleApp.scala */
import org.apache.spark.sql.SparkSession
import java.util.Properties
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SaveMode


object SimpleApp {
  def main(args: Array[String]) {
    // print working directory
    var cwd = System.getProperty("user.dir")
    println(cwd)


    //set logging level
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    //read data from postgresql
    // Change to Your Database Config
    val conn_str = System.getenv("JDBC_BI_DB_CONNSTRING")
    println(conn_str)

    val connectionProperties = new Properties()
    connectionProperties.setProperty("Driver", "org.postgresql.Driver")

    //define the query
    val query1 = "(SELECT * FROM dwh.audits LIMIT 1000) as q1"

    val spark = SparkSession.builder.appName("Simple Application").config("spark.master", "local").getOrCreate()
    val query1df = spark.read.jdbc(conn_str, query1, connectionProperties)
    //show the output of the dataframe
    query1df.show(100)

    //write the dataframe backt to Postgresql database:
    query1df.write
      .format("jdbc")
      .option("url", conn_str)
      .option("dbtable", "test.spark_test_audits_table")
      .mode(SaveMode.Append)
      //.option("user", "username")
      //.option("password", "password")
      .save()
    println("Saved dataframe to Postgresql table.")

    //stop the spark cluster
    spark.stop
  }
}

