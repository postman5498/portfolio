/* SimpleApp.scala */
import org.apache.spark.sql.SparkSession
import java.util.Properties

object SimpleApp {
  def main(args: Array[String]) {
    // print working directory
    var cwd = System.getProperty("user.dir")
    println(cwd)

    //read data from postgresql
    // Change to Your Database Config
    val conn_str = System.getenv("JDBC_BI_DB_CONNSTRING")
    println(conn_str)

    val connectionProperties = new Properties()
    connectionProperties.setProperty("Driver", "org.postgresql.Driver")

    //define the query
    val query1 = "(SELECT * FROM test.test_table) as q1"

    val spark = SparkSession.builder.appName("Simple Application").config("spark.master", "local").getOrCreate()
    val query1df = spark.read.jdbc(conn_str, query1, connectionProperties)

    query1df.show()
  }
}

