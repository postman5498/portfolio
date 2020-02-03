/* SimpleApp.scala */
import java.sql.{Connection, DriverManager, ResultSet}

object read_postgresql_data {
  def main(args: Array[String]) {
    var cwd = System.getProperty("user.dir")
    println(cwd)

    // Change to Your Database Config
    val conn_str = System.getenv("JDBC_BI_DB_CONNSTRING")
    println(conn_str)

    // Load the driver
    val driver = "org.postgresql.Driver"

    Class.forName(driver)

    // Setup the connection
    val conn = DriverManager.getConnection(conn_str)

    try {
      // Configure to be Read Only
      val statement = conn.createStatement(ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY)

      // Execute Query
      val rs = statement.executeQuery("SELECT id FROM dwh.companies_view LIMIT 10 ")

      // Iterate Over ResultSet
      while (rs.next) {
        println(rs.getString("id"))
      }
    }
    finally {
      conn.close
    }

    }
}

