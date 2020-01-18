import java.io._
import java.nio.file.Paths


object Demo {
  def main(args: Array[String]) {
    println("PWD: " + Paths.get(".").toAbsolutePath)
    val writer = new PrintWriter(new File("test.txt" ))

    writer.write("Hello Scala overwritten this?")
    writer.close()
  }
}