/* SimpleApp.scala */
// import org.apache.spark.sql.SparkSession

//object SimpleApp {
//  def main(args: Array[String]) {
//    val logFile = "YOUR_SPARK_HOME/README.md" // Should be some file on your system
//    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
//    val logData = spark.read.textFile(logFile).cache()
//    val numAs = logData.filter(line => line.contains("a")).count()
//    val numBs = logData.filter(line => line.contains("b")).count()
//    println(s"Lines with a: $numAs, Lines with b: $numBs")
//    spark.stop()
//  }
//}

//object Demo {
//  def main(args: Array[String]) {
//    // var pwd =
//    println(System.getProperty("user.dir"))
//    val ages = Seq(42, 75, 29, 64)
//    println(s"The oldest person is ${ages.max}")
//  }
//}


import java.io.File
import javax.imageio.ImageIO
import java.awt.image.BufferedImage

object Demo {
  def main(args: Array[String]) {
    def phototest(img: BufferedImage): BufferedImage = {
      // obtain width and height of image
      val w = img.getWidth
      val h = img.getHeight

      // create new image of the same size
      val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)

      // copy pixels (mirror horizontally)
      for (x <- 0 until w)
        for (y <- 0 until h)
          out.setRGB(x, y, img.getRGB(w - x - 1, y) & 0xffffff)

      // draw red diagonal line
      for (x <- 0 until (h min w))
        out.setRGB(x, x, 0xff0000)

      out
    }
    def test() {
      // read original image, and obtain width and height
      val photo1 = ImageIO.read(new File("files/dashboard_design.jpg"))
      printf("Photo size is %d x %d\n", photo1.getWidth, photo1.getHeight)
      val photo2 = phototest(photo1)

      // save image to file "test.jpg"
      ImageIO.write(photo2, "jpg", new File("files/test.jpg"))
    }



    test()
  }
  }

