package simd

import java.nio.file.{Files,Paths}
import java.util.Properties

/** The same checked-in contract is consumed by Python and physical scripts. */
object Architecture {
  private val properties=new Properties
  private val stream=Files.newInputStream(Paths.get("configs/architecture.properties"))
  try properties.load(stream) finally stream.close()
  def value(key:String):String=Option(properties.getProperty(key)).getOrElse(
    throw new IllegalArgumentException(s"missing architecture property $key"))
  def number(key:String):Int=value(key).toInt
  val stages=number("stages")
  val sramReadLatency=number("sram_read_latency")
  require(stages==2 && value("process")=="asap7","only F2 ASAP7 is supported")
}
