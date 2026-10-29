ThisBuild / scalaVersion := "2.13.16"
ThisBuild / version := "0.1.0"
val chiselVersion = "7.0.0"
lazy val root = (project in file(".")).settings(
  name := "simd-systolic",
  libraryDependencies += "org.chipsalliance" %% "chisel" % chiselVersion,
  addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
  scalacOptions ++= Seq("-language:reflectiveCalls", "-deprecation", "-feature")
)
