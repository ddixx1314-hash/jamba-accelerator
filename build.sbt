// Jamba 2.0 mini accelerator — Chisel project
ThisBuild / scalaVersion := "2.13.12"
ThisBuild / version      := "0.1.0"

val chiselVersion = "6.2.0"

lazy val root = (project in file("."))
  .settings(
    name := "jamba-accelerator",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel"     % chiselVersion,
      "edu.berkeley.cs"   %% "chiseltest" % "6.0.0" % Test
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature"
    ),
    addCompilerPlugin(
      "org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full
    ),
  )
