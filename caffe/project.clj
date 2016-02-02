(defproject thinktopic/cortex-caffe "0.1.0-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :dependencies [[org.clojure/clojure "1.8.0-RC5"]
                 [thinktopic/cortex "0.1.0-SNAPSHOT"]
                 ]

  :java-source-paths ["java"]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"])
