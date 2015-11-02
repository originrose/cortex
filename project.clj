(defproject thinktopic/cortex "0.1.0-SNAPSHOT"
  :description "A neural network toolkit for ThinkTopic projects."
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [com.taoensso/timbre "4.1.4"]
                 [net.mikera/vectorz-clj "0.35.0"]
                 [thinktopic/datasets "0.1.1"]
                 [jarohen/chord "0.6.0"]]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"])
