(defproject thinktopic/cortex "0.1.0-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :dependencies [[org.clojure/clojure "1.8.0-RC4"]
                 [com.taoensso/timbre "4.2.0"]
                 [net.mikera/vectorz-clj "0.40.0"]
                 [org.clojure/test.check "0.9.0"]]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"])
