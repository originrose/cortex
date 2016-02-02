(defproject thinktopic/cortex "0.1.0-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.taoensso/timbre "4.2.1"]
                 [net.mikera/vectorz-clj "0.43.0"]
                 [org.clojure/test.check "0.9.0"]
                 [thinktopic/matrix.fressian "0.2.1"]
                 [com.google.protobuf/protobuf-java "2.6.1"]]

  :profiles {:dev {:dependencies [[net.mikera/cljunit "0.3.1"]]
                   :java-source-paths ["test"]}}

  :source-paths ["src" "test"]

  :java-source-paths ["java"]

  :resource-paths ["resources"]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"]

  :main cortex.run-all-tests)
