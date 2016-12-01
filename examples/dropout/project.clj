(defproject dropout "0.3.1-SNAPSHOT"
  :description "Cortex dropout example project implementing http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/cortex "0.3.1-SNAPSHOT"]
                 [thinktopic/cortex-datasets "0.3.1-SNAPSHOT"]
                 [thinktopic/cortex-visualization "0.3.1-SNAPSHOT"]
                 [thinktopic/resource "1.1.0"]]

  :jvm-opts  ["-Xmx2g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"]

  :main dropout.autoencoder)
