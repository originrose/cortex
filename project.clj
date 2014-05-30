(defproject diabolo "0.1.0-SNAPSHOT"
  :description "Auto-associative (autoencoder) neural networks in Clojure."
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [com.nuroko/nurokit "0.0.3"]
                 [net.mikera/imagez "0.3.1"]]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"]

  :main ^{:skip-aot true} diabolo.core)
