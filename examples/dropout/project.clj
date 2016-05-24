(defproject dropout "0.1.0-SNAPSHOT"
  :description "Cortex dropout example project implementing http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/cortex "0.1.1-SNAPSHOT"]
                 [thinktopic/cortex-datasets "0.3.0-SNAPSHOT"]
                 [thinktopic/cortex-gpu "0.1.0-SNAPSHOT"]
                 [net.mikera/imagez "0.10.0"]
                 [thinktopic/tsne-core "0.1.0"]
                 [incanter/incanter-core "1.5.7"]
                 [incanter/incanter-charts "1.5.7"]]

  :main dropout.autoencoder)
