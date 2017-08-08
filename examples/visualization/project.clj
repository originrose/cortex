(defproject thinktopic/cortex-visualization "0.9.12-SNAPSHOT"
  :description "Visualization library to aid in neural net training"
  :url "http://cortex.thinktopic.com"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 [thinktopic/think.tsne "0.1.1"]
                 [incanter/incanter-core "1.5.7"]
                 [incanter/incanter-charts "1.5.7"]
                 [thinktopic/think.image "0.4.2"]
                 [thinktopic/cortex "0.9.12-SNAPSHOT"]]
  :plugins [[lein-codox "0.10.2"]])
