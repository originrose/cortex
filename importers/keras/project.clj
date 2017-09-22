(defproject thinktopic/cortex-keras "0.9.20-SNAPSHOT"
  :description "Import of keras models into cortex descriptions"
  :url "http://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 [thinktopic/hdf5 "0.2.1"]
                 [thinktopic/cortex "0.9.20-SNAPSHOT"]
                 [cheshire "5.6.3"]
                 [thinktopic/think.image "0.4.12"]]
  :profiles {:ci {:dependencies [[thinktopic/hdf5 "0.1.3"]]}}
  :plugins [[lein-codox "0.10.2"]]
  :test-selectors {:ci (complement :skip-ci)}
  :main think.cortex.keras.core)
