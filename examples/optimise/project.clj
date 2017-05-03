(defproject thinktopic/cortex-optimise "0.9.7"
  :description "General purpose optimization framework"
  :url "http://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :think/meta {:type :library
               :tags [:clojure :optimization :gradient-descent
                      :machine-learning :exploratory :mathematica
                      :repl]}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/cortex "0.9.7"]
                 [net.mikera/vectorz-clj "0.45.0"]
                 [net.mikera/core.matrix "0.57.0"]
                 [thinktopic/lazy-map "0.1.0"]]
  :main cortex.optimise.descent)
