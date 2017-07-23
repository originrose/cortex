(defproject learn-cortex "0.0.1-SNAPSHOT"
  :dependencies [[org.clojure/clojure "1.8.0"]

                 [thinktopic/cortex "0.9.11"]
                 [thinktopic/experiment "0.9.11"]]
  :repl-options {:init-ns xor-mlp.core})
