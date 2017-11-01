(defproject mnist-classification "0.9.23-SNAPSHOT"
  :description "An example of using experiment/classification on mnist."
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 [thinktopic/experiment "0.9.23-SNAPSHOT"]
                 [org.clojure/tools.cli "0.3.5"]
                 ;;If you need cuda 8...
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 ;;If you need cuda 7.5...
                 ;;[org.bytedeco.javacpp-presets/cuda "7.5-1.2"]
                 ]

  :main mnist-classification.main
  :aot [mnist-classification.main]
  :jvm-opts ["-Xmx2000m"]
  :uberjar-name "classify-example.jar"

  :profiles {:liq {:dependencies [[mogenslund/liquid "0.8.2"]]
             :main dk.salza.liq.core}}
  :aliases {"liq" ["with-profile" "liq" "run" "--load=.liq"]}

  :clean-targets ^{:protect false} [:target-path
                                    "figwheel_server.log"
                                    "resources/public/out/"
                                    "resources/public/js/app.js"])
