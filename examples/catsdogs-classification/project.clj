(defproject catsdogs-classification "0.9.13"
  :description "An example of using experiment/classification on mnist."
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 [thinktopic/experiment "0.9.13"]
                 [org.clojure/tools.cli "0.3.5"]

                 ; to manipulate images
                 [thinktopic/think.image "0.4.11"]
                 ; [net.mikera/imagez "0.12.0"]
                 ;;If you need cuda 8...
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 ;;If you need cuda 7.5...
                 ;;[org.bytedeco.javacpp-presets/cuda "7.5-1.2"]
                 ]

  :jvm-opts ["-Xmx2000m"]
  :uberjar-name "classify-example.jar"

  :clean-targets ^{:protect false} [:target-path
                                    "figwheel_server.log"
                                    "resources/public/out/"
                                    "resources/public/js/app.js"])
