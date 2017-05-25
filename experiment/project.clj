(defproject thinktopic/experiment "0.9.9-SNAPSHOT"
  :description "A higher-level library for performing experiments with cortex."
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/cortex "0.9.9-SNAPSHOT"]
                 [thinktopic/think.image "0.4.10"]
                 [org.shark8me/tfevent-sink "0.1.2-SNAPSHOT"]
                 ;;Default way of displaying anything is a web page.
                 ;;Because if you want to train on aws (which you should)
                 ;;you need to get simple servers up and running easily.
                 [thinktopic/think.gate "0.1.3"]
                 ;;This had better precisely match the version of figwheel that think.gate uses
                 ;;Tried with 1.9.XXX and had odd unexplainable failures.
                 [org.clojure/clojurescript "1.8.51"]]

  :plugins [[lein-cljsbuild "1.1.5"]
            [lein-garden "0.3.0"]]

  :garden {:builds [{:id "dev"
                     :source-paths ["src"]
                     :stylesheet css.styles/styles
                     :compiler {:output-to "resources/public/css/app.css"}}]}

  :cljsbuild {:builds
              [{:id "dev"
                :figwheel true
                :source-paths ["src/cljs/"]
                :compiler {:main "cortex.experiment.classify"
                           :asset-path "out"
                           :output-to "resources/public/js/app.js"
                           :output-dir "resources/public/out"}}
               {:id "prod"
                :source-paths ["src/cljs/"]
                :jar true
                :compiler {:main "cortex.experiment.classify"
                           :output-to "resources/public/js/app.js"
                           :output-dir "target/uberjar"
                           :optimizations :advanced
                           :pretty-print false}}]}

  :figwheel {:css-dirs ["resources/public/css"]}

  :prep-tasks ["compile" ["garden" "once"] ["cljsbuild" "once" "prod"]]

  :clean-targets ^{:protect false} [:target-path
                                    "figwheel_server.log"
                                    "resources/public/out/"
                                    "resources/public/js/app.js"])
