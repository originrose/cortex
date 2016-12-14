(defproject suite-classification "0.3.1-SNAPSHOT"
  :description "Example of using the turn key classification system"
  :url "http://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/cortex.suite "0.3.1-SNAPSHOT"]
                 ;;Default way of displaying anything is a web page.
                 ;;Because if you want to train on aws (which you should)
                 ;;you need to get simple servers up and running easily.
                 [thinktopic/think.gate "0.1.2"]
                 ;;This had better precisely match the version of figwheel that think.gate uses
                 ;;Tried with 1.9.XXX and had odd unexplainable failures.
                 [org.clojure/clojurescript "1.8.51"] ;;Match figwheel
                 ;;If you need cuda 8...
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 ]


  :plugins [[lein-cljsbuild "1.1.5"]
            [lein-garden "0.3.0"]]


  :garden {:builds [{:id "dev"
                     :source-paths ["src"]
                     :stylesheet css.styles/styles
                     :compiler {:output-to "resources/public/css/app.css"}}]}


  :cljsbuild {:builds
              [{:id "dev"
                :figwheel true
                :source-paths ["cljs"]
                :compiler {:main "suite-classification.classify"
                           :asset-path "out"
                           :output-to "resources/public/js/app.js"
                           :output-dir "resources/public/out"}}]}


  :figwheel {:css-dirs ["resources/public/css"]}


  :main suite-classification.main
  :aot [suite-classification.main]
  :uberjar-name "classify-example.jar")
