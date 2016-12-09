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
                 [thinktopic/think.gate "0.1.1"]
                 ;;If you need cuda 8...
                 ;;[org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 ]

  :figwheel {:css-dirs ["resources/public/css"]}


  :source-paths ["src"]

  :clean-targets ^{:protect false} ["pom.xml"
                                    "target"
                                    "resources/public/out"
                                    "resources/public/js/app.js"
                                    "figwheel_server.log"]

  :cljsbuild {:builds
              [{:id "dev"
                :figwheel true
                :source-paths ["cljs"]
                :compiler {:main "suite-classification.classify"
                           :asset-path "out"
                           :output-to "resources/public/js/app.js"
                           :output-dir "resources/public/out"}}]}

  :main suite-classification.main
  :aot [suite-classification.main])
