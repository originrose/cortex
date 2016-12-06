(defproject suite-classification "0.3.1-SNAPSHOT"
  :description "Example of using the turn key classification system"
  :url "http://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/cortex.suite "0.3.1-SNAPSHOT"]
                 [thinktopic/gpu-compute "0.3.1-SNAPSHOT"]
                 [thinktopic/gate "0.1.1-SNAPSHOT"]
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]]



  :source-paths ["src" "cljs"]

  :clean-targets ^{:protect false} ["pom.xml"
                                    "target"
                                    "resources/public/out"
                                    "resources/public/js/app.js"
                                    "figwheel_server.log"]

  :cljsbuild {:builds
              [{:id "dev"
                :figwheel true
                :source-paths ["cljs"]
                :compiler {:main "cortex.suite.classify"
                           :asset-path "out"
                           :output-to "resources/public/js/app.js"
                           :output-dir "resources/public/out"}}]}

  :main suite-classification.main
  :aot [suite-classification.main])
