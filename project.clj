(defproject thinktopic/cortex "0.1.0-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.taoensso/timbre "4.2.1"]
                 [net.mikera/vectorz-clj "0.43.1"]
                 [net.mikera/core.matrix "0.50.0-SNAPSHOT"]
                 [org.clojure/test.check "0.9.0"]
                 [thinktopic/matrix.fressian "0.3.0-SNAPSHOT"]
                 [caffe-protobuf "0.1.0"]
                                        ;[thinktopic/netlib-ccm "0.1.0-SNAPSHOT"]
                 [net.mikera/clojure-utils "0.6.2"]
                 ]

  :profiles {:dev {:dependencies [[net.mikera/cljunit "0.4.0"]  ;; allows JUnit testing
                                  [criterium/criterium "0.4.3"] ;; benchmarking tool
                                  [clatrix "0.5.0" :exclusions [net.mikera/core.matrix]]] ;; alternate core.matrix implementation
                   :source-paths ["src" "test"]
                   :java-source-paths ["test"]}

             :test {:dependencies [[net.mikera/cljunit "0.4.0"]
                                   [criterium/criterium "0.4.3"]
                                   [clatrix "0.5.0" :exclusions [net.mikera/core.matrix]]
               ]
                    :source-paths ["src" "test"]
                    :java-source-paths ["test"]
                    :main cortex.run-all-tests}

             :cljs {:dependencies [[org.clojure/clojurescript "1.7.228" :scope "provided"]
                                   ;[net.unit8/fressian-cljs "0.2.0"]
                                   [doo "0.1.6-SNAPSHOT"]
                                   [thinktopic/aljabr "0.1.0-SNAPSHOT"]]

                    :plugins [[lein-cljsbuild "1.1.2"]]

                    :cljsbuild {:builds [{:id :test
                                          :source-paths ["src" "test"]
                                          :compiler     {:output-to "resources/test/unit-tests.js"
                                                         :output-dir "resources/test/out"
                                                         :asset-path "out"
                                                         :optimizations :none
                                                         :main 'cortex.test
                                                         :pretty-print true}}]

                                :test-commands {"unit-tests"   ["phantomjs"
                                                                "resources/test/runner.js"
                                                                "resources/test/unit-tests.js"]}}}}
  :resource-paths ["resources"]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"])
