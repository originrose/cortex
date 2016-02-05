(defproject thinktopic/cortex "0.1.0-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.taoensso/timbre "4.2.1"]
                 [net.mikera/core.matrix "0.45.0-CLJS-SNAPSHOT"]
                 [net.mikera/vectorz-clj "0.43.0"]
                 [org.clojure/test.check "0.9.0"]
                 [thinktopic/matrix.fressian "0.2.1"]
                 [com.google.protobuf/protobuf-java "2.6.1"]

                 ;; cljs
                 [org.clojure/clojurescript "1.7.228" :scope "provided"]
                 [net.unit8/fressian-cljs "0.2.0"]
                 [doo "0.1.6-SNAPSHOT"]
                 [thi.ng/ndarray "0.3.1-SNAPSHOT"]
                 [caffe-protobuf "0.1.0"]]

  :profiles {:dev {:dependencies [[net.mikera/cljunit "0.3.1"]
                                  [criterium/criterium "0.4.3"]
                                  [clatrix "0.5.0" :exclusions [net.mikera/core.matrix]]
                                  ]
                   :java-source-paths ["test"]}}

  :plugins [[lein-cljsbuild "1.1.2"]]

  :cljsbuild {;;You can't use none optimizations in general because then the files reference files under out/
              :builds [{:id :test
                        :source-paths ["src" "test"]
                        :compiler     {:output-to     "resources/test/unit-tests.js"
                                       :optimizations :none
                                       :main          cortex.test
                                       :pretty-print  true}}]

              :test-commands {"unit-tests"   ["phantomjs"
                                              "resources/test/runner.js"
                                              "resources/test/unit-tests.js"]}}
  :source-paths ["src" "test"]

  :resource-paths ["resources"]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"]

  :main cortex.run-all-tests)
