(defproject thinktopic/cortex "0.3.1-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :url "https://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.taoensso/timbre "4.3.1"]
                 [org.clojure/test.check "0.9.0"]
                 [thinktopic/matrix.fressian "0.3.1"]
                 [net.mikera/vectorz-clj "0.45.0"]
                 [net.mikera/core.matrix "0.57.0"]
                 [net.mikera/clojure-utils "0.7.1"]
                 [com.github.fommil.netlib/all "1.1.2" :extension "pom"]
                 [thinktopic/think.parallel "0.3.4"]
                 [thinktopic/resource "1.1.0"]]

  :java-source-paths ["java"]

  :profiles {:dev {:dependencies [[net.mikera/cljunit "0.6.0"]  ;; allows JUnit testing
                                  [criterium/criterium "0.4.4"] ;; benchmarking tool
                                  ] ;; alternate core.matrix implementation
                   :source-paths ["src" "test/cljc" "test/clj"]
                   :java-source-paths ["test/clj"]}

             :test {:dependencies [[net.mikera/cljunit "0.6.0"]
                                   [criterium/criterium "0.4.4"]
                                   [clatrix "0.5.0" :exclusions [net.mikera/core.matrix]]]
                    :source-paths ["src" "test/cljc" "test/cljs" "test/clj"]
                    :java-source-paths ["test/clj"]
                    :main cortex.run-all-tests}

             :cljs {:dependencies [[org.clojure/clojurescript "1.7.228" :scope "provided"]
                                   [doo "0.1.6"]
                                   [thinktopic/aljabr "0.1.1"]]

                    :plugins [[lein-cljsbuild "1.1.2"]]

                    :cljsbuild {:builds [{:id :test
                                          :source-paths ["src" "test/cljc" "test/cljs"]
                                          :compiler     {:output-to "target/js/unit-tests.js"
                                                         :output-dir "target/js/out"
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
              "-XX:-OmitStackTraceInFastThrow"]

  )
