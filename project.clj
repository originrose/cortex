(defproject thinktopic/cortex "0.1.0-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.taoensso/timbre "4.2.1"]
                 [net.mikera/vectorz-clj "0.41.0"]
                 [org.clojure/test.check "0.9.0"]
                 [thinktopic/matrix.fressian "0.2.1"]

                 ;; cljs
                 [org.clojure/clojurescript "1.7.228" :scope "provided"]
                 [net.unit8/fressian-cljs "0.2.0"]
                 [doo "0.1.6-SNAPSHOT"]
                 [thi.ng/ndarray "0.3.1-SNAPSHOT"]]

  :profiles {:dev {:dependencies [[net.mikera/cljunit "0.3.1"]]
                   :java-source-paths ["test"]}}

  :plugins [[lein-cljsbuild "1.1.2"]]

  :cljsbuild {;;You can't use none optimizations in general because then the files reference files under out/
              :builds {
                       :test {
                              :source-paths ["src" "test"]
                              :compiler     {:output-to     "resources/test/unit-tests.js"
                                             :optimizations :whitespace
                                             :main          cortex.test
                                             :pretty-print  true}}}

              :test-commands {"unit-tests"   ["phantomjs"
                                              "resources/test/runner.js"
                                              "resources/test/unit-tests.js"]}}

  :resource-paths ["resources"]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"])
