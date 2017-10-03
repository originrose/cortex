(defproject thinktopic/cortex "0.9.23-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :url "https://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 [thinktopic/think.datatype "0.3.17"]
                 [com.github.fommil.netlib/all "1.1.2" :extension "pom"]
                 [com.taoensso/nippy "2.13.0"]
                 ;; Change the following dep to depend on different versions of CUDA
                 ;[org.bytedeco.javacpp-presets/cuda "7.5-1.2"]
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 [thinktopic/think.parallel "0.3.7"]
                 [thinktopic/think.resource "1.2.1"]
                 [org.clojure/math.combinatorics "0.1.4"]]

  :java-source-paths ["java"]

  :profiles {:dev {:source-paths ["src" "test/cljc" "test/clj"]}
             :test {:source-paths ["src" "test/cljc" "test/clj"]}
             :cpu-only {:test-selectors {:default (complement :gpu)}
                        :source-paths ["src" "test/cljc" "test/clj"]}}
  :plugins [[lein-codox "0.10.2"]])
