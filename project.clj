(defproject thinktopic/cortex "0.9.0-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :url "https://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]

                 [net.mikera/core.matrix "0.57.0"]
                 [net.mikera/vectorz-clj "0.45.0"]
                 [com.github.fommil.netlib/all "1.1.2" :extension "pom"]
                 [com.indeed/util-mmap "1.0.20"]
                 [com.github.ben-manes.caffeine/caffeine "2.3.1"]
                 [com.taoensso/nippy "2.12.2"]

                 ;; Change the following dep to depend on different versions of CUDA
                 ;[org.bytedeco.javacpp-presets/cuda "7.5-1.2"]
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]

                 ;;The dataset abstraction uses parallel and optionally resource management.
                 [thinktopic/think.parallel "0.3.5"]
                 [thinktopic/think.datatype "0.3.10"]

                 [thinktopic/resource "1.1.0"]
                 [thinktopic/think.image "0.4.6"]
                 [org.clojure/math.combinatorics "0.1.4"]]

  :java-source-paths ["java"]

  :profiles {:dev {:source-paths ["src" "test/cljc" "test/clj"]}
             :test {:source-paths ["src" "test/cljc" "test/clj"]}
             :cpu-only {:test-selectors {:default (complement :gpu)}
                        :source-paths ["src" "test/cljc" "test/clj"]}}
  :plugins [[lein-codox "0.10.2"]])
