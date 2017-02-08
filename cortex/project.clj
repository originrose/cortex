(defproject thinktopic/cortex "0.5.1-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :url "https://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/vectorz-clj "0.45.0"]
                 [net.mikera/core.matrix "0.57.0"]
                 ;;The dataset abstraction uses parallel and optionally resource management.
                 [thinktopic/think.parallel "0.3.4"]
                 [thinktopic/resource "1.1.0"]]

  :profiles {:dev {:source-paths ["src" "test/cljc" "test/clj"]}
             :test {:source-paths ["src" "test/cljc" "test/clj"]}}

  :plugins [[lein-codox "0.10.2"]])
