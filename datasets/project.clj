(defproject thinktopic/cortex-datasets "0.3.1-SNAPSHOT"
  :description "Library to provide datasets and to encapsulate persistent storage for cortex."
  :url "https://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/vectorz-clj "0.43.1"]
                 [net.mikera/core.matrix "0.50.0"]
                 [thinktopic/resource "1.1.0"]
                 [com.indeed/util-mmap "1.0.20"]
                 [com.github.ben-manes.caffeine/caffeine "2.3.1"]])
