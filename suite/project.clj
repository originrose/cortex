(defproject thinktopic/cortex.suite "0.3.1-SNAPSHOT"
  :description "Full pipelines for building various models bringing together multiple cortex and clojure libraries."
  :url "http://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/gpu-compute "0.3.1-SNAPSHOT"]
                 [thinktopic/think.image "0.4.2"]
                 [com.taoensso/nippy "2.12.2"]
                 [garden "1.3.2"]]
  :resource-paths ["cljs"])
