(defproject thinktopic/cortex "0.3.1-SNAPSHOT"
  :description "A neural network toolkit for Clojure."
  :url "https://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/think.datatype "0.1.0"]
                 [net.mikera/vectorz-clj "0.45.0"]
                 [net.mikera/core.matrix "0.57.0"]
                 [thinktopic/think.parallel "0.3.4"]
                 [thinktopic/resource "1.1.0"]]

  :java-source-paths ["java"])
