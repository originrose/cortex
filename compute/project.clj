(defproject thinktopic/compute "0.5.0-SNAPSHOT"
  :description "Compute abstraction and cpu implementation.  Meant to abstract things like openCL and CUDA usage."
  :url "http://thinktopic.com"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/think.datatype "0.3.7"]
                 [thinktopic/cortex "0.5.0-SNAPSHOT"]
                 [thinktopic/cortex-datasets "0.5.0-SNAPSHOT"]
                 [com.github.fommil.netlib/all "1.1.2" :extension "pom"]]
  :java-source-paths ["java"]
  :plugins [[lein-codox "0.10.2"]])
