(defproject thinktopic/compute "0.4.0"
  :description "Compute abstraction and cpu implementation.  Meant to abstract things like openCL and CUDA usage."
  :url "http://thinktopic.com"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/think.datatype "0.3.7"]
                 [thinktopic/cortex "0.4.0"]
                 [thinktopic/cortex-datasets "0.4.0"]
                 [com.github.fommil.netlib/all "1.1.2" :extension "pom"]]
  :java-source-paths ["java"]
  :plugins [[lein-codox "0.10.2"]])
