(defproject thinktopic/compute "0.3.1-SNAPSHOT"
  :description "Compute abstraction and cpu implementation.  Meant to abstract things like openCL and CUDA usage."
  :url "http://thinktopic.com"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/core.async "0.2.391"]
                 [thinktopic/datatype "0.1.0"]
                 [thinktopic/cortex "0.3.1-SNAPSHOT"]
                 [thinktopic/cortex-datasets "0.3.1-SNAPSHOT"]]
  :java-source-paths ["java"]
  )
