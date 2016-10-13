(defproject thinktopic/compute "0.1.0-SNAPSHOT"
  :description "Compute abstraction and cpu implementation.  Meant to abstract things like openCL and CUDA usage."
  :url "http://thinktopic.com"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/core.async "0.2.391"]
                 [thinktopic/resource "1.1.0"]
                 [thinktopic/cortex "0.2.1-SNAPSHOT"]
                 [thinktopic/cortex-datasets "0.3.0-SNAPSHOT"]]
  :java-source-paths ["java"]

  :plugins [[s3-wagon-private "1.1.2"]]
  :repositories  {"snapshots"  {:url "s3p://thinktopic.jars/snapshots/"
                                :passphrase :env
                                :username :env
                                :releases false}
                  "releases"  {:url "s3p://thinktopic.jars/releases/"
                               :passphrase :env
                               :username :env
                               :snapshots false
                               :sign-releases false}}
  )
