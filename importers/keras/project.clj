(defproject thinktopic/cortex-keras "0.9.3-SNAPSHOT"
  :description "Import of keras models into cortex descriptions"
  :url "http://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/hdf5 "0.1.2"]
                 [thinktopic/cortex "0.9.3-SNAPSHOT"]
                 [cheshire "5.6.3"]
                 [net.mikera/imagez "0.10.0"]]
  :plugins [[s3-wagon-private "1.1.2"]
            [lein-codox "0.10.2"]]
  :repositories  {"snapshots"  {:url "s3p://thinktopic.jars/snapshots/"
                                :passphrase :env
                                :username :env
                                :releases false}
                  "releases"  {:url "s3p://thinktopic.jars/releases/"
                               :passphrase :env
                               :username :env
                               :snapshots false
                               :sign-releases false}})
