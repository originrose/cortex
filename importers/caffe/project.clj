(defproject thinktopic/cortex-caffe "0.9.12-SNAPSHOT"
  :description "Caffe support for cortex."
  :url "http://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 [thinktopic/hdf5 "0.2.1"]
                 [thinktopic/cortex "0.9.12-SNAPSHOT"]
                 [instaparse "1.4.3"]
                 [com.taoensso/nippy "2.12.2"]]
  :profiles {:ci {:dependencies [[thinktopic/hdf5 "0.1.3"]]}}
  :plugins [[lein-codox "0.10.2"]]
  :test-selectors {:ci (fn [m] (not (:skip-ci m)))})
