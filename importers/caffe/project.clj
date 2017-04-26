(defproject thinktopic/cortex-caffe "0.9.4"
  :description "Caffe support for cortex."
  :url "http://github.com/thinktopic/cortex"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/hdf5 "0.1.2"]
                 [thinktopic/cortex "0.9.4"]
                 [instaparse "1.4.3"]
                 [com.taoensso/nippy "2.12.2"]]
  :plugins [[lein-codox "0.10.2"]])
