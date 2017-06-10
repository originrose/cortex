(defproject model-upgrader "0.9.10-SNAPSHOT"
  :description "Upgrade a cortex model to the most recent version."
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.taoensso/nippy "2.13.0"]
                 [thinktopic/think.datatype "0.3.10"]
                 ;;Last version of vectorz library that we saved to the file.
                 [net.mikera/vectorz-clj "0.45.0"]]

  :main model-upgrader.core)
