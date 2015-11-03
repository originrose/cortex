(ns fmri.server
  (:require [fmri.handler :refer [app]]
            [environ.core :refer [env]]
            [org.httpkit.server :as http-kit])
  (:gen-class))

(def server* (atom nil))

 (defn -main [& args]
   (let [port (Integer/parseInt (or (env :port) "3000"))]
     (http-kit/run-server app {:port port})))

(defn stop-server []
  (when @server*
    (@server*)))

