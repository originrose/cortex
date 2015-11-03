(ns fmri.repl
  (:require 
    [org.httpkit.server :as http-kit]
    [fmri.handler :refer [app]]
    [ring.middleware.reload :refer [wrap-reload]]
    [ring.middleware.file-info :refer [wrap-file-info]]
    [ring.middleware.file :refer [wrap-file]]))

(defonce server* (atom nil))

(defn get-handler []
  (-> #'app
      (wrap-file "resources")

      ; Content-Type, Content-Length, and Last Modified headers for files in body
      (wrap-file-info)))

(defn start-server
  "used for starting the server in development mode from REPL"
  [& [port]]
  (let [port (if port (Integer/parseInt port) 3000)]
    (reset! server* (http-kit/run-server (get-handler) {:port port}))
    (println (str "Go to http://localhost:" port))))

(defn stop-server []
  (@server*)
  (reset! server* nil))

