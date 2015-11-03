(ns fmri.handler
  (:require [compojure.core :refer [GET defroutes]]
            [compojure.route :refer [not-found resources]]
            [ring.middleware.defaults :refer [site-defaults wrap-defaults]]
            [hiccup.core :refer [html]]
            [hiccup.page :refer [include-js include-css]]
            [prone.middleware :refer [wrap-exceptions]]
            [ring.middleware.reload :refer [wrap-reload]]
            [environ.core :refer [env]]
            [chord.http-kit :refer [with-channel]]
            [chord.format.binary]
            [clojure.core.async :refer [<! >! go] :as async]))

(defn home-page []
  (html
   [:html
    [:head
     [:meta {:charset "utf-8"}]
     [:meta {:name "viewport"
             :content "width=device-width, initial-scale=1"}]
     (include-css (if (env :dev) "css/site.css" "css/site.min.css"))]
    [:body 
     [:div#app
      [:h3 "ClojureScript has not been compiled!"]
      [:p "please run "
       [:b "lein figwheel"]
       " in order to start the compiler"]]
     (include-js "js/app.js")]]))

(defn jack-in
  [req]
  (with-channel req ws-ch {:format :binary}
    (let [{:keys [message]} (<! ws-ch)]
      (prn "Message received:" message)
      (>! ws-ch "Hello client from server!")
      (async/close! ws-ch))))

(defroutes routes
  (GET "/" [] (home-page))
  (GET "/fmri/jack-in" [] (jack-in))
  (resources "/")
  (not-found "Not Found"))

(def app
  (let [handler (wrap-defaults #'routes site-defaults)]
    (if (env :dev) (-> handler wrap-exceptions wrap-reload) handler)))

