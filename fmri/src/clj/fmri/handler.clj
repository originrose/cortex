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
            [chord.format :as cf]
            [clojure.core.async :refer [<! >! go] :as async]
            [clojure.core.matrix :as mat]
            [thinktopic.matrix.fressian :as mfress]
            [clojure.data.fressian :as fressian])
  (:import [org.fressian.handlers WriteHandler ReadHandler]
           [org.fressian.impl ByteBufferInputStream BytesOutputStream]))

(mat/set-current-implementation :vectorz)

(defmethod cf/formatter* :fressian [_]
  (reify cf/ChordFormatter
    (freeze [_ obj]
      (println "writing fressian obj: " (type obj))
      (ByteBufferInputStream. 
        (fressian/write obj :handlers (mfress/array-write-handlers mikera.arrayz.impl.AbstractArray))))

    (thaw [_ s]
      (fressian/read s))))

(defn jack-in
  [req]
  (println "\njack-in request:")
  (with-channel req ws-ch {:format :fressian}
    (go
      (let [{:keys [message error]} (<! ws-ch)
            ary (mat/array [[1 2 3] [1 2 3]])]
        (if error
          (prn "Error:" error)
          (prn "Message:" message))
        ;(>! ws-ch "This is a test...")
        (>! ws-ch ary)
        ;(async/close! ws-ch)
        ))))

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

(defroutes routes
  (GET "/" [] (home-page))
  (GET "/fmri/jack-in" req (jack-in req))
  (resources "/")
  (not-found "Not Found"))

(def app
  (let [handler (wrap-defaults #'routes site-defaults)]
    (if (env :dev) (-> handler wrap-exceptions wrap-reload) handler)))

