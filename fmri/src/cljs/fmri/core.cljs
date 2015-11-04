(ns fmri.core
  (:require
    [reagent.core :as reagent :refer [atom]]
    [reagent.session :as session]
    [secretary.core :as secretary :include-macros true]
    [goog.events :as events]
    [goog.history.EventType :as EventType]
    [accountant.core :as accountant]
    [chord.client :refer [ws-ch]]
    [fressian-cljs.core :as fressian]
    [fressian-cljs.reader :as freader]
    [chord.format :as cf]
    [chord.format.fressian]
    [cljs.core.async :refer [<! >!] :as async])
  (:require-macros
    [cljs.core.async.macros :refer [go]]))

(enable-console-print!)

(def SERVER-URL "ws://localhost:3000/fmri/jack-in")
(def conn* (atom nil))

(defmethod cf/formatter* :fressian [_]
  (reify cf/ChordFormatter
    (freeze [_ obj] (fressian/write obj))
    (thaw [_ s] 
      (println "incoming fressian object:")
      (js/console.log s)
      (fressian/read s :handlers 
                     (assoc fressian/cljs-read-handler "array"
                            (fn [reader tag component-count]
                              (println "component-count: " component-count)
                              (let [shape (freader/read-object reader)
                                    size (apply + shape)]
                                (println "shape: " shape)
                                shape)))))))


(defn read-array
  [obj]
  (println "read-array")
  (let [shape (freader/read-object obj)]
    (println "SHAPE: " shape)
    shape))

(defn home-page []
  (let [msg (atom "")]
    (go
      (let [conn (or @conn* (<! (ws-ch SERVER-URL {:format :fressian})))
            {:keys [ws-channel error]} conn
            _ (>! ws-channel {:foo 2 :bar "baz" :baz [1 23 45]})
            {:keys [message error]} (<! ws-channel)]
        (if error
          (do
            (println "ERROR:")
            (js/console.log error))
          (do
            (println "MSG:")
            (js/console.log message)
            (js/console.log (type message))
            (reset! msg (read-array message))))
        (reset! conn* conn)))
    (fn []
      [:div [:h2 "Welcome to fmri"]
       [:p "MESSAGE: " @msg]
       [:div [:a {:href "#/about"} "go to about page"]]])))

(defn about-page []
  [:div [:h2 "About fmri"]
   [:div [:a {:href "#/"} "go to the home page"]]])

(defn current-page []
  [:div [(session/get :current-page)]])

(secretary/set-config! :prefix "#")

(secretary/defroute "/" []
  (session/put! :current-page #'home-page))

(secretary/defroute "/about" []
  (session/put! :current-page #'about-page))

(defn mount-root []
  (reagent/render [current-page] (.getElementById js/document "app")))

(defn init! []
  (accountant/configure-navigation!)
  (accountant/dispatch-current!)
  (mount-root))
