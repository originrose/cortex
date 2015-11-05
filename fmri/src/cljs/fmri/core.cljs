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
    [cljs.core.async :refer [<! >!] :as async]
    [thi.ng.ndarray.core :as mat])
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
                                    size (apply + shape)
                                    ary  (double-array size)]
                                (println "shape: " shape)
                                (doseq [i (range size)]
                                  (aset ary i (freader/read-double reader)))
                                (mat/ndarray :float64 ary shape))))))))

(defn canvas-element
  []
  (.createElement js/Document "canvas"))

(defn matrix->image
  [m & {:keys [canvas scale min-val max-val]}]
  (let [scale (or scale 1)
        depth 4 ; rgba
        [height width] (mat/shape m)
        img-height (* scale height)
        img-width (* scale width)
        canvas (or canvas (canvas-element))
        _ (set! (.-width canvas) img-width)
        _ (set! (.-height canvas) img-height)
        _ (println "canvas:")
        _ (js/console.log canvas)
        ctx (.getContext canvas "2d")
        img-data (.getImageData ctx 0 0 img-width img-height)
        pixels (.-data img-data)
        min-v (or min-val (apply min m))
        max-v (or max-val (apply max m))
        dv (- max-v min-v)
        c-range (/ 255.0 dv)]
    (println "rendering with scale: " scale)
    (doseq [x (range width)]
      (doseq [y (range height)]
        ; Normalize values betweeen 0-1, then put into color range [0-255].
        (let [pval (* (- (mat/get-at m y x) min-v) c-range)]
          (doseq [sx (range scale)]
            (doseq [sy (range scale)]
              (let [row-offset (* img-width (+ (* y scale) sy))
                    col-offset (+ sx (* x scale))
                    idx (* depth (+ row-offset col-offset))]
                (aset pixels idx pval)            ; r
                (aset pixels (+ idx 1) pval)      ; g
                (aset pixels (+ idx 2) pval)      ; b
                (aset pixels (+ idx 3) 255))))))) ; a
    (.putImageData ctx img-data 0 0)
    canvas))

(defn call-on-mount 
  [element callback]
  (with-meta element 
             {:component-did-mount (fn [this]
                                     (let [node (reagent/dom-node this)]
                                       (println "canvas node:")
                                       (js/console.log node)
                                       (callback node)))}))

(defn matrix-renderer
  [m & {:keys [scale]}]
  (reagent/create-class
    {:component-did-mount
     (fn [this]
       (let [canvas (reagent/dom-node this)
             scale (or scale 1)]
         (matrix->image m :scale scale :canvas canvas)))
     :display-name "Matrix Renderer"
     :reagent-render (fn [] [:canvas])
     }))

(defn get-remote-matrix
  [in-atom]
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
          (println message)
          (reset! in-atom message)))
      (reset! conn* conn))))

(defn cross-matrix
  "Generates a matrix with a cross in it as a test for painting
  matrices."
  [size]
  (let [ary (double-array (* size size))
        m (mat/ndarray :float64 ary [size size])]
    (doseq [i (range (* size size))] (aset m i 0))
    (doseq [i (range size)]
      (mat/set-at m 5 i 1)
      (mat/set-at m i 5 1))
    m))

(defn home-page []
  (let [msg (atom "")
        m (cross-matrix 11)]
    (fn []
      [:div [:h2 "Welcome to fmri"]
       [:p "MESSAGE: " @msg]
       [matrix-renderer m :scale 30]
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
