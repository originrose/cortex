(ns thinktopic.cortex.gui
  (:use [thinktopic.cortex.lab core]
        ;[task core]
        ;[incanter core stats charts]
        )
  (:require [incanter core charts]
            [mikera.image.core :as image]
            [mc ui image]
            [clojure.core.matrix :as mat])
  (:import [javax.swing JFrame JComponent JLabel JPanel]
           [java.awt Graphics2D Color GridLayout Component Dimension]
           [java.awt.event ActionEvent ActionListener]
           [java.awt.image BufferedImage]
           [nuroko.module ALayerStack AWeightLayer]
           [nuroko.core IComponent Components IThinker]
           [mikera.gui Frames JIcon]
           [mikera.util Maths]
           [mikera.vectorz AVector Vectorz Vector Vector2]
           [org.jfree.chart ChartPanel JFreeChart]))

;; This comes originally from Mike Anderson's Nurokit

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

;; =============================================
;; chart display functions

(def window-count (atom 1))

(defn show-chart
  ([incanter-chart]
    (show-chart incanter-chart (str "Chart" (swap! window-count inc))))
  ([incanter-chart title]
    (mc.ui/show-component (ChartPanel. incanter-chart) title)))

;; =============================================
;; example code

(comment
  ; test chart
  (view (function-plot sin -10 10))
  )

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

(declare component)
(declare grid)

;; Colour functions

(defmacro clamp-colour-value [val]
  `(let [v# (float ~val)]
     (Math/min (float 1.0) (Math/max (float 0.0) v#))))

(defn weight-colour
  ([^double weight]
    (Color.
      (clamp-colour-value (Math/tanh (- weight)))
      (clamp-colour-value (Math/tanh weight))
      0.0)))

(defn weight-colour-rgb
  (^long [^double weight]
    (mc.image/rgb
      (clamp-colour-value (Math/tanh (- weight)))
      (clamp-colour-value (Math/tanh weight))
      0.0)))

(defn weight-colour-mono
  (^long [^double weight]
    (let [v (Maths/logistic (double weight))]
      (mc.image/rgb v v v))))

(defn mono-rgb
  (^long [^double cv]
    (mc.image/rgb cv cv cv)))


(defn sigmoid-rgb
  (^long [^double cv]
    (let [s (Maths/logistic cv)]
      (mc.image/rgb s s s))))

(defn activation-colour
  ([^double x]
    (Color.
      (clamp-colour-value x)
      (clamp-colour-value (Math/abs x))
      (clamp-colour-value (- x)))))


;; image builder

(defn image-generator
  "Creates an image generator that builds images from vector data"
  ([& {:keys [offset width height skip colour-function]
       :or {offset 0}}]
    (let [width (int width)
          height (int height)
          offset (int offset)
          skip (int (or skip width))
          colour-function (or colour-function mono-rgb)
          size (* width height)]
      (fn [v]
		      (let [v? (instance? AVector v)
                ^AVector v (if v? v (thinktopic.cortex.lab.core/avector v))
                ^BufferedImage bi (mc.image/buffered-image width height)
                ^ints data (int-array size)]
	         (dotimes [y height]
	           (let [yo (int (* y width))
                   vo (int (* y skip))]
              (dotimes [x width]
                (aset data (+ x yo) (let [cv (.get ^AVector v (int (+ vo x)))]
                                      (int (colour-function cv)))))))
	         (.setDataElements (.getRaster bi) (int 0) (int 0) width height data)
	         bi)))))

(defn img
  "Function to create an image from a vector. Defaults to fitting to a square"
  ([^AVector vector & {:keys [width colour-function zoom]}]
    (let [vlen (.length vector)
          width (long (or width (Math/sqrt vlen)))
          height (quot vlen width)
          colour-function (or colour-function weight-colour-rgb)
          im ((image-generator :width width :height height :colour-function colour-function) vector)]
      (if zoom
        (image/resize im (* width zoom))
        im))))

(defn spatio [vs & {:keys [colour-function] }]
  (let [vs (mapv (fn [v] (if (instance? AVector v) v (thinktopic.cortex.lab.core/avector v))) vs)
        colour-function (or colour-function mono-rgb)
        width (.length ^AVector (nth vs 0))
        height (count vs)
        ^BufferedImage bi (mc.image/buffered-image width height)
        ^ints data (int-array (* width height))]
    (dotimes [y height]
      (let [v (nth vs y)
            yo (int (* y width))]
        (dotimes [x width]
          (aset data (+ x yo) (let [cv (.get ^AVector v (int x))]
                                (int (colour-function cv)))))))
    (.setDataElements (.getRaster bi) (int 0) (int 0) width height data)
	  bi))

(defn label
  "Creates a JLabel with the given content"
  (^JLabel [s]
    (let [^String s (str s)
          label (JLabel. s JLabel/CENTER)]
      (.setToolTipText label s)
      label)))


(defn component
  "Creates a component as appropriate to visualise an object x"
  (^JComponent [x]
    (cond
      (instance? JComponent x) x
      (instance? BufferedImage x) (JIcon. ^BufferedImage x)
	    (instance? JFreeChart x) (ChartPanel. ^JFreeChart x)
      (sequential? x) (grid (seq x))
      :else (label x))))

(defn grid [things]
  (let [n (count things)
        size (int (Math/ceil (Math/sqrt n)))
        grid-layout (GridLayout. 0 size)
        grid (JPanel.)]
    (.setLayout grid grid-layout)
    (doseq [x things]
      (.add grid (component x)))
    grid))


(defn show
  "Shows a component in a new frame"
  ([com
    & {:keys [^String title]
       :as options
       :or {title nil}}]
  (let [com (component com)]
    (Frames/display com (str title)))))


(defn default-dimensions
  "Returns the default dimensions for a new frame"
  (^java.awt.Dimension []
    (java.awt.Dimension. 400 300)))


(defn network-graph
  ([nn
    & {:keys [border repaint-speed activation-size line-width max-nodes-displayed]
       :or {border 20
            repaint-speed 50
            line-width 1
            activation-size 5}}]
    (let [^ALayerStack nn (Components/asLayerStack nn)
          graph (proxy [javax.swing.JComponent java.awt.event.ActionListener] []
        (actionPerformed [^ActionEvent e]
          (.repaint ^JComponent this))
        (paintComponent [^Graphics2D g]
          (let [border (double border)
                this ^JComponent this
                width (double (.getWidth this))
                height (double (.getHeight this))
                layers (.getLayerCount nn)
                sizes (vec (cons (input-length nn) (map #(.getOutputLength ^AWeightLayer %) (.getLayers nn))))
                sizes (if max-nodes-displayed (mapv #(min max-nodes-displayed %) sizes) sizes)
                max-size (reduce max sizes)
                step (/ (double width) max-size)
                as (double activation-size)]
            (.setColor g (Color/BLACK))
            (.fillRect g 0 0 width height)
            (.setStroke g (java.awt.BasicStroke. (float line-width)))
            (dotimes [i layers]
              (let [layer (.getLayer nn i)
                    layer-inputs (long (if max-nodes-displayed (sizes i) (.getInputLength layer)))
                    layer-outputs (long (if max-nodes-displayed (sizes (inc i)) (.getOutputLength layer)))
                    sy (int (+ border (* (- height (* 2 border)) (/ (- layers 0.0 i) layers))))
                    ty (int (+ border (* (- height (* 2 border)) (/ (- layers 1.0 i) layers))))
                    soffset (double border)
                    toffset (double border)
                    sskip (double (/ (- width (* 2 border)) (max 1.0 (dec layer-inputs))))
                    tskip (double (/ (- width (* 2 border)) (max 1.0 (dec layer-outputs))))]
                (dorun (for [y (sort-by #(hash (* 0.23 %)) (range layer-outputs))] ;; random order of drawing
                  (let [link-count (.getLinkCount layer y)
                        y (int y)
                        tx (int (+ toffset (* tskip y)))]
                    (dotimes [ii link-count]
	                    (let [x (.getLinkSource layer y ii)
                            sx (int (+ soffset (* sskip x)))
                            ii (int ii)]
	                      (when (< x layer-inputs)
                          (.setColor g ^Color (weight-colour (double (.getLinkWeight layer y ii))))
	                        (.drawLine g sx sy tx ty)))))))))
            (dotimes [i (inc layers)]
              (let [data ^AVector (.getData nn i)
                    len (sizes i)
                    ty (int (+ border (* (- height (* 2 border)) (/ (- layers i) layers))))
                    toffset (double border)
                    tskip (double (/ (- width (* 2 border)) (max 1.0 (dec len))))]
                (dotimes [y len]
                  (let [activation (.get data y)
                        tx (int (+ toffset (* tskip y)))]
                    (.setColor g ^Color (activation-colour activation))
                    (.fillRect g (- tx as) (- ty as) (* 2 as) (* 2 as))
                    (.setColor g Color/GRAY)
                    (.drawRect g (- tx as) (- ty as) (* 2 as) (* 2 as)))))))))
          timer (javax.swing.Timer. (int repaint-speed) graph)]
      (.start timer)
      (.setPreferredSize graph (default-dimensions))
      graph)))

(defn xy-chart ^JFreeChart [xs ys]
  (let [chart (incanter.charts/xy-plot xs ys)]
    ; (incanter.charts/set-y-range chart 0.0 1.0)
    chart))

(defn xy-chart-multiline ^JFreeChart [xs yss]
  (let [chart (xy-chart xs (first yss))]
    (doseq [ys (rest yss)]
      (incanter.charts/add-lines chart xs ys))
    chart))

(defn time-chart
  "Creates a continously updating time chart of one or more calculations, which should be functions with zero arguments."
  ([calcs
    & {:keys [repaint-speed time-periods y-min y-max]
       :or {repaint-speed 250
            time-periods 1200}}]
    (let [line-count (count calcs)
          start-millis (System/currentTimeMillis)
          times (atom '())
          values (atom (repeat line-count '()))
          next-chart  (fn []
                       (let [time (/ (- (System/currentTimeMillis) start-millis) 1000.0)]
                         (swap! times #(take time-periods (cons time %)))
                         (swap! values #(for [[calc ss] (map vector calcs %)]
                                         (take time-periods (cons (calc) ss))))
                         (let [chart (xy-chart-multiline @times @values)]
                           (if y-max (incanter.charts/set-y-range chart (double (or y-min 0.0)) (double y-max)))
                           chart)))
          panel (ChartPanel. ^JFreeChart (next-chart))
          timer (javax.swing.Timer.
                  (int repaint-speed)
                  (proxy [java.awt.event.ActionListener] []
                    (actionPerformed
                      [^ActionEvent e]
                      (when (.isShowing panel)
                        (.setChart panel ^JFreeChart (next-chart))
                        (.repaint ^JComponent panel)))))]
      (.start timer)
      (.setPreferredSize panel (default-dimensions))
      panel)))


(defn scatter-outputs
  "Shows a scatter graph of 2 output values."
  ([data
    & {:keys [labels x-index y-index]
       :or {x-index 0
            y-index 1}}]
    (let [res (map (fn [^mikera.vectorz.AVector v] [(.get v (int x-index)) (.get v (int y-index))]) data)
          labels labels
          xs (map first res)
          ys (map second res)
          scatter-chart (incanter.charts/scatter-plot xs ys :group-by labels)
          panel (ChartPanel. ^JFreeChart scatter-chart)]
     panel)))



(defn layer-feature-calc ^AVector [^nuroko.module.AWeightLayer wl ^AVector out-weights]
  (let [ol (output-length wl)
        in-weights ^AVector (Vectorz/newVector (input-length wl))]
    (dotimes [i ol]
      (let [i (int i)
            y (double (.get out-weights i))]
        (.addMultiple in-weights (.getSourceWeights wl i) (.getSourceIndex wl i) y)))
    in-weights))

(defn stack-feature-calc ^AVector [stack ^AVector out-weights
                                   & {:keys []}]
  (let [stack (Components/asLayerStack stack)
        lc (.getLayerCount stack)]
    (loop [i (dec lc)
           output out-weights]
      (if (< i 0)
        output
        (recur (dec i) (layer-feature-calc (.getLayer stack i) output))))))

(defn feature-maps
  "Returns feature map vectors for an AThinkStack"
  ([stack
    & {:keys [scale max-layers]
       :or {scale 1.0}}]
    (let [stack (Components/asLayerStack stack)
          ol (output-length stack)
          scale (double scale)]
      (for [i (range ol)]
        (let [top-vector (Vectorz/axisVector (int i) ol)
              result (stack-feature-calc stack top-vector)]
          (.multiply result scale)
          result)))))

(defn spatio-map
  ([stack
    examples]
    (let [stack (.clone ^IComponent stack)
          ig (image-generator :width (output-length stack)
                              :height (count examples))
          ]
      (ig (Vectorz/join ^java.util.List (vec (map (partial think stack) examples)))))))

;; class separation chart
(defn class-point
  "Turns a classification vector into a point on the unit circle."
  ([^AVector v]
    (let [n (.length v)
          v (.clone v)
          vsum (double (mat/esum v))
          ^AVector v (if (== vsum 0) v (mat/scale v (/ 1.0 vsum)))
          r (Vectorz/newVector 2)]
      (dotimes [i n]
        (let [d (.get v (int i))
              a (/ (* i (* 2.0 Math/PI)) n)]
          (mat/add! r (Vector2/of (* d (Math/sin a)) (* d (Math/cos a))))))
      r)))

(defn class-separation-chart
  "Shows a chart of class separation with each class pulled to a different point on the unit circle"
  ([^IThinker t inputs classes]
    (let [t (.clone t)
          cps (map #(class-point (think t %)) inputs)]
      (incanter.charts/scatter-plot (map first cps) (map second cps) :group-by classes))))

(defn vector-bars
  "Draws a bar chart of the values in a vector"
  ([v]
    (let [n (mat/ecount v)]
      (incanter.charts/bar-chart (range n) (mat/eseq v)))))

;; DEMO CODE

(defn demo []
  (def nn (neural-network :inputs 10 :outputs 3))
  (show (network-graph nn))
  (show (time-chart [#(Math/random)])))

;(demo)
