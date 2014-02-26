(ns diabolo.letters
  (:use [mikera.image core colours])
  (:import [java.awt GraphicsEnvironment Font Color]
           [java.awt.image BufferedImage]))

(defn font-names
  []
  (let [ge (GraphicsEnvironment/getLocalGraphicsEnvironment)]
    (.getAllFonts ge)
    (vec (.getAvailableFontFamilyNames ge))))

(def FONT-STYLES
  {:plain  Font/PLAIN
   :italic Font/ITALIC
   :bold   Font/BOLD})

(defn font
  [& {:keys [name size style]
      :or {size 12
           style :plain}}]
  (Font. name (FONT-STYLES style) size))

(defn str-size
  [font txt]
  (let [img (new-image 1 1)
        g (.getGraphics img)
        _ (.setFont g font)
        metrics (.getFontMetrics g)]
    {:width (.stringWidth metrics txt)
     :height (.getHeight metrics)
     :ascent (.getMaxAscent metrics)
     :descent (.getMaxDescent metrics)
     :font font}))


(defn ^BufferedImage draw-str
  [^BufferedImage image ^Font font ^String txt color x y]
  (let [g (.getGraphics image)]
    (.setPaint g color)
    (.setFont g font)
    (.drawString g txt x y)
    image))


(defn char-frame
  [width height margin font-name c]
  (let [img (new-image width height)
        w (- width margin)
        h (- height margin)
        fonts (map #(font :name font-name :size %) (iterate dec 100))
        sizes (map #(str-size % c) fonts)
        fit-size (first (filter (fn [{:keys [width height]}]
                                  (and (< width w) (< height h)))
                                sizes))
        font (:font fit-size)
        y (:ascent fit-size)]
    (draw-str img font c (Color/black) margin y)))


(defn alpha-set
  [width height margin font]
  (map #(char-frame width height margin font (str (char %))) (range (int \A) (int \z))))


(defn dataset
  []
  (mapcat #(alpha-set 28 28 0 %) (font-names)))


