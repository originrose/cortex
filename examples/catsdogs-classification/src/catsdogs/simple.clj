(ns catsdogs.simple)

(require
  '[cortex.util :as util]
  '[cortex.nn.execute :as execute]
  '[mikera.image.core :as i]
  '[think.image.patch :as patch])

(defn image-file->observation
  "Create an observation from input file."
  [image-path]
{:labels ["test"]
 :data
  (patch/image->patch
    (-> (i/load-image image-path) (i/resize 52 52))
    ; can do without datatype
    :datatype :float
    :colorspace :gray)})

; (def class-mapping
;   {:class-name->index (zipmap ["cat" "dog"] (range))
;    :index->class-name (zipmap (range) ["cat" "dog"])})
; do not need the full mapping
(defn index->class-name[n]
  (nth ["cat" "dog"] n))

(defn guess [nippy image-path]
  (let [obs (image-file->observation image-path)]
    (-> (execute/run nippy [obs])
        first
        :labels
        util/max-index
        index->class-name)))

(defn guesses [nippy image-paths]
  (let[obs (map #(image-file->observation %) image-paths) ]
  (map #(index->class-name (util/max-index (:labels %))) (execute/run nippy (into-array obs)))))

(comment

  (def nippy
    (util/read-nippy-file "trained-network.nippy"))

  (i/show (i/load-image "data-cats-dogs/testing/cat/1000.png"))

  (guess nippy "data-cats-dogs/testing/cat/1000.png")
  (guess nippy "data-cats-dogs/testing/dog/1929.png")
  (guess nippy (java.net.URL. "https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559_1280.jpg"))

  ; (defn a-dog-is-a-dog [dog-file]
  ;   (require '[clojure.java.io :as io])
  ;   (with-open [rdr (io/reader dog-file)] 
  ;   (doseq [ r (line-seq rdr)] 
  ;     (println 
  ;       (if (= "dog" (guess nippy (java.net.URL. r )))
  ;         "A dog is a dog"
  ;         "sometimes, its a dog's life.")))))

  ; (a-dog-is-a-dog "resources/dogs.txt")
   
)
