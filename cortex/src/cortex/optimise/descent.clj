(ns cortex.optimise.descent
  "Contains API functions for performing gradient descent on pure
  functions using gradient optimisers.

  The basic function used for instrumented gradient descent is 'study'.
  It takes a function, an optimiser, an initial parameter vector,
  and a termination function. Termination functions are simply functions
  that are called with a data structure to determine whether or not to
  stop, but it is most convenient to use the 'term' macro to construct
  termination functions as compositions of the termination function
  transformations defined in this namespace.

  Here is how you can run gradient descent until the magnitude of the
  gradient drops below a certain quantity.

  cortex.optimise.descent> (study function/axis-parallel-hyper-ellipsoid (opts/adam-clojure) [1 2 3 4] (term :out (limit-gradient 1e-10)))
  #vectorz/vector [4.629436394067726E-123,-3.335029160295636E-66,1.5700405042650353E-26,1.2425540969118818E-11]

  To explore the behavior of gradient descent on a particular function,
  you can use 'interactive' instead of 'study', which will allow you to
  select the size of each step, among other features. Make sure to call
  (emacs) first, if you are running your REPL inside Emacs.

  cortex.optimise.descent> (interactive
                             function/cross-paraboloid
                             [1 2 3])
  f =  5.00000e+01, |grad| =  2.78568e+01, learning rate = 1
  f =  5.00000e+01, |grad| =  2.78568e+01, learning rate = 0.1
  f =  3.20000e+00, |grad| =  5.98665e+00, learning rate =
  f =  8.96000e-01, |grad| =  2.12264e+00, learning rate =
  f =  5.27360e-01, |grad| =  1.46503e+00, learning rate = 0.01
  f =  5.06126e-01, |grad| =  1.43377e+00, learning rate = 0.05
  f =  4.08792e-01, |grad| =  1.28312e+00, learning rate =
  f =  3.30701e-01, |grad| =  1.15189e+00, learning rate =
  f =  2.67716e-01, |grad| =  1.03553e+00, learning rate =
  f =  2.16795e-01, |grad| =  9.31507e-01, learning rate = 0.1
  f =  1.38723e-01, |grad| =  7.44924e-01, learning rate =
  f =  8.87815e-02, |grad| =  5.95925e-01, learning rate =
  f =  5.68201e-02, |grad| =  4.76739e-01, learning rate =
  f =  3.63649e-02, |grad| =  3.81392e-01, learning rate =
   .
   .
   .
  f =  2.68325e-04, |grad| =  3.27613e-02, learning rate =
  f =  1.71728e-04, |grad| =  2.62090e-02, learning rate =
  f =  1.09906e-04, |grad| =  2.09672e-02, learning rate = 1
  f =  1.09906e-04, |grad| =  2.09672e-02, learning rate = 0.5
  f =  6.75202e-28, |grad| =  1.03939e-13, learning rate =
  f =  6.75202e-28, |grad| =  1.03939e-13, learning rate = 0.1
  f =  2.70081e-29, |grad| =  2.07877e-14, learning rate =
  f =  1.08032e-30, |grad| =  4.15754e-15, learning rate =
  f =  4.32130e-32, |grad| =  8.31509e-16, learning rate =
  f =  1.72855e-33, |grad| =  1.66302e-16, learning rate =
  f =  6.91607e-35, |grad| =  3.32615e-17, learning rate =
  f =  2.77845e-36, |grad| =  6.65592e-18, learning rate =
  f =  1.18829e-37, |grad| =  1.34269e-18, learning rate =
  f =  9.67577e-39, |grad| =  3.02990e-19, learning rate =
  f =  3.53749e-39, |grad| =  1.27569e-19, learning rate = stop
  #vectorz/vector [1.9825480630431532E-20,-5.1172682557712045E-20,1.9825480630431532E-20]

  Returning to 'study', you can use additional termination function
  transformations to output information about the gradient descent,
  such as the parameter vector at each step.

  cortex.optimise.descent> (study
                             function/axis-parallel-hyper-ellipsoid
                             (opts/adam-clojure)
                             [1 2 3 4]
                             (term
                               :out
                               (output :params)
                               (limit-gradient 3.75e+1)))
  [1.0 2.0 3.0 4.0]
  [0.999000000005 1.99900000000125 2.9990000000005557 3.9990000000003123]
  [0.9980000262138343 1.9980000130698583 2.9980000087050085 3.9980000065256704]
  [0.9970000960651408 1.9970000478935221 2.9970000318981462 3.997000023912038]
  [0.9960002269257634 1.9960001131178096 2.9960000753352967 3.996000056472771]
  [0.995000436052392 1.9950002173271746 2.995000144729716 3.995000108489174]
  [0.9940007405541528 1.9940003690264732 2.994000245740315 3.9940001842013166]
  [0.9930011573564278 1.993000576623114 2.993000383959816 3.9930002877991675]
  [0.9920017031661642 1.9920008484099758 2.992000564903421 3.992000423414121]
  [0.9910023944389119 1.9910011925492044 2.9910007939980736 3.9910005951109664]
  [0.9900032473478027 1.9900016170569963 2.99000107657238 3.9900008068803516]
  [0.9890042777546556 1.989002129789457 2.9890014178472484 3.9890010626317904]
  [0.9880055011833645 1.9880027384296082 2.9880018229272984 3.9880013661872438]
  [0.9870069327956914 1.9870034504756067 2.9870022967930767 3.9870017212753104]
  #vectorz/vector [0.9870069327956914,1.9870034504756067,2.9870022967930767,3.9870017212753104]

  Using 'outputf', you can also format the output nicely.

  cortex.optimise.descent> (study
                             function/axis-parallel-hyper-ellipsoid
                             (opts/adam-clojure)
                             [1 2 3 4]
                             (term
                               :out
                               (outputf
                                 \"% .2e % .2e%n\"
                                 :params
                                 :gradient)
                               (limit-gradient 3.75e+1)))
  [ 1.00e+00  2.00e+00  3.00e+00  4.00e+00] [ 2.00e+00  8.00e+00  1.80e+01  3.20e+01]
  [ 9.99e-01  2.00e+00  3.00e+00  4.00e+00] [ 2.00e+00  8.00e+00  1.80e+01  3.20e+01]
  [ 9.98e-01  2.00e+00  3.00e+00  4.00e+00] [ 2.00e+00  7.99e+00  1.80e+01  3.20e+01]
  [ 9.97e-01  2.00e+00  3.00e+00  4.00e+00] [ 1.99e+00  7.99e+00  1.80e+01  3.20e+01]
  [ 9.96e-01  2.00e+00  3.00e+00  4.00e+00] [ 1.99e+00  7.98e+00  1.80e+01  3.20e+01]
  [ 9.95e-01  2.00e+00  3.00e+00  4.00e+00] [ 1.99e+00  7.98e+00  1.80e+01  3.20e+01]
  [ 9.94e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.99e+00  7.98e+00  1.80e+01  3.20e+01]
  [ 9.93e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.99e+00  7.97e+00  1.80e+01  3.19e+01]
  [ 9.92e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.97e+00  1.80e+01  3.19e+01]
  [ 9.91e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.96e+00  1.79e+01  3.19e+01]
  [ 9.90e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.96e+00  1.79e+01  3.19e+01]
  [ 9.89e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.96e+00  1.79e+01  3.19e+01]
  [ 9.88e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.95e+00  1.79e+01  3.19e+01]
  [ 9.87e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.97e+00  7.95e+00  1.79e+01  3.19e+01]
  #vectorz/vector [0.9870069327956914,1.9870034504756067,2.9870022967930767,3.9870017212753104]

  This next example will wait one second before each step, and stop
  after five steps. By swapping the order of 'slow-steps' and
  'outputf', you can cause the printing to happen after the
  one-second delay, instead of before. Notice that six vectors are
  printed -- this is because the termination function is also called
  before the initial step is taken, with the initial parameter vector.

  cortex.optimise.descent> (study
                             function/sum-of-different-powers
                             (opts/sgd-clojure)
                             [1 1 1 1]
                             (term
                               :out
                               (outputf
                                 \"% .4e%n\"
                                 :params)
                               (slow-steps 1000)
                               (limit-steps 5)))
  [ 1.0000e+00  1.0000e+00  1.0000e+00  1.0000e+00]
  [ 8.0000e-01  7.0000e-01  6.0000e-01  5.0000e-01]
  [ 6.4000e-01  5.5300e-01  5.1360e-01  4.6875e-01]
  [ 5.1200e-01  4.6126e-01  4.5941e-01  4.4461e-01]
  [ 4.0960e-01  3.9743e-01  4.2062e-01  4.2507e-01]
  [ 3.2768e-01  3.5004e-01  3.9086e-01  4.0875e-01]
  #vectorz/vector [0.32768,0.3500446745673379,0.3908563109881773,0.4087480714355686]

  Using the 'pause' transformation, you can view steps in groups of a
  specified size, instead of all at once. You can change the group
  size during the run, and stop when you wish.

  Each blank line represents a line read from stdin. (Remember to call
  (emacs) if your REPL is running within Emacs.) The first group of
  lines printed has three lines, not two, because of the initial step.
  After that, two-line groups are printed until the user switches to
  three-line groups.

  cortex.optimise.descent> (study
                             function/axis-parallel-hyper-ellipsoid
                             (opts/adam-clojure)
                             [1 2 3 4]
                             (term
                               :out
                               (outputf
                                 \"% .2e % .2e%n\"
                                 :params
                                 :gradient)
                               (pause 2)))
  [ 1.00e+00  2.00e+00  3.00e+00  4.00e+00] [ 2.00e+00  8.00e+00  1.80e+01  3.20e+01]
  [ 9.99e-01  2.00e+00  3.00e+00  4.00e+00] [ 2.00e+00  8.00e+00  1.80e+01  3.20e+01]
  [ 9.98e-01  2.00e+00  3.00e+00  4.00e+00] [ 2.00e+00  7.99e+00  1.80e+01  3.20e+01]

  [ 9.97e-01  2.00e+00  3.00e+00  4.00e+00] [ 1.99e+00  7.99e+00  1.80e+01  3.20e+01]
  [ 9.96e-01  2.00e+00  3.00e+00  4.00e+00] [ 1.99e+00  7.98e+00  1.80e+01  3.20e+01]

  [ 9.95e-01  2.00e+00  3.00e+00  4.00e+00] [ 1.99e+00  7.98e+00  1.80e+01  3.20e+01]
  [ 9.94e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.99e+00  7.98e+00  1.80e+01  3.20e+01]
  3
  [ 9.93e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.99e+00  7.97e+00  1.80e+01  3.19e+01]
  [ 9.92e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.97e+00  1.80e+01  3.19e+01]
  [ 9.91e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.96e+00  1.79e+01  3.19e+01]

  [ 9.90e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.96e+00  1.79e+01  3.19e+01]
  [ 9.89e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.96e+00  1.79e+01  3.19e+01]
  [ 9.88e-01  1.99e+00  2.99e+00  3.99e+00] [ 1.98e+00  7.95e+00  1.79e+01  3.19e+01]
  stop
  #vectorz/vector [0.9880055011833645,1.9880027384296082,2.9880018229272984,3.9880013661872438]

  You can use various tricks to help avoid printing long vectors. For
  instance, 'do-study' is the same as 'study', but it returns nil instead
  of the last parameter vector. Also, you can pass vectors and functions
  to outputf, which allow you to display only the magnitude of vectors.
  Note that the :last-step will be nil on the initial step, so you have
  to check that you are not on the initial step before trying to take its
  magnitude. Alternatively, you could add ':in skip-initial' to the
  beginning of the argument list to 'term', which would skip the initial
  step for all transformations further down the pipeline. Using ':in'
  instead of ':out' specifies that the 'skip-initial' transformation
  is a transformation on the *argument* to the termination function,
  rather than a transformation on the *output* of the function.

  cortex.optimise.descent> (do-study
                             function/de-jong
                             (opts/adam-clojure)
                             (repeatedly 1000 rand)
                             (term
                               :out
                               (outputf
                                 \"f = % .8e, |grad| = % .8e, |dx| = % .8e%n\"
                                 :value
                                 [:gradient m/magnitude]
                                 (fn [state]
                                   (if-not (:initial-step state)
                                     (m/magnitude
                                       (:last-step state))
                                     Double/NaN)))
                               (pause 10)))
  f =  3.28332495e+02, |grad| =  3.62398948e+01, |dx| = NaN
  f =  3.27334167e+02, |grad| =  3.61847574e+01, |dx| =  3.16227756e-02
  f =  3.26337893e+02, |grad| =  3.61296495e+01, |dx| =  3.16120894e-02
  f =  3.25343704e+02, |grad| =  3.60745730e+01, |dx| =  3.15932624e-02
  f =  3.24351632e+02, |grad| =  3.60195298e+01, |dx| =  3.15756061e-02
  f =  3.23361711e+02, |grad| =  3.59645220e+01, |dx| =  3.15557906e-02
  f =  3.22373971e+02, |grad| =  3.59095515e+01, |dx| =  3.15304123e-02
  f =  3.21388442e+02, |grad| =  3.58546199e+01, |dx| =  3.14993455e-02
  f =  3.20405151e+02, |grad| =  3.57997291e+01, |dx| =  3.14640620e-02
  f =  3.19424124e+02, |grad| =  3.57448807e+01, |dx| =  3.14267860e-02
  f =  3.18445388e+02, |grad| =  3.56900764e+01, |dx| =  3.13891230e-02
  stop
  nil

  It is often more convenient to just use 'stdout', though.

  cortex.optimise.descent> (do-study
                             function/de-jong
                             (opts/adam-clojure)
                             (repeatedly 1000 rand)
                             (term
                               :out
                               stdout
                               (pause 10)))
  f =  3.30264828e+02, |grad| =  3.63463796e+01, |dx| = NaN
  f =  3.29271516e+02, |grad| =  3.62916804e+01, |dx| =  3.16227750e-02
  f =  3.28280255e+02, |grad| =  3.62370118e+01, |dx| =  3.15879282e-02
  f =  3.27291076e+02, |grad| =  3.61823756e+01, |dx| =  3.15591349e-02
  f =  3.26304008e+02, |grad| =  3.61277737e+01, |dx| =  3.15260513e-02
  f =  3.25319081e+02, |grad| =  3.60732079e+01, |dx| =  3.14842618e-02
  f =  3.24336322e+02, |grad| =  3.60186797e+01, |dx| =  3.14425309e-02
  f =  3.23355757e+02, |grad| =  3.59641909e+01, |dx| =  3.14043550e-02
  f =  3.22377415e+02, |grad| =  3.59097432e+01, |dx| =  3.13678305e-02
  f =  3.21401322e+02, |grad| =  3.58553384e+01, |dx| =  3.13304656e-02
  f =  3.20427505e+02, |grad| =  3.58009779e+01, |dx| =  3.12906640e-02
  stop
  nil

  Following is an extremely complicated example. It serves mostly to
  illustrate just how extensible the termination function
  transformations can be when pushed to their limit. The purpose of
  the example is to investigate the oscillations exhibited by ADADELTA
  when it approaches a minimum. First is displayed the oscillation
  vector, which is defined as the ratio between successive
  gradients. This will be less than one when the gradient is
  shrinking, greater than one when it is growing, and negative when
  oscillations are occurring. On the far right-hand side, the number of
  consecutive steps with oscillations is shown, as well as the text
  '[OSCILLATION]' when this number if non-zero -- to make it easier to
  notice.

  cortex.optimise.descent> (let [osc #(/ (:gradient (:current %))
                                         (:gradient (:last %)))]
                             (study
                               function/cross-paraboloid
                               (opts/adadelta-clojure)
                               [1 2 3]
                               (term
                                 :in
                                 count-steps
                                 stagger
                                 (skip-initial 2)
                                 :out
                                 (outputf
                                   \"%4d. osc = % 7.2f, f = % .2e, grad = % .2e %4d%s%n\"
                                   [:current
                                    :step-count]
                                   osc
                                   [:current :value]
                                   [:current :gradient]
                                   (let [neg-osc* (atom 0)]
                                     #(if (some neg? (osc %))
                                        (swap! neg-osc* inc)
                                        (reset! neg-osc* 0)))
                                   #(if (some neg? (osc %))
                                      \"  [OSCILLATION]\"
                                      \"\"))
                                 (pause 100))))
     2. osc = [   1.00    1.00    1.00], f =  4.96e+01, grad = [ 1.39e+01  1.59e+01  1.79e+01]    0
     3. osc = [   1.00    1.00    1.00], f =  4.94e+01, grad = [ 1.39e+01  1.59e+01  1.79e+01]    0
     4. osc = [   1.00    1.00    1.00], f =  4.91e+01, grad = [ 1.39e+01  1.59e+01  1.79e+01]    0
   .
   .
   .
   101. osc = [   1.00    1.00    1.00], f =  3.08e+01, grad = [ 1.04e+01  1.24e+01  1.44e+01]    0
   102. osc = [   1.00    1.00    1.00], f =  3.06e+01, grad = [ 1.04e+01  1.24e+01  1.43e+01]    0

   103. osc = [   1.00    1.00    1.00], f =  3.05e+01, grad = [ 1.03e+01  1.23e+01  1.43e+01]    0
   104. osc = [   1.00    1.00    1.00], f =  3.03e+01, grad = [ 1.03e+01  1.23e+01  1.43e+01]    0
   .
   .
   .
   559. osc = [   0.55    0.99    1.00], f =  1.66e+00, grad = [ 1.12e-02  1.48e+00  3.13e+00]    0
   560. osc = [   0.21    0.99    1.00], f =  1.65e+00, grad = [ 2.30e-03  1.47e+00  3.11e+00]    0
   561. osc = [  -2.80    0.99    1.00], f =  1.64e+00, grad = [-6.44e-03  1.45e+00  3.10e+00]    1  [OSCILLATION]
   562. osc = [   2.34    0.99    1.00], f =  1.63e+00, grad = [-1.50e-02  1.44e+00  3.08e+00]    0
   563. osc = [   1.56    0.99    1.00], f =  1.62e+00, grad = [-2.35e-02  1.43e+00  3.07e+00]    0
   .
   .
   .
  1015. osc = [   0.85    0.84    0.98], f =  2.29e-03, grad = [-4.22e-04 -3.91e-04  1.10e-01]    0
  1016. osc = [   1.44    1.49    0.98], f =  2.18e-03, grad = [-6.07e-04 -5.84e-04  1.08e-01]    0
  1017. osc = [  -0.03   -0.10    0.98], f =  2.09e-03, grad = [ 1.67e-05  5.76e-05  1.06e-01]    1  [OSCILLATION]
  1018. osc = [-115.22  -33.83    0.96], f =  1.99e-03, grad = [-1.93e-03 -1.95e-03  1.02e-01]    2  [OSCILLATION]
  1019. osc = [  -2.14   -2.20    1.02], f =  1.90e-03, grad = [ 4.13e-03  4.28e-03  1.04e-01]    3  [OSCILLATION]
   .
   .
   .
  1300. osc = [  -1.00   -1.00   -1.00], f =  1.32e-03, grad = [-8.84e-02 -8.90e-02 -6.60e-02]  284  [OSCILLATION]
  1301. osc = [  -1.00   -1.00   -1.00], f =  1.33e-03, grad = [ 8.86e-02  8.91e-02  6.61e-02]  285  [OSCILLATION]
  1302. osc = [  -1.00   -1.00   -1.00], f =  1.33e-03, grad = [-8.88e-02 -8.93e-02 -6.62e-02]  286  [OSCILLATION]
  stop
  #vectorz/vector [-0.013839740893427578,-0.014110179325547643,-0.0025868279787373397]

  Here is one way to look at the full map passed to the termination
  function. Since this is a fairly nonstandard thing, we define
  a termination function transformation inline. After calling the
  original termination function, it pretty-prints the state map.
  Notice that some values are marked as <unrealized> in the printed
  map. This is because the returned map is lazy. To avoid forcing
  the lazy map when we pretty-print it, we have to use the
  lazy map pprint dispatch function defined in util. (Of course,
  if you want to see everything, you can just pprint it, with no
  special dispatch, but computing e.g. the function value might
  take a long time, which is why the map is lazy in the first
  place.)

  cortex.optimise.descent> (do-study
                             function/cross-paraboloid
                             (opts/sgd-clojure)
                             [1 2 3]
                             (term
                               :out
                               ((fn [terminate?]
                                  (fn [state]
                                    (or (terminate? state)
                                        (clojure.pprint/with-pprint-dispatch
                                          util/lazy-map-dispatch
                                          (clojure.pprint/pprint
                                            state))))))
                               (limit-steps 2)))
  {:params #vectorz/vector [1.0,2.0,3.0],
   :value <unrealized>,
   :gradient #vectorz/vector [14.0,16.0,18.0],
   :optimiser
   #function[cortex.optimise.optimisers/sgd-clojure/fn--76714],
   :initial-step true}
  {:last-step <unrealized>,
   :params #vectorz/vector [-0.40000000000000013,0.3999999999999999,1.2],
   :value <unrealized>,
   :gradient
   #vectorz/vector [1.5999999999999992,3.1999999999999993,4.799999999999999],
   :optimiser
   {:update #function[cortex.optimise.optimisers/fn->map/fn--76693],
    :state
    {:params
     #vectorz/vector [-0.40000000000000013,0.3999999999999999,1.2]}},
   :initial-step false}
  {:last-step <unrealized>,
   :params #vectorz/vector [-0.56,0.07999999999999996,0.72],
   :value <unrealized>,
   :gradient
   #vectorz/vector [-0.6400000000000003,0.6399999999999997,1.92],
   :optimiser
   {:update #function[cortex.optimise.optimisers/fn->map/fn--76693],
    :state {:params #vectorz/vector [-0.56,0.07999999999999996,0.72]}},
   :initial-step false}
  nil

  You can also extract the complete state of an algorithm at any given step
  by examining the log. This allows you to do detailed investigation of a
  run without continually restarting it.

  cortex.optimise.descent> (clojure.pprint/pprint
                             (nth (log
                                    function/cross-paraboloid
                                    (opts/sgd-clojure)
                                    [1 2 3]
                                    (term
                                      :out
                                      (limit-steps 2)))
                                  2))
  {:last-step
   #vectorz/vector [-0.15999999999999992,-0.31999999999999995,-0.48],
   :params #vectorz/vector [-0.56,0.07999999999999996,0.72],
   :value 0.8959999999999999,
   :gradient
   #vectorz/vector [-0.6400000000000003,0.6399999999999997,1.92],
   :optimiser
   {:update #function[cortex.optimise.optimisers/fn->map/fn--76693],
    :state {:params #vectorz/vector [-0.56,0.07999999999999996,0.72]}},
   :initial-step false}
  nil

  Other useful functions are util/check-gradient and export-for-mathematica."
  (:refer-clojure :exclude [+ - * / defonce])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [cortex.nn.protocols :as cp]
            [cortex.optimise.functions :as function]
            [cortex.optimise.optimisers :as opts]
            [cortex.util :as util :refer [defonce]]))

;;;; Config

(defonce ^:dynamic *emacs*
  "In the Emacs REPL, user input read from stdin is not printed to
  the console. This leads to a very confusing and unreadable output
  for interactive sessions (such as when running the 'interactive'
  gradient descent function), so you can turn on stdin echoing by
  toggling *emacs* to a truthy value."
  false)

(defn emacs
  "Turn on stdin echoing support (do this if you use Emacs)."
  []
  (alter-var-root #'*emacs* (constantly true)))

(defn no-emacs
  "Turn off stdin echoing support, if it has been turned on."
  []
  (alter-var-root #'*emacs* (constantly false)))

;;;; Interactive gradient descent

(defn interactive
  "Performs interactive gradient descent. The interactive part is
  that before each step, you will be prompted for the learning rate.
  (If :normalize? is falsy, though, the gradient vector will be
  normalized before multiplying it by the learning rate -- so in this
  case, you are specifying the step size rather than the learning rate.)
  You can enter the learning rate in any format accepted by
  Double/parseDouble, and it can even be negative (although you would
  then be performing gradient *ascent* rather than descent). If you
  don't specify a learning rate, the last one you entered is used again.
  (But you cannot do this on the first step, because there is not any
  previous entry at that point.) By default, a step will be aborted if
  it would increase the value of the objective function, but you can
  override this behavior by starting your entry with 'force' (as in
  'force 1', or just 'force' -- the latter version would reuse your
  last entered learning rate). A complete history of past steps is
  kept, and every input you make (even if the step is aborted)
  corresponds to an entry in the history. You can revert to previous
  entries in the history by typing 'back 5' (to go back five entries)
  or just 'back' (to go back one entry). Note that the 'previously
  entered step' is determined by the history, so if you go to a previous
  state in the history and then take a step without providing a learning
  rate, the learning rate used will be the last one entered *before that
  point in the history*, not the last one that you literally entered
  in real life. You can customize the prompt by passing a new string for
  :prompt; this string will be passed to util/sformat with the function
  value and gradient as positional arguments."
  [function initial-params
   & {:keys [normalize? prompt]
      :or {normalize? false
           prompt "f = %1$ .5e, |grad| = %2$ .5e, learning rate = "}}]
  (let [function (cp/update-parameters function initial-params)]
    (loop [lrate-history []
           params-history [(m/array :vectorz initial-params)]
           value-history [(cp/output function)]
           gradient-history [(cp/gradient function)]]
      (if-let [{:keys [lrate force? back]}
               (loop []
                 (print (util/sformat prompt
                                      (peek value-history)
                                      (m/magnitude (peek gradient-history))))
                 (flush)
                 (let [input (read-line)
                       _ (when *emacs* (println input))
                       input (str/trim input)]
                   (if (.startsWith input "back")
                     (let [input (-> input
                                   (str/replace-first #"^back" "")
                                   (str/trim))]
                       (or (try
                             {:back (if (seq input)
                                      (let [back (util/parse-long input)]
                                        (if (neg? back)
                                          (throw (NumberFormatException.))
                                          back))
                                      1)}
                             (catch NumberFormatException _))
                           (recur)))
                     (let [force? (.startsWith input "force")
                           input (-> input
                                   (str/replace-first #"^force" "")
                                   (str/trim))]
                       (if (seq input)
                         (or (try
                               {:lrate (util/parse-double input)
                                :force? force?}
                               (catch NumberFormatException _))
                             (when force? (recur)))
                         (if-let [last-lrate (peek lrate-history)]
                           {:lrate last-lrate
                            :force? force?}
                           (recur)))))))]
        (if back
          (let [length (max 1 (- (count params-history) back))]
            (recur (m/subvector lrate-history 0 (dec length))
                   (m/subvector params-history 0 length)
                   (m/subvector value-history 0 length)
                   (m/subvector gradient-history 0 length)))
          (let [step (* (if normalize?
                          (m/normalise (peek gradient-history))
                          (peek gradient-history))
                        (- lrate))
                params (+ (peek params-history)
                          step)
                function (cp/update-parameters function params)
                value (cp/output function)]
            (if (or (< value (peek value-history))
                    force?)
              (recur (conj lrate-history lrate)
                     (conj params-history params)
                     (conj value-history value)
                     (conj gradient-history (cp/gradient function)))
              (recur (conj lrate-history lrate)
                     (conj params-history (peek params-history))
                     (conj value-history (peek value-history))
                     (conj gradient-history (peek gradient-history))))))
        (peek params-history)))))

;;;; Instrumented gradient descent

(defn study
  "Performs gradient descent on the given function using the provided
  optimiser, starting from the specified initial parameter vector,
  until terminate? returns true. Before each step (even the first
  one), terminate? is called with a lazy map containing :params (the
  current parameter vector), :value (the current value of the
  objective function), :gradient (the current gradient of the
  objective function), :optimiser (the current optimiser object),
  and :initial-step (whether or not at least one step has actually
  been performed). On every step but the initial step, this map will
  also contain :last-step (the last step taken, as a vector) and
  anything from the map returned by calling get-state on the
  optimiser.  Returns the parameter vector at the first time the
  terminate? function returns truthy."
  [function optimiser initial-params terminate?]
  (loop [last-params nil
         params (m/array :vectorz initial-params)
         optimiser optimiser
         initial-step true]
    (let [function (cp/update-parameters function params)
          gradient (cp/gradient function)
          state (util/->?LazyMap
                  (merge
                    (if-not initial-step
                      (merge
                        ;; This *needs* to be first, because lazy maps will be coerced
                        ;; to regular maps if they appear in the tail of a merge, and
                        ;; the map returned by get-state might be lazy.
                        (cp/get-state optimiser)
                        {:last-step (delay (- params last-params))}))
                    {:params params
                     :value (delay (cp/output function))
                     :gradient gradient
                     :optimiser optimiser
                     :initial-step initial-step}))]
      (if-not (terminate? state)
        (let [optimiser (cp/compute-parameters optimiser gradient params)
              new-params (cp/parameters optimiser)]
          (recur params
                 new-params
                 optimiser
                 false))
        params))))

(def do-study
  "Like study, but returns nil."
  (comp (constantly nil)
        study))

(defn log
  "Like study, but returns a vector of the maps that were passed to
  the terminate? function. Each map is fully realized (i.e. it is made
  from a lazy map into a regular map)."
  [function optimiser initial-params terminate?]
  (let [log* (atom [])]
    (study
      function
      optimiser
      initial-params
      (fn [state]
        (swap! log* conj (util/force-map state))
        (terminate? state)))
    @log*))

(defn review
  "Like study, but instead of actually performing gradient descent,
  it just reads from one or more logs to get the maps to pass to the
  terminate? function. If only one log is passed, each map is passed
  in turn. If more than one log is passed, successive vectors of log
  entries (which are maps) are passed in turn. To access data in this
  case, the terminate? function will have to index into the vector,
  then access one of the keys in the map. Passing multiple logs is a
  nice way to view more than one algorithm (or a single algorithm
  with different hyperparameters) side by side."
  [terminate? & logs]
  (loop [entry-vectors (apply map vector logs)]
    (when-not (or (empty? entry-vectors)
                  (terminate? (if (> (count logs) 1)
                                (first entry-vectors)
                                (ffirst entry-vectors))))
      (recur (rest entry-vectors)))))

(defn run-test
  "Run a predefined gradient descent test. Effectively passes the args
  in the second argument to study, with some special functionality
  depending on mode. If mode is :profile, returns a map containing the
  number of times the function's value was calculated (:f), the number
  of times the function's gradient was calculated (:grad), and the
  total CPU time required for the gradient descent (:time, in
  milliseconds). If mode is :study, :do-study, or :log, returns like
  the respective functions. Passing another mode is an error."
  [mode {:keys [function optimiser initial-params terminate?]}]
  (case mode
    :profile
    (let [value* (atom 0)
          gradient* (atom 0)]
      (-> (util/ctime*
            (study
              {:value (fn [params]
                        (swap! value* inc)
                        (function/value function params))
               :gradient (fn [params]
                           (swap! gradient* inc)
                           (function/gradient function params))}
              optimiser
              initial-params
              terminate?))
        (dissoc :return)
        (assoc :value @value*)
        (assoc :gradient @gradient*)))
    (:study :do-study :log)
    (apply (case mode
             :study study
             :do-study do-study
             :log log)
           function
           optimiser
           initial-params
           terminate?)
    (throw (IllegalArgumentException. (str "Unknown mode (" mode ")")))))

(defn run-tests
  "Runs multiple tests in a single mode. Returns a lazy sequence of test
  results."
  [mode & tests]
  (map (partial run-test mode) tests))

(defn export-for-mathematica
  "Given a log, exports information to files on disk about the
  parameter vectors and gradient, in a format compatible with
  the PathVisualizer.nb Mathematica applet. desc should be a
  human-readable description of the log, like \"ADADELTA with
  ε = 1e-3\". This will show up in the drop-down menu in the
  Mathematica applet. Unique filenames are used to avoid
  overwriting any existing files. Note that for the Mathematica
  applet to detect all the files, their numbers must be
  consecutive. This is the default for this function, but if
  you start deleting files manually, you have to make sure that
  you don't break the consecutivity requirement."
  [log desc]
  (let [log (util/vectorize log)
        [parameter-fname gradient-fname]
        (loop [index 1]
          (let [parameter-fname (str "resources/mathematica/opt" index "-param.txt")
                gradient-fname (str "resources/mathematica/opt" index "-grad.txt")]
            (if-not (and (.exists (io/as-file parameter-fname))
                         (.exists (io/as-file gradient-fname)))
              [parameter-fname gradient-fname]
              (recur (inc index)))))]
    (->> log
      (map :params)
      (map (partial str/join ","))
      (cons desc)
      (str/join \newline)
      (spit parameter-fname))
    (->> log
      (map :gradient)
      (map (partial str/join ","))
      (cons desc)
      (str/join \newline)
      (spit gradient-fname))))

;;;; Termination function builders

(defmacro term
  "Constructs a termination function from a collection of transformations.
  Each transformation is a higher-order function that takes a termination
  function (and optionally some additional parameters) and returns another
  termination function. To make it easy to reason about the order in
  which transformations are applied, transformations are grouped into
  two types: 'in' and 'out'. Conceptually, 'in' transformations modify the
  argument passed to the termination function before calling the original
  function; 'out' transformations call the original termination function,
  optionally cause side effects, and optionally modify the return value.

  The syntax for labels+transforms is as follows. Each element must be
  either a transformation function or one of the keywords :in or :out.
  Whether a transformation function is treated as an 'in' or an 'out'
  transformation is determined by looking at the last keyword specified
  in the argument vector. It is an error to specify a transformation as
  the first argument, since it cannot be determined which type the
  transformation is meant to be.

  The threading order of the transformations is designed to be as intuitive
  as possible. That is, provided that standard transformation functions are
  used, when the final termination function is called, first each of the
  'in' transformations are applied to its argument, from left to right,
  then each of the 'out' labels take effect, with the order of the side
  effects being from left to right and the rightmost modifications to the
  return value being last.

  This macro expands into a call to the single-arrow threading macro (->),
  so you can provide additional arguments to transformation functions by
  wrapping them in lists, just like with regular -> calls.

  If you create additional transformation functions, be sure to conform to
  the interface expected by this macro, which is as follows:

  - 'in' transformations should transform the argument, call the original
    termination function with the transformed argument, and return whatever
    is returned by original termination function.
  - 'out' transforms should call the original termination function with the
    given argument, and then terminate if the original termination function
    returns truthy. If the original termination function returns falsy,
    then they should perform any side effects, and return the possibly
    transformed or overridden return value.

  This will provide a consistent user experience at the REPL."
  [& labels+transforms]
  (when (and (seq labels+transforms)
             (not (keyword? (first labels+transforms))))
    (throw
      (IllegalArgumentException.
        (str "If you provide arguments to 'term', the first argument must be a keyword (was "
             (first labels+transforms)
             ")"))))
  (let [{:keys [in out]}
        (reduce (fn [xform-map [kws xforms]]
                  (doseq [kw kws]
                    (when-not (#{:in :out} kw)
                      (throw
                        (IllegalArgumentException.
                          (str "Invalid keyword, must be :in or :out (was "
                               kw
                               ")")))))
                  (update xform-map (last kws) (partial apply (fnil conj [])) xforms))
                {}
                (->> labels+transforms
                  (partition-by keyword?)
                  (partition 2)))]
    `(-> (constantly false) ~@out ~@(reverse in))))

;;;; Termination function 'in' transformations

(defn skip-initial
  "Skips the initial step for any transformations further down the
  pipe. This may be useful because certain pieces of information are
  not available on the first step (such as the last step). For
  instance, you may want to terminate when the step size falls below
  a certain magnitude, but this check will yield a NPE on the initial
  step unless you have skip-initial earlier in the transformation
  pipeline. You can also skip n steps, in general, by passing n."
  ([terminate?]
   (skip-initial terminate? 1))
  ([terminate? n]
   (let [num-calls* (atom 0)]
     (fn [state]
       (when (> (swap! num-calls* inc) n)
         (terminate? state))))))

(defn stagger
  "Transforms the argument into a map where :current is the original
  argument, and :last is the argument from the previous call (or nil
  if it is on the initial step)."
  [terminate?]
  (let [last-state* (atom nil)]
    (fn [state]
      (let [last-state @last-state*]
        (terminate?
          {:last last-state
           :current (reset! last-state* state)})))))

(defn count-steps
  "Adds a :step-count key to the argument, which is assumed to be
  a map, for any transformations further down the pipe. The step
  count is the number of steps that have happened previously, so
  on the initial step it will be 0."
  [terminate?]
  (let [num-steps* (atom -1)]
    (fn [state]
      (terminate? (assoc state :step-count (swap! num-steps* inc))))))

;;;; Termination function 'out' transformations -- return value modifications

(defn limit-steps
  "Stop after the given number of steps, or when the original
  termination function returns truthy."
  [terminate? n]
  (let [num-steps* (atom 0)]
    (fn [state]
      (swap! num-steps* inc)
      (or (terminate? state)
          (> @num-steps* n)))))

(defn limit-params
  "Stop after the distance between the parameter vector and the
  target vector is less than the given cutoff, or when the original
  termination function returns truthy."
  [terminate? cutoff target]
  (fn [state]
    (or (terminate? state)
        (<= (m/distance (- (:params state)
                           target))
            cutoff))))

(defn limit-value
  "Stop after the objective function value drops below the given
  cutoff, or when the original termination function returns truthy."
  [terminate? cutoff]
  (fn [state]
    (or (terminate? state)
        (<= (:value state)
            cutoff))))

(defn limit-gradient
  "Stop after the magnitude of the gradient drops below the given
  cutoff, or when the original termination function returns truthy."
  [terminate? cutoff]
  (fn [state]
    (or (terminate? state)
        (<= (m/magnitude (:gradient state))
            cutoff))))

;;;; Termination function 'out' transformations -- side effects

(defn slow-steps
  "Waits the specified number of milliseconds."
  [terminate? msecs]
  (fn [state]
    (when-not (terminate? state)
      (Thread/sleep msecs))))

(defn outputf
  "Prints information from the data structure passed to the termination
  function. Each of the args should be a function that can be called on
  the data structure in order to return the value that should be printed.
  Alternatively, args can be keys to look up in the data structure,
  which should then be associative. Or, args can be vectors, in which
  case the first function or key is used to get an object from the
  original data structure, then the second function or key is applied
  (or looked up) on the result, and so on, like get-in but allowing for
  functions in the vector. The results obtained by applying each arg to
  the data structure are passed to util/sformat along with fmt to
  determine the string to print. Note that you will have to provide a
  trailing %n if you want a newline to be printed at the end of the line.
  However, flushing is done automatically."
  [terminate? fmt & args]
  (fn [state]
    (or (terminate? state)
        (do
          (apply util/sprintf fmt
                 (for [arg args]
                   (let [transforms (if (vector? arg)
                                      arg
                                      [arg])]
                     (loop [state state
                            transforms transforms]
                       (if (seq transforms)
                         (let [transform (first transforms)]
                           (recur (cond
                                    (fn? transform)
                                    (transform state)
                                    (associative? state)
                                    (get state transform))
                                  (rest transforms)))
                         state)))))
          (flush)
          false))))

(defn output
  "Same as outputf, but with the format as '%s %s ... %s'."
  [terminate? & args]
  (apply outputf terminate?
         (str (str/join \space
                        (repeat (count args) "%s"))
              "%n")
         args))

(defn stdout
  "Same as outputf, but with a predefined fmt and args that print the
  value of the objective function, the magnitude of the gradient, and
  the magnitude of the last step in an easy-to-read tabular format."
  [terminate?]
  (outputf terminate?
           "f = % .8e, |grad| = % .8e, |dx| = % .8e%n"
           :value
           [:gradient m/magnitude]
           (fn [state]
             (if-not (:initial-step state)
               (m/magnitude (:last-step state))
               Double/NaN))))

(defn pause
  "Waits for user input after each period (defaults to 1) steps. When
  prompted for input, press enter to run another period steps, enter
  a new positive integer to change the period, or enter something
  non-numeric to terminate."
  ([terminate?] (pause terminate? 1))
  ([terminate? period]
   (let [step-count* (atom -1)
         period* (atom period)]
     (fn [state]
       (or (terminate? state)
           (when (>= (swap! step-count* inc) @period*)
             (let [input (read-line)]
               (when *emacs* (println input))
               (reset! step-count* 0)
               (when (seq input)
                 (try
                   (reset! period* (util/parse-long input))
                   false
                   (catch NumberFormatException _ true))))))))))
