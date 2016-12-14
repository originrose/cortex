CLJS := $(shell find cljs -name "*.cljs")
SRC := $(shell find src -name "*.clj*")
APPJS := resources/public/js/app.js
APPCSS := resources/public/css/app.css

#JAVA := $(shell find java -name "*.java*")


all: uberjar


uberjar: target/classify-example.jar


$(APPCSS): project.clj $(SRC)
	rm $(APPCSS)
	lein garden once


$(APPJS): project.clj $(CLJS)
	rm $(APPJS)
	lein cljsbuild once


target/classify-example.jar: project.clj $(CLJS) $(SRC) $(JAVA) $(APPCSS) $(APPJS)
	lein uberjar
