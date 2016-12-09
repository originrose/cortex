CLJS := $(shell find cljs -name "*.cljs")
SRC := $(shell find src -name "*.clj*")
APPJS := resources/public/js/app.js
APPCSS := resources/public/css/app.css

#JAVA := $(shell find java -name "*.java*")

all: uberjar

uberjar: target/classify-example.jar

resources/public/css/app.css: project.clj $(SRC)
	lein garden once

resources/public/js/app.js: project.clj $(CLJS)
	lein cljsbuild once

target/classify-example.jar: project.clj $(CLJS) $(SRC) $(JAVA) $(APPCSS) $(APPJS)
	lein uberjar
