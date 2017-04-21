all: uberjar


uberjar: target/classify-example.jar


target/classify-example.jar: project.clj $(SRC) $(JAVA)
	lein uberjar
