FROM thinktopic/cortex-base
MAINTAINER ThinkTopic

ENV service docker-example
ADD target/${service}.jar /srv/${service}.jar
WORKDIR /srv
CMD "/usr/bin/java" "-jar" "/srv/${service}.jar"
