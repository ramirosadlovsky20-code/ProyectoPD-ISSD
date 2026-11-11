FROM apache/airflow:2.8.1-python3.10

# 1. Instalamos Java 17 como root
USER root
RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless && \
    apt-get clean

# 2. Definimos JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# 3. Volvemos al usuario airflow e instalamos pyspark
USER airflow
RUN pip install --no-cache-dir pyspark