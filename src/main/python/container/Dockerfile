# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:14.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

# Set up the program in the image
COPY logistic_regression /opt/work
WORKDIR /opt/work

# analytics zoo env
ARG ANALYTICS_ZOO_VERSION=0.2.0-SNAPSHOT
ARG SPARK_VERSION=2.2.0
ENV ANALYTICS_ZOO_VERSION_ENV   ${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION_ENV           ${SPARK_VERSION}
#ENV SPARK_HOME                  /opt/work/spark-${SPARK_VERSION}
ENV ANALYTICS_ZOO_HOME          /opt/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME                   /opt/jdk
# PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH                        ${JAVA_HOME}/bin:/opt/work:${PATH}

RUN apt-get update && \
    apt-get install -y vim curl nano wget unzip maven git

# basic tools
RUN apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# java
RUN wget http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 /opt/jdk
# python
RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties python-pkg-resources && \
    add-apt-repository -y ppa:jonathonf/python-2.7 && \
    apt-get update && \
    apt-get install -y build-essential python python-setuptools python-dev && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python2 get-pip.py && \
    pip2 install --upgrade setuptools && \
    pip2 install numpy scipy pandas scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests && \
    pip2 install flask gevent gunicorn && \
    pip2 install pyspark==2.2.1 bigdl==0.6.0 analytics-zoo==0.2.0 && \
    python2 -m pip install ipykernel && \
    python2 -m ipykernel install --user && \
    (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
    rm -rf /root/.cache
# spark
RUN wget https://d3kbcqa49mib13.cloudfront.net/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz
# analytics-zoo
#RUN git config --global http.proxy $http_proxy && \
#    git config --global https.proxy $https_proxy &&\
#RUN    git clone https://github.com/intel-analytics/analytics-zoo.git
