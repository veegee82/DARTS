FROM python:3.5-stretch
ADD . /app
WORKDIR /app
# RUN apt-get update
# RUN apt-get install -y git
RUN pip install -r requirements.txt
RUN pip install -e git://github.com/Shumway82/tf_base.git#egg=tf_base
RUN pip install -e .
CMD python Binary_Classifier/app.py