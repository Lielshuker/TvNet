FROM python:3.7
# copy the requirements file into the image
COPY requirements.txt /app/requirements.txt
#
# switch working directory
WORKDIR /app
#
## install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt
#
## copy every content from the local file to the image
COPY . /app
#
## configure the container to run in an executed manner
#ENTRYPOINT [ "python" ]
ENV FLASK_APP=manage.py
#
#EXPOSE 5000
#CMD ["-b", ":5000", "api:app"]
CMD ["python", "manage.py"]