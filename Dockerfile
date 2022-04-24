FROM python:3.9

COPY . /app

WORKDIR /app

#Install necessary packages from requirements.txt with no cache dir allowing for installation on machine with very little memory on board
RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt

#expose default streamlit port
EXPOSE 8501

#Running the streamlit app
ENTRYPOINT ["streamlit", "run", "--server.maxUploadSize=5"]
CMD ["src/app.py"]