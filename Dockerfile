# base image
FROM python:3.9
#workdir
WORKDIR /app

# copy
COPY . /app

# run
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# port
EXPOSE 8501

# command
CMD ["streamlit", "run", "app.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]