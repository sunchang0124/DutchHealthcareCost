FROM python:3.6-slim-stretch

RUN apt-get update && apt-get install -y python-pip

RUN mkdir /output

COPY analysis_OverYears.py analysis_OverYears.py

COPY func.py func.py

RUN pip install pandas numpy sklearn seaborn bokeh

CMD ["python", "analysis_OverYears.py"]
