# Visualize Dutch healthcare cost data (Open data from Vektis) #

### Motivation

One of my PhD research projects is to analyze diabetes patients healthcare costs data in a privacy-preserving manner collaborating with [The Maastricht Study](https://www.demaastrichtstudie.nl/research) and [Statistics Netherlands](https://www.cbs.nl). To better understand and make use of the data, I did some research on the Dutch healthcare system, insurance policy, and annual healthcare costs of Dutch citizens. [Vektis](https://www.vektis.nl/) published (open data) some aggregated data of Dutch annual healthcare costs at postcode and municipality level. For the project and my personal interests, I did some pre-processing and visualization on these data. I would like to share the code with people who are also interested in this type of data. 

### About Data [[Download link](https://www.vektis.nl/intelligence/open-data)]

**Please carefully read the text on the website and data description file!**

>Vektis healthcare costs open data files include different cost types within the Dutch health insurance law, such as specialist medical care, pharmacy, mental health care, general practitioner care, medical aids, oral care (for children), paramedical care and maternity care. The files are at an aggregated level, so that privacy is guaranteed and data can never be traced to individuals and health insurers. The files contain aggregated data of all insured persons within the Health Insurance Act. 

### 0. Prerequisites

#### Software
- Docker Community Edition
  - For Ubuntu [Install](https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository)
  - For Windows [Install](https://hub.docker.com/editions/community/docker-ce-desktop-windows)
  - For Mac [Install](https://hub.docker.com/editions/community/docker-ce-desktop-mac)
  
- Or Python 3.6 (with pip as dependency manager) and Jupyter Notebook

  

### 1. Or run in Python (version 3.6+)

- Analyze data from a single year: configure **request_SingleYear.json** file, run the following line in the terminal:

  ```
  python request_SingleYear.py
  ```

  

- Analyze data from all years: configure **request_SingleYear.json** file, run the following line in the terminal:

  ```
  python request_OverYears.py
  ```

  

### 2. Or run in Jupyter Notebook

- Analyze data from a single year: **analysis_SingleYear.ipynb**
- Analyze data from all years: **analysis_OverYears.ipynb**



### 3. Pull Docker Images

```shell
docker pull sophia921025/healthcare_singleyear:v0.1 # Analysis on single year dataset

docker pull sophia921025/healthcare_multipleyears:v0.1 # Analysis on multiple (all) year datasets
```



### 4. Locally build Docker Image

First, build a Docker Image - in the folder (where the Dockerfile is) and run the following line in terminal:

```shell
docker build -t docker build -t healthcost .   
# "healthcost" can be replaced by any name you like but no capital letters .   
# "healthcost" can be replaced by any name you like but no capital letters
```

Then, configure the **request_SingleYear.json** file

Finally, run the following line in terminal:

- Linux/MacOS

```shell
docker run --rm \
-v "$(pwd)/output:/output" \
-v "$(pwd)/Vektis2011.csv:/Vektis2011.csv" \
-v "$(pwd)/request_SingleYear.json:/request_SingleYear.json" healthcost 
```

- Windows:

```shell
docker run --rm \
-v "%cd%/output:/output" \
-v "%cd%/Vektis2011.csv:/Vektis2011.csv" \
-v "%cd%/request_SingleYear.json:/request_SingleYear.json" healthcost
```



### 4. Example output

Heat map - sum of costs

<img src="https://github.com/sunchang0124/DutchHealthcareCost/raw/master/example_output/SumofCost.png" width="720">

Line plot - sum of costs

<img src="https://github.com/sunchang0124/DutchHealthcareCost/raw/master/example_output/SumofCost_line.png" width="720">


Correlation Matrix

<img src="https://github.com/sunchang0124/DutchHealthcareCost/raw/master/example_output/Output_CM/2011/CM_0To4.png" width="720">




Stacked area plot - sum of costs

<img src="https://github.com/sunchang0124/DutchHealthcareCost/raw/master/example_output/Output_Stacked/StackedArea_2011.png" width="720">




Distribution plot

<img src="https://github.com/sunchang0124/DutchHealthcareCost/raw/master/example_output/Output_Dist/2011/Dist_70To79.png" width="720">
