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

### 1. Run with Docker 

First, build a Docker Image - in the folder (where the Dockerfile is) and run the following line in terminal:

```shell
docker build -t healthcost .   
# "healthcost" can be replaced by any name you like but no capital letters
```

Then, configure the **request.json** file

Finally, run the following line in terminal:

- Linux/MacOS

```shell
docker run --rm \
-v "$(pwd)/output:/output" \
-v "$(pwd)/Vektis2011.csv:/Vektis2011.csv" \
-v "$(pwd)/request.json:/request.json" healthcost
```

- Windows:

```shell
docker run --rm \
-v "%cd%/output:/output" \
-v "%cd%/Vektis2011.csv:/Vektis2011.csv" \
-v "%cd%/request.json:/request.json" healthcost
```



### 2. Or run in Python (version 3.6+)

After configuring **request.json** file, run the following line in the terminal:

```shell
python requestBasicInfo.py
```



### 3. Or run in Jupyter Notebook

Run Jupyter Notebook and open the **analysis_nb.ipynb** file



### 4. Example output