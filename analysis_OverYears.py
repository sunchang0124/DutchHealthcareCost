### Read healthcare cost data from vektis https://www.vektis.nl/intelligence/open-data ###
### Please read the data description before using the data ###
import os
import json
import func
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('request_OverYears.json', 'r') as f:
    input = json.load(f)

file_list = input['data_fileList']
postcode = input['postcode']
selected_features = input['selected_features']
name_features = input['name_features']
ageRange = input['ageRange']
ageRange_string = input['ageRange_string']
one_cate = input['one_cate']

df_years = {}
for file in file_list:
    fileName = file[6:-4]
    df = pd.read_csv(file,delimiter=';')
    df_years[fileName] = func.prepare(df, selected_features, name_features, postcode)

years = list(df_years.keys())

categories = ['medical_specialist','GP','pharmacy','dental','transport','abroad','paramedical', \
                 'others', '1stPsy2ndGGZ', 'GGZ','rehabilitation','nursing']

df_mean_allYears = {}
for i in range(0,len(ageRange)):
    df_mean_byYear = {}
    for year in years:
        temp_df = func.merge(func.groupAgeRange(df_years[year], ageRange[i], 0))
        df_mean_byYear[str(year)] = temp_df.mean()
    df_mean_allYears[ageRange_string[i]] = df_mean_byYear

if input['HeatMap_all_dategories_different_years'] == True:
    func.allCategoriesDiffYear(df_mean_allYears,ageRange_string,years,categories,fileName)

if input['HeatMap_all_dategories_different_ages'] == True:
    func.allCategoriesDiffAge(df_mean_allYears,ageRange_string,years,categories,fileName)

if input['HeatMap_sum_cost'] == True:
    sum_cost_plt_pivot = func.SumCost_pivot(df_mean_allYears,ageRange_string,years,categories)
    func.SumCost_heatmap(sum_cost_plt_pivot)

if input['LinePlot_sum_cost'] == True:
    start_ageGroup = input['start_ageGroup']
    end_ageGroup = input['end_ageGroup']
    sum_cost_plt_pivot = func.SumCost_pivot(df_mean_allYears,ageRange_string,years,categories)
    func.Sumcost_line(sum_cost_plt_pivot,ageRange_string, start_ageGroup, end_ageGroup)

if input['HeatMap_single_category_cost'] == True:
    sglCat_cost_plt_pivot = func.CatCost_pivot(df_mean_allYears,ageRange_string,years,categories,one_cate)
    func.Catcost_heatmap(sglCat_cost_plt_pivot,one_cate)

if input['LinePlot_single_category_cost'] == True:
    start_ageGroup = input['start_ageGroup']
    end_ageGroup = input['end_ageGroup']
    sglCat_cost_plt_pivot = func.CatCost_pivot(df_mean_allYears,ageRange_string,years,categories,one_cate)
    func.Catcost_line(sglCat_cost_plt_pivot,ageRange_string, start_ageGroup, end_ageGroup,one_cate)