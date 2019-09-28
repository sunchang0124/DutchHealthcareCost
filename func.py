import os
import errno
import numpy as np
from math import pi
import pandas as pd
import seaborn as sns
from decimal import Decimal
from collections import Counter
import matplotlib.pyplot as plt
from bokeh.transform import cumsum
from bokeh.io import output_file, show
from bokeh.core.properties import value
from bokeh.palettes import Category10,Spectral10,Paired,Category20
from bokeh.plotting import figure, show, output_file,save
from bokeh.models import HoverTool

import warnings
warnings.filterwarnings('ignore')


###########################################
### Function for checking missing values ##
###########################################
def check_missing(df, col, file):
    
    ##### Replace customized missing valve #####
    mis_value_code = None  # Input #
    if mis_value_code != None :
        df = df.replace({mis_value_code : np.nan})
    
    ##### Search missing valves #####
    missing  = 0
    misVariables = []
    CheckNull = df.isnull().sum()
    for var in range(0, len(CheckNull)):
        if CheckNull[var] != 0:
            misVariables.append([col[var], CheckNull[var], round(CheckNull[var]/len(df),3)])
            missing = missing + 1

    if missing == 0:
        print('Dataset is complete with no blanks.')
    else:
        print('Totally, %d features have missing values (blanks).' %missing)
        df_misVariables = pd.DataFrame.from_records(misVariables)
        df_misVariables.columns = ['Variable', 'Missing', 'Percentage (%)']
        sort_table = df_misVariables.sort_values(by=['Percentage (%)'], ascending=False)
        # display(sort_table.style.bar(subset=['Percentage (%)'], color='#d65f5f'))
        
        outputFile = 'output/%s_missings.csv' %file
        os.makedirs(os.path.dirname(outputFile), exist_ok=True)
        sort_table.to_csv(outputFile)
        print('Check missing outcome is saved to Output/%s_missings.csv' %file)
    print('Missing values check is done!')

def data_describe(df, col, file):
    outputFile = 'output/%s_describe.csv' %file
    os.makedirs(os.path.dirname(outputFile), exist_ok=True)
    df.describe().to_csv(outputFile)
    print('There is %d rows and %d columns' %(len(df), len(col)))
    print('Data description is done!')

###########################################
### Function for plot Correlation Matrix ##
###########################################
def corr_Matrix(df, age_range, year):
    sns.set(style="white")
    corr = df.corr() # [df_avg['SEX']=='M']
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 12))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr,  cmap=cmap, annot=False, vmax=0.7, vmin=-0.7, #mask=mask,#center=0,
                square=True, linewidths=.2, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix (Age between %d to %d) in %s' %(age_range[0],age_range[1],year))

    filename = 'output/Output_CM/%s/CM_%dTo%d.png'%(year,age_range[0],age_range[1])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print('Correlation Matrix plot is done')
    plt.clf()

#######################################
### Function for plotting Pie Chart ###
#######################################
def merge(df):
    ### Merge variables ###
    if 'SUM' in df.columns:
        df_merge = df.drop('SUM',axis=1)
     
    merged_col = df_merge.columns
    if 'SEX' in merged_col:
        df_merge = df_merge.drop('SEX',axis=1)
    
    if 'AGE' in merged_col:
        df_merge = df_merge.drop('AGE',axis=1)
        
    if 'GP_registration' in merged_col and 'GP_consult' in merged_col and 'GP_others' in merged_col:
        df_merge['GP'] = df['GP_registration']+df['GP_consult']+df['GP_others']
        df_merge = df_merge.drop(['GP_registration','GP_consult','GP_others'],axis=1)   
         
    if 'transport_seat' in merged_col and 'transport_land' in merged_col:
        df_merge['transport'] = df['transport_seat']+df['transport_land']
        df_merge = df_merge.drop(['transport_seat','transport_land'],axis=1)
        
    if 'basicGGZ' in merged_col and 'longGGZ' in merged_col:
        df_merge['GGZ'] = df['basicGGZ']+df['longGGZ']
        df_merge = df_merge.drop(['basicGGZ', 'longGGZ'],axis=1)
        
    if 'paramedical_phy' in merged_col and 'paramedical_others' in merged_col:
        df_merge['paramedical'] = df['paramedical_phy']+df['paramedical_others']
        df_merge = df_merge.drop(['paramedical_phy','paramedical_others'],axis=1)
    
    merged_col_new = list(df_merge.columns)
    full_col = ['medical_specialist','GP','pharmacy','dental','transport','abroad','paramedical', \
                 'others', 'firstLinePsy', 'GGZ','rehabilitation','nursing']
    order_col = []
    for i in full_col:
        if i in merged_col_new:
            order_col.append(merged_col_new[merged_col_new.index(i)])

    df_merge_ordered = df_merge[order_col]
    return df_merge_ordered 


def pie_Chart(df,age_range, year):

    ### Merge variables ###
    df_merge = merge(df)
    x = (df_merge.sum(axis=0)/len(df_merge)).to_dict()

    data = pd.Series(x).reset_index(name='value').rename(columns={'index':'Categories'})
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['color'] = Paired[len(x)]
    data['percentage'] = data['value']/data['value'].sum()
    
    hover = HoverTool(tooltips=[("Categories","@label"),("percentage", "@percentage{%0.2f}")])
    
    p = figure(plot_height=400, title="Pie Chart on costs (Age between %d to %d) in %s" %(age_range[0],age_range[1],year), \
                tools=['save', hover], tooltips="@Categories: @value", x_range=(-0.5, 1.0))
    
    legend_new = []
    for i in range(0,len(data['percentage'])):
        legend_new.append(str(data['Categories'][i])+': '+ str(float(round(Decimal(data['percentage'][i]),4)*100))+"%")
    data['legend_new'] = legend_new
        
    p.wedge(x=0, y=1, radius=0.4, 
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='legend_new', source=data)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None

    filename = "output/Output_Pie/%s/Pie_%dTo%d.html"%(year,age_range[0],age_range[1])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save(p, filename=filename)

    print('Pie plot is done')

##########################################
### Function for plotting Distribution ###
##########################################
def make_hist_plot(title, hist, edges, x, pdf):
    p = figure(title=title, toolbar_location='below', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], 
           fill_color="navy", line_color="white", alpha=0.5)
    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend="PDF")
#     p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend="CDF")

    p.x_range.start = 0
    p.x_range.end = 8000
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'Cost'
    p.yaxis.axis_label = 'Pr(cost)'
    p.grid.grid_line_color="white"
    return p

def dist_Plot (df,featureName,age_range,year):
    F = featureName
    fea = df[F].dropna()
    mu = fea.mean()
    sigma = fea.std()

    hist, edges = np.histogram(fea, density=True, bins=120)

    x = np.linspace(fea.min(), fea.max(), len(df))
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    #   cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2
    p = make_hist_plot("Total healthcare cost in %s (Age between %d to %d)  (μ=%d, σ=%s)" \
                       %(year, age_range[0], age_range[1], mu, sigma), hist, edges, x, pdf)
#     show(p)
    filename = "output/Output_Dist/%s/Dist_%dTo%d.html"%(year,age_range[0],age_range[1])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save(p, filename=filename)
    print('Distribution plot is done')

##########################################
######## Function for box plot ###########
##########################################
def box_Plot(df, featureSet, file):
    sns.set(style="ticks", palette="pastel")

    # Draw a nested boxplot to show bills by day and time
    p = sns.boxplot(x=featureSet[0], y=featureSet[1],
                hue=featureSet[2], palette=Spectral10, data=df)

    sns.despine(offset=10, trim=True)
    filename = "output/Output_BoxPlot/%s_%s_%s_%s.png" %(featureSet[0],featureSet[1],featureSet[2],file)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.clf()

##########################################
### Function for plotting Stacked Area ###
##########################################
def stacked_Plot(df,ageRange,year):
    N = len(df.columns)

    names = ["%s" % i for i in df.columns]

    p = figure(x_range=(0, len(df)-1), y_range=(0, 10000))
    p.grid.minor_grid_line_color = '#eeeeee'

    p.varea_stack(stackers=names, x='index', color=Category20[N], \
                  legend=[value(x) for x in names], source=df)

    p.legend.location = "top_left"
    p.x_range.start = ageRange[0]
    p.x_range.end = ageRange[-1]
    p.xaxis.axis_label = 'Age'
    p.yaxis.axis_label = 'Total healthcare cost in %s' %year
    # reverse the legend entries to match the stacked order
    p.legend[0].items.reverse()

#     show(p)
    filename = "output/Output_Stacked/StackedArea_%s.html" %year
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save(p, filename=filename)

### Give a age range (0-90) ###
def groupAgeRange(df_vektis, age_range, df_stack):
    if type(df_stack) == pd.DataFrame:
        df_vektis_aged = df_vektis[df_vektis['AGE']==age_range]
    else:
        df_vektis_aged = df_vektis[age_range[0]<=df_vektis['AGE']]
        df_vektis_aged = df_vektis_aged[df_vektis['AGE']<=age_range[1]]
        print("The number of insured people between %d to %d: " %(age_range[0],age_range[1]), len(df_vektis_aged))
#     print("The gender balance: ", Counter(df_vektis_aged['SEX']))

    ### As the original data is aggregated data 
    ### so we need to calculate the average costs for per insured person
    df_avg = df_vektis_aged[['SEX','AGE']]
    col = df_vektis_aged.columns
    for i in range(2, len(col)):
        df_avg[col[i]] = df_vektis_aged[col[i]]/df_vektis_aged['BSNs']

    ### Add one column - total healthcare costs ###
    df_avg['SUM'] = df_avg.drop(['SEX','AGE'],axis=1).sum(axis=1, skipna=True)
    
    return df_avg

  
##########################################
### Function for num-num relation plot ###
##########################################
def plot_numNum(df,featureSet,file):
    num1_feature = featureSet[0]
    num2_feature = featureSet[1]
    tar_feature = featureSet[2]
    
    if tar_feature == 'None':
        sns.set(style="white")
        p = sns.jointplot(x=num1_feature, y=num2_feature, data = df, kind="kde", color="b")
        p.plot_joint(plt.scatter, c="r", s=30, linewidth=1, marker="+")
        
        filename = "output/Output_NumNum/%s_%s_%s.png" %(featureSet[0],featureSet[1],featureSet[2])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        p.savefig(filename)
        
    else:
        p = sns.lmplot(x=num1_feature, y=num2_feature, hue=tar_feature, data=df, \
                   palette = 'magma', height = 6)
        filename = "output/Output_NumNum/%s_%s_%s_%s.png" %(featureSet[0],featureSet[1],featureSet[2],file)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        p.savefig(filename)

    print('Numerical-numerical feature plot is done')
    plt.clf()
    



####################################################
### Function for prepare datasets from all years ###
####################################################

def prepare (df, col, name_col, postcode):
    ### Select features you are interested in ###
    ### Feature descriptions are provided by https://www.vektis.nl/intelligence/open-data ###
    # KOSTEN_MEDISCH_SPECIALISTISCHE_ZORG
    # col = ["GESLACHT", "AANTAL_BSN","POSTCODE_3","AANTAL_VERZEKERDEJAREN","KOSTEN_MEDISCH_SPECIALISTISCHE_ZORG",\
    #        "KOSTEN_HUISARTS_INSCHRIJFTARIEF", "KOSTEN_HUISARTS_CONSULT","KOSTEN_HUISARTS_OVERIG", \
    #        "KOSTEN_FARMACIE", "KOSTEN_MONDZORG", "KOSTEN_ZIEKENVERVOER_ZITTEND", "KOSTEN_ZIEKENVERVOER_LIGGEND", \
    #        "KOSTEN_GRENSOVERSCHRIJDENDE_ZORG","KOSTEN_PARAMEDISCHE_ZORG_FYSIOTHERAPIE", \
    #        "KOSTEN_PARAMEDISCHE_ZORG_OVERIG","KOSTEN_OVERIG","KOSTEN_GERIATRISCHE_REVALIDATIEZORG",\
    #        "KOSTEN_VERPLEGING_EN_VERZORGING","KOSTEN_EERSTELIJNS_PSYCHOLOGISCHE_ZORG","KOSTEN_TWEEDELIJNS_GGZ",\
    #        "KOSTEN_SPECIALISTISCHE_GGZ","KOSTEN_GENERALISTISCHE_BASIS_GGZ","KOSTEN_LANGDURIGE_GGZ"]

    ### As some features are available in some years, we need to check before select certain features ###
    data_col = df.columns
    present = []
    for c in col:
        if c in data_col:
            present.append(col.index(c))

    df_vektis = df[np.array(col)[present]]

    ### Give new columns names which are understandable for yourself ###
    # medical_specialist
    # name_col = ["SEX", "BSNs", "Postcode", "Insured_year","medical_specialist", "GP_registration",\
    #             "GP_consult","GP_others","pharmacy","dental","transport_seat", "transport_land",\
    #             "abroad","paramedical_phy","paramedical_others", "others","rehabilitation","nursing",\
    #             "firstLinePsy","secondLineGGZ","specialGGZ","basicGGZ","longGGZ"]
    
    new_col = np.array(name_col)[present]
    df_vektis.columns = new_col

    ### Change the types (int,float,str --> float) of values in the AGE column ###
    age = []
    for i in df['LEEFTIJDSKLASSE']:
        if type(i) == str:
            try:
                age.append(float(i))
            except:
                age.append(float(i[:-1]))
        elif type(i) == float:
            age.append(i)
        elif type(i) == int:
            age.append(i)

    ### Add new age column ###
    df_vektis['AGE'] = age
    ### Remove the first row (sum) ###
    df_vektis = df_vektis[1:]
    
    ### search for certain area? ###
    if postcode == "ALL":
        df_vektis_analysis = df_vektis
    elif 100 < postcode and postcode < 1000:
        df_vektis_analysis = df_vektis[df_vektis['Postcode']==postcode]
    else:
        print("Please give a postcode greater than 100 and less than 1000")
    len(df_vektis_analysis)
    
    return df_vektis_analysis


##############################################################################
### 1. All categories in different years from the same age group (heatmap) ###
##############################################################################
def allCategoriesDiffYear (df_mean_allYears,ageRange_string,years,categories,fileName):
    for elem in ageRange_string:
        avg_cost = []
        year_plt = []
        cate_plt = []
        for i in categories:
            for j in years:
                year_plt.append(j)
                cate_plt.append(i)

                if i in list(df_mean_allYears[elem][j].keys()):
                    avg_cost.append(df_mean_allYears[elem][j][i])
                else:
                    avg_cost.append(0)
        cost_plt = pd.DataFrame.from_records([cate_plt,year_plt,avg_cost]).transpose()
        cost_plt.columns = ['category','Year','Cost']

        sns.set()

        cost_df_pivot = cost_plt.pivot('category','Year','Cost')
        cost_df_pivot.fillna(value=np.nan, inplace=True)

        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(13, 10))
        sns.heatmap(cost_df_pivot, annot=True, fmt="0.4g",linewidths=.5, ax=ax, vmax=500,\
                       cmap=sns.cubehelix_palette(10), cbar=True)
        plt.title("Costs in different categories between 2011-2016 in age group of %s"%elem)

        filename = 'Output_Vektis/withMedSpecialist/%s_Category_Years.png'%ageRange_string[elem]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)


##############################################################################
### 2. All categories in the same year from different age groups (heatmap) ###
##############################################################################
def allCategoriesDiffAge (df_mean_allYears,ageRange_string,years,categories,fileName):
    for elem in years:
        avg_cost = []
        ages_plt = []
        cate_plt = []
        for i in categories:
            for j in ageRange_string:
                ages_plt.append(j)
                cate_plt.append(i)

                if i in list(df_mean_allYears[j][elem].keys()):
                    avg_cost.append(df_mean_allYears[j][elem][i])
                else:
                    avg_cost.append(0)
        cost_plt = pd.DataFrame.from_records([cate_plt,ages_plt,avg_cost]).transpose()
        cost_plt.columns = ['category','Age range','Cost']

        sns.set()

        cost_df_pivot = cost_plt.pivot('category','Age range','Cost')
        cost_df_pivot.fillna(value=np.nan, inplace=True)

        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(13, 10))
        sns.heatmap(cost_df_pivot, annot=True, fmt="0.4g",linewidths=.5, ax=ax, vmax=500,\
                       cmap=sns.cubehelix_palette(10), cbar=True)
        plt.title("Costs in different categories from different age groups in %s"%fileName)

        filename = 'Output_Vektis/%s_Category_Change.png'%fileName
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)


#################################################
### 3. Sum for different age groups and years ###
#################################################
def SumCost_pivot (df_mean_allYears,ageRange_string,years,categories):
    age_plt = []
    year_plt = []
    sum_plt = []
    for i in ageRange_string:
        for j in years:
            age_plt.append(i)
            year_plt.append(j)
            sum_plt.append(df_mean_allYears[i][j].sum())

    sum_cost_plt = pd.DataFrame.from_records([age_plt,year_plt,sum_plt]).transpose()
    sum_cost_plt.columns = ['Age range','Year','Sum of costs']

    sum_cost_plt_pivot = sum_cost_plt.pivot('Age range','Year','Sum of costs')
    sum_cost_plt_pivot = sum_cost_plt_pivot.reindex(ageRange_string)
    sum_cost_plt_pivot.fillna(value=np.nan, inplace=True)
    
    return sum_cost_plt_pivot

def SumCost_heatmap(sum_cost_plt_pivot):
    sns.set()

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(sum_cost_plt_pivot, annot=True, fmt="0.4g",linewidths=.5, ax=ax, vmax=7000,\
                   cmap=sns.cubehelix_palette(10), cbar=True)
    plt.title("Sum of costs in different years from different age groups")
    plt.show()

    filename = 'Output_Vektis/SumVisualization/SumofCost_FULL.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)

##############################
######## Sum line plot #######
##############################
def Sumcost_line (sum_cost_plt_pivot,ageRange_string, start_ageGroup, end_ageGroup):
#     start_ageGroup = 15
#     end_ageGroup = 20

    plt.subplots(figsize=(12, 9))
    palette = sns.color_palette("muted")
    p = sns.lineplot(data=sum_cost_plt_pivot[start_ageGroup:end_ageGroup].transpose(),\
                     linewidth=2.5,legend='full',dashes=False) 

    filename = 'Output_Vektis/SumVisualization/from %s - to %s.png'%(ageRange_string[start_ageGroup],
                                                                ageRange_string[end_ageGroup-1])
    plt.xlabel('Year')
    # plt.ylim(1450,3800)
    plt.ylabel('Sum of costs')
    plt.title('Sum of costs from %s - to %s between 2011-2017' %(ageRange_string[start_ageGroup],
                                                                ageRange_string[end_ageGroup-1]))
    plt.legend(loc='upper right')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)

#################################################
######## 4. Single category plot (heatmap) ######
#################################################
def CatCost_pivot(df_mean_allYears,ageRange_string,years,categories,one_cate):
    age_plt = []
    year_plt = []
    sglCat_plt = []
    for i in ageRange_string:
        for j in years:
            age_plt.append(i)
            year_plt.append(j)
            if one_cate in df_mean_allYears[i][j].keys():
                sglCat_plt.append(df_mean_allYears[i][j][one_cate])
    #         if '1stPsy2ndGGZ' in df_mean_allYears[i][j].keys():
    #             medSpe_plt.append(df_mean_allYears[i][j]['1stPsy2ndGGZ']) # medical_specialist
    #         elif 'GGZ' in df_mean_allYears[i][j].keys():
    #             medSpe_plt.append(df_mean_allYears[i][j]['GGZ'])

    sglCat_cost_plt = pd.DataFrame.from_records([age_plt,year_plt,sglCat_plt]).transpose()
    sglCat_cost_plt.columns = ['Age range','Year','Cost from %s'%one_cate] #Medical Specialists
    
    sglCat_cost_plt_pivot = sglCat_cost_plt.pivot('Age range','Year','Cost from %s'%one_cate) #Medical Specialists
    sglCat_cost_plt_pivot = sglCat_cost_plt_pivot.reindex(ageRange_string)
    sglCat_cost_plt_pivot.fillna(value=np.nan, inplace=True)
    
    return sglCat_cost_plt_pivot

def Catcost_heatmap(sglCat_cost_plt_pivot, one_cate):
    sns.set()

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(sglCat_cost_plt_pivot, annot=True, fmt="0.4g",linewidths=.5, ax=ax, \
                cmap=sns.cubehelix_palette(10), cbar=True)
    plt.title("Costs from %s in different years from different age groups" %one_cate)

    filename = 'Output_Vektis/%s/%s.png' %(one_cate,one_cate)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)


##########################################
####### 5. Sum line plot (heatmap) #######
##########################################
def Catcost_line(sglCat_cost_plt_pivot,ageRange_string, start_ageGroup,end_ageGroup, one_cate):
#     start_ageGroup = 14
#     end_ageGroup = 20

    plt.subplots(figsize=(12, 9))
    p = sns.lineplot(data=sglCat_cost_plt_pivot[start_ageGroup:end_ageGroup].transpose(), \
                     linewidth=2.5,legend='full',dashes=False) 
    filename = 'Output_Vektis/%s/from %s to %s.png' \
    %(one_cate, ageRange_string[start_ageGroup],ageRange_string[end_ageGroup-1])

    plt.ylim(35,135)
    plt.legend(loc='upper right')
    plt.xlabel('Age range')
    plt.ylabel('Costs from %s Package' %one_cate)
    plt.title('Costs from %s from %s to %s '\
              %(one_cate, ageRange_string[start_ageGroup],ageRange_string[end_ageGroup-1]))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    # plt.show()