"""
    Name: Derek Chiu, Atharva Nilapwar, Karan
    Course: DS 2500
    Assignment: DS 2500 Project
    Due Date: 04/12/2024
    File name: stock.py
"""

import csv
import requests
import json
import pandas as pd
from io import StringIO
import os

NAME_FILE = "List_of_S%26P_500_companies_1.csv"
API_KEY = "ZA8MZKEKQZ8XT7AO"

"""
Reminder:
    This file should be ran first before running the "Project.py" file, so all
    500 stock price csv files are downloaded. Or else error will occur.
"""

def get_company_names(file_path):
    """
    Parameter: Folder with 500 companies' csv files of daily close stock price
    ----------
    Does: Read every files from the folder, store each company's stock price
    in a df, and store all df into a dictionary with keys being company symbol
    ----------
    Return: A dictionary with 500 key-value pairs, its keys are company symbol, 
    and values are df that contains the close stock price
    """
    
    all_name_list = []
    with open(file_path, "r", encoding="utf-8") as infile:
        name_file = csv.reader(infile)
        for row in name_file:
            name = row[0]
            all_name_list.append(name)
        all_name_list = all_name_list[1:]
    return all_name_list

def store_csv(name_list_75):
    """
    Parameter: Folder with 500 companies' csv files of daily close stock price
    ----------
    Does: Read every files from the folder, store each company's stock price
    in a df, and store all df into a dictionary with keys being company symbol
    ----------
    Return: A dictionary with 500 key-value pairs, its keys are company symbol, 
    and values are df that contains the close stock price
    """
    
    folder_path = "500 company stock csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for company_name in name_list_75:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&\
symbol={company_name}&outputsize=full&apikey={API_KEY}&datatype=csv"
        response = requests.get(url)
        all_data = []
        if response.status_code == 200:
            csv_data = response.text
            csv_reader = csv.reader(StringIO(csv_data))
            for row in csv_reader:
                all_data.append(row)
        data_dct = {}
        data_dct["Date"], data_dct["Close"] = [], []
        for idx in range(1, len(all_data)):
            data_dct["Date"].append(all_data[idx][0])
            data_dct["Close"].append(all_data[idx][4])
        df = pd.DataFrame(data_dct)
        csv_file_name = company_name + ".csv"
        file_path = os.path.join(folder_path, csv_file_name)
        df.to_csv(file_path, index=False)
    return "75 csv files stored"
        
def main():
    all_name_list = get_company_names(NAME_FILE)
    name_list_1, name_list_2 = all_name_list[:75], all_name_list[75:150]
    name_list_3, name_list_4 = all_name_list[150:225], all_name_list[225:300]
    name_list_5, name_list_6 = all_name_list[300:375], all_name_list[375:450]
    name_list_7 = all_name_list[450:500]
    
    """
    Our api plan for Nasdaq stock price data is 75 call per minute, so we have
    to split the 500 companies into 7 groups, and call them group by group to
    get all the stock data. After each call, we'll wait for one minute for the
    call limit to reset then call for the next group of companies.
    We iterate through the 7 companies group by switching the parameter inside
    the store_csv() function from name_list_1 to name_list_7.
    Eventually, we had 500 stock price csv files for each company stored in a 
    folder on our laptops.
    """
    
    print (store_csv(name_list_1))
 
if __name__ == "__main__":
    main()