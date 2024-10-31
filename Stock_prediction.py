"""
    Name: Derek Chiu, Atharva Nilapwar, Karan
    Course: DS 2500
    Assignment: DS 2500 Project
    Due Date: 04/12/2024
    File name: Project.py
"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import requests
import json
import csv
import os

PROJECT = "en.wikipedia.org"
ACCESS = "all-access"
AGENT = "user"
FREQUENCY = "monthly"
START_DATE = "20150701"
END_DATE = "20190701"
VIEWS_HEADERS = {"User-Agent": "user"}
STOCK_FOLDER = "500 company stock csv"
S_P_500 = "List_of_S%26P_500_companies_1.csv"
SAMPLE_FILE = "AMZN.csv"

"""
Before start running this file:
    - Run the code in the "stock.py" file first to get all 500 stock price csv
      files downloaded in a folder on your device
    - Download the csv file of the S&P 500 company list from Wikipedia
    - Download the csv file for Amazon stock price from 2015-07-01 to
      2019-07-01 from Nasdaq website
    - Without running the stock.py file and downloading these two files first, 
      the code in this file cannot be ran
"""

def read_stock_file(stock_folder):
    """
    Parameter: Folder with 500 companies' csv files of daily close stock price
    ----------
    Does: Read every files from the folder, store each company's stock price
    in a df, and store all df into a dictionary with keys being company symbol
    ----------
    Return: A dictionary with 500 key-value pairs, its keys are company symbol, 
    and values are df that contains the close stock price
    """
    
    stock_file_list = os.listdir(stock_folder)
    curr_dir = os.getcwd()
    all_paths = []
    all_stock_df = {}
    for stock_file in stock_file_list:
        stock_path = os.path.join(STOCK_FOLDER, stock_file)
        stock_file_path = os.path.join(curr_dir, stock_path)
        all_paths.append(stock_file_path)
        company_name = stock_file.split(".")[0]
        all_stock_df[company_name] = "DataFrame"
    all_names = list(all_stock_df.keys())
    for path in range(len(all_paths)):
        df = pd.read_csv(all_paths[path])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"] > "2015-06-30"]
        df = df[df["Date"] < "2019-07-02"]
        df.set_index("Date", inplace=True)
        df = df.resample('MS').first()
        all_stock_df[all_names[path]] = df
    return all_stock_df

def get_all_wiki_names(S_P_500, all_stock_df):
    """
    Parameter: A file of S&P 500 companies, a dictionary returned by 
    read_stock_file function
    ----------
    Does: Get all company names from their Nasdaq symbol
    ----------
    Return: A list of 500 company names obtained from their Nasdaq symbol
    """
    
    symbol_name_mapping = {}
    all_wiki_names = []
    with open(S_P_500, "r", encoding="utf-8") as infile:
        name_file = csv.reader(infile)
        next(name_file)
        for row in name_file:
            symbol_name_mapping[row[0]] = row[1]
        all_symbols = list(all_stock_df.keys())
        for symbol in all_symbols:
            if symbol == "BRK":
                symbol = "BRK.B"
            wiki_name = symbol_name_mapping[symbol]
            all_wiki_names.append(wiki_name)
    return (all_wiki_names, symbol_name_mapping)

def request_views_api(all_wiki_names):
    """
    Parameter: A list of company names returned by get_all_wiki_names function
    ----------
    Does: Request wiki pageviews data by api, store companies' monthly view 
    data in a df, and store all df in a dictionary with key being company name.
    Companies with more than 3 months of missing data are specifically labeled.
    ----------
    Return: A dictionary with 500 key-value pairs, its keys are company names, 
    and values are df that contains the monthly wiki pageviews
    """
    
    all_views_data = {}
    for company_name in all_wiki_names:
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-\
article/{PROJECT}/{ACCESS}/{AGENT}/{company_name}/{FREQUENCY}/{START_DATE}/\
{END_DATE}"
        response = requests.get(url, headers=VIEWS_HEADERS)
        if response.status_code == 200:
            views_data = response.json()
            data_count = len(views_data["items"])
            if data_count in [46, 47, 48, 49]:
                all_views = {}
                all_views["timestamp"] = []
                all_views["views"] = []
                for dct in views_data["items"]:
                    all_views["timestamp"].append(dct["timestamp"][:8])
                    all_views["views"].append(dct["views"])
                all_views_df = pd.DataFrame(all_views)
            else:
                all_views_df = "missing data"
        else:
            all_views_df = "status code error"
        all_views_data[company_name] = all_views_df
    return all_views_data

def get_working_stock_views(all_views_data, all_stock_df):
    """
    Parameter: Two dictionaries, one with company's wiki pageviews, one with 
    their close stock price
    ----------
    Does: Remove companies with more than 3 months of missing views data in
    both stock and pageviews dictionary
    ----------
    Return: The stock and pageviews dictionary with only 400 key-value pairs
    """
    
    all_wrong_idx = []
    for key, value in all_views_data.items():
        if type(value) == str:
            name_list = list(all_views_data.keys())
            idx = name_list.index(key)
            all_wrong_idx.append(idx)
    all_symbols = list(all_stock_df.keys())
    for idx in all_wrong_idx:
        del all_views_data[name_list[idx]]
        del all_stock_df[all_symbols[idx]]
    return (all_stock_df, all_views_data)

def fill_views_data(working_views):
    """
    Parameter: The pageviews dictionary with 400 key-value pairs
    ----------
    Does: For companies in pageviews dictionary with less than 3 months of
    missing views data, fill their data using linear regression prediction
    ----------
    Return: Pageviews dictionary with complete views data
    """
    
    full_dates = pd.date_range(start="2015-07-01", end="2019-07-01", \
                               freq="MS")
    edited_full_dates = [str(date).split(" ")[0] for date in full_dates]
    for key, value in working_views.items():
        if value.shape[0] == 49:
            continue
        else:
            timestamps = list(value["timestamp"])
            views = list(value["views"])
            int_dates = [int(time) for time in timestamps]
            model = LinearRegression()
            model.fit(np.array(int_dates).reshape(-1, 1), views)
            for date in edited_full_dates:
                date = int(date.replace("-", ""))
                if date not in int_dates:
                    predicted = int(model.predict([[date]]))
                    working_views[key].loc[len(value.index)] = \
                        [str(date), predicted]
    return working_views

def fill_stock_data(working_stock_df):
    """
    Parameter: The stock dictionary with 400 key-value pairs
    ----------
    Does: Remove companies from the stock dictionary with missing stock data
    ----------
    Return: The stock dictionary with 391 key-value pairs, each has complete 
    stock price data. (Too much data missing, so can't use linear regression)
    A list of wrong keys index with missing data is also returned
    """
    
    stock_keys = list(working_stock_df.keys())
    all_wrong_key = []
    for key, value in working_stock_df.items():
        if value.shape[0] != 49:
            key_idx = stock_keys.index(key)
            all_wrong_key.append(key_idx)
            replace_key = stock_keys[key_idx + 1]
            working_stock_df[key] = working_stock_df[replace_key]
    return (working_stock_df, all_wrong_key)
    
def merge_stock_views(working_views, working_stock_df, all_wrong_key):
    """
    Parameter: Pageviews dictionary, stock dictionary, wrong key index list
    ----------
    Does: Merge the pageviews and stock dictionary together, then remove the
    companies from the merged dict that has missing stock price data
    ----------
    Return: A dictionary with 391 key-value pairs, with keys being company name
    and values being a df with shape of (49, 2), which the 2 columns are page-
    views and close stock price of. The 49 rows represents each first day of a
    month from 2015-07-01 to 2019-07-01.
    """

    for key in working_views.keys():
        working_views[key]["timestamp"] = \
            pd.to_datetime(working_views[key]["timestamp"])
        working_views[key].set_index("timestamp", inplace=True)
        working_views[key].sort_index(axis=0, inplace=True)
    name, symbol = list(working_views.keys()), list(working_stock_df.keys())
    name_mapping = {}
    merged_dct = {}
    for idx in range(len(name)):
        name_mapping[name[idx]] = symbol[idx]
    for key, value in working_views.items():
        views_df = working_views[key]
        stock_df = working_stock_df[name_mapping[key]]
        merged_df = pd.merge(views_df, stock_df, \
                             left_index=True, right_index=True)
        merged_dct[key] = merged_df
    for key in all_wrong_key:
        del merged_dct[list(merged_dct.keys())[key]]
    return merged_dct

def percent_return(merged_dct):
    '''
    Parameters: merged_dct (dictionary where keys are company names and values 
    are dataframes with pageviews and closeprice; 2015-19)
    Description: Get percent return by taking close prices from merged_dct
    Returns: Add a column to merged_dct that shows % return in close price
    '''
    for company, data_frame in merged_dct.items():
        close_prices = data_frame['Close']
        price_diff = close_prices.diff()
        percent_return = (price_diff / close_prices.shift(1)) * 100
        percent_return.fillna(0, inplace=True)
        data_frame['Percent Return'] = percent_return
    return merged_dct

def categorize_quantiles(merged_dct):
    """
    Parameter: Merged dict with 391 companies' stock price & pageviews data
    ----------
    Does: Categorize the 391 companies into 5 quintiles based on the 
    average and variance of their monthly pageviews
    ----------
    Return: A dictionary with keys being the quintile names and values being
    a list containing company names within the quintile
    """
    
    all_company_scores = []
    for key, value in merged_dct.items():
        pageviews = list(value["views"])
        avg_views = statistics.mean(pageviews)
        var_views = statistics.variance(pageviews)
        score = round(0.5 * (avg_views + var_views), 2)
        all_company_scores.append([key, score])
    sorted_score = sorted(all_company_scores, key=lambda x: x[1], reverse=True)
    sorted_companies = [score[0] for score in sorted_score]
    quantile_range = ["0:79", "79:157", "157:235", "235:313", "313:391"]
    quantiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    all_quantiles = {}
    for q in range(len(quantile_range)):
        list_range = quantile_range[q].split(":")
        start, end = int(list_range[0]), int(list_range[1])
        all_quantiles[quantiles[q]] = sorted_companies[start:end]
    return all_quantiles

def create_lin_regres(merged_dct, all_quantiles):
    """
    Parameter: The merged dictionary, and the quintile dictionary
    ----------
    Does: Create a linear regression model for each quintile, the x_train is
    the average pageviews and y_train is the average stock price return
    ----------
    Return: A list of five items, each item is a linear regression model
    """
    
    all_quantile_views, all_quantile_price = [], []
    for quantile in all_quantiles.values():
        quantile_views, quantile_price = [], []
        for name in quantile:
            df = merged_dct[name]
            pageviews = list(df["views"])
            if df.shape[1] == 3:
                stock_price = list(df["Percent Return"])
            else:
                stock_price = list(df["Close"])
            quantile_views.append(pageviews)
            quantile_price.append(stock_price)
        all_quantile_views.append(quantile_views)
        all_quantile_price.append(quantile_price)
    all_avg_x_test, all_avg_y_test = [], []
    for quantile in range(len(all_quantile_views)):
        array_x_2d = np.array(all_quantile_views[quantile])
        array_y_2d = np.array(all_quantile_price[quantile])
        avg_views = np.mean(array_x_2d, axis=0)
        avg_price = np.mean(array_y_2d, axis=0)
        all_avg_x_test.append(avg_views)
        all_avg_y_test.append(avg_price)
    all_models = []
    for idx in range(len(all_avg_x_test)):
        lin_regress = LinearRegression()
        X_train = all_avg_x_test[idx].reshape(-1, 1)
        y_train = all_avg_y_test[idx].reshape(-1, 1)
        model = lin_regress.fit(X_train, y_train)
        all_models.append(model)
    return all_models

def request_test_views(merged_dct, start="20190801", end="20230701"):
    """
    Parameter: The merged dictionary, a start and end date for api request
    ----------
    Does: Request the pageviews data for each company inside the merged dict
    from 20190801 to 20230701, organize the data into a list
    ----------
    Return: A dictionary with 391 key-value pairs, with keys being company
    names and values being a list containing their pageviews data
    """
    
    company_names = list(merged_dct.keys())
    all_test_views = {}
    for name in company_names:
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-\
article/{PROJECT}/{ACCESS}/{AGENT}/{name}/{FREQUENCY}/{start}/{end}"
        response = requests.get(url, headers=VIEWS_HEADERS)
        if response.status_code == 200:
            views_data = response.json()
            all_views = []
            for dct in views_data["items"]:
                all_views.append(dct["views"])
        else:
            all_views = "status code error"
        all_test_views[name] = all_views
    return all_test_views
    
def predict_stock_price(all_test_views, all_quantiles, all_models):
    """
    Parameter: Dictionary with pageviews from 2019 to 2023, quintile dictionary
    , and a list of the regression models
    ----------
    Does: Predict the stock price of each quintile from 2019 to 2023 using the
    linear regression model, with newly requested pageviews being the x_test
    ----------
    Return: A list containing 5 inner list, each contains the predicted average
    monthly stock price for each quintile
    """
    
    for key, value in all_test_views.items():
        if len(value) == 47:
            all_test_views[key].append(statistics.mean(value))
    all_quantile_test_p = []
    for value in all_quantiles.values():
        quantile_views = []
        for company_name in value:
            quantile_views.append(all_test_views[company_name])
        array_2d = np.array(quantile_views)
        avg_quantile_views = np.mean(array_2d, axis=0)
        all_quantile_test_p.append(avg_quantile_views)
    for quantile in range(len(all_quantile_test_p)):
        all_quantile_test_p[quantile][-1] = all_quantile_test_p[quantile][-2]
    all_predicted_price = []
    for testing_quantile in range(len(all_quantile_test_p)):
        predicted_price = all_models[testing_quantile].predict\
            (all_quantile_test_p[testing_quantile].reshape(-1, 1))
        all_predicted_price.append(list(predicted_price))
    return all_predicted_price

def find_dates(start_date, end_date):
    """
    Parameter: A start and end date
    ----------
    Does: Find a list of dates given the date range, in the YYYY/MM/D format
    ----------
    Return: A list of dates within the desired range
    """
    
    dates = list(pd.date_range(start=start_date, end=end_date, freq="MS"))
    for d in range(len(dates)):
        dates[d] = str(dates[d]).split(" ")[0]
        dates[d] = str(dates[d]).split("-")
        dates[d] = dates[d][0] + "-" + dates[d][1] + "-" + list(dates[d][2])[1]
    return dates

def plot_predict_stock(all_predicted_price, all_quantiles, plot_kind: str):
    """
    Parameter: A list of the predicted price, the quintile dictionary
    ----------
    Does: Plot the predicted stock price for each quintile from 2019 to 2023
    ----------
    Return: A time-series plot with x-axis being dates and y-axis being the
    predicted stock price of the 5 quintiles
    """
    
    dates = find_dates("2019-08-01", "2023-07-01")
    color_mapping = {0: "brown", 1: "red", 2: "blue", 3: "green", 4: "yellow"}
    if plot_kind == "price":
        plot_title = "Monthly stock price prediction of 5 groups from \n\
2019-08-01 to 2023-07-01"
        y_label = "Stock price ($)"
        return_statement = "Predicted stock price plotted."
        all_predicted_price[1], all_predicted_price[4] = \
            all_predicted_price[4], all_predicted_price[1]
    else:
        plot_title = "Monthly percent return prediction of 5 groups from \n\
2019-08-01 to 2023-07-01"
        y_label = "Percent return (%)"
        return_statement = "Predicted stock percent return plotted."
    for quantile in range(len(all_predicted_price)):
        label_name = list(all_quantiles.keys())[quantile]
        plt.plot(dates, all_predicted_price[quantile], \
                 color=color_mapping[quantile], label=label_name)
    plt.title(plot_title)
    plt.xlabel("Dates")
    plt.ylabel(y_label)
    plt.xticks(dates[::len(dates)//5])
    plt.legend()
    plt.show()
    return return_statement

def normalize_data(lst):
    """
    Parameter: An input list
    ----------
    Does: Normalize the item inside the list
    ----------
    Return: A list of the normalized version of the original input list
    """
    
    mx, mn = max(lst), min(lst)
    all_normal_data = []
    for i in lst:
        normal_data = (i - mn) / (mx - mn)
        all_normal_data.append(normal_data)
    return all_normal_data

def plot_sample_corr(sample_file, merged_dct):
    """
    Parameter: The merged dictionary, Amazon stock price csv file
    ----------
    Does: Plot a graph showing the correlation between Amazon's pageviews and
    monthly stock price from 2015 to 2019
    ----------
    Return: A plot of Amazon's pageviews & stock price correlation
    """
    
    df = pd.read_csv(sample_file)
    sample_stock = list(df["Close"])
    normal_stock = normalize_data(sample_stock)
    sample_views = list(merged_dct["Amazon"]["views"])
    normal_views = normalize_data(sample_views)
    corr_score = round(statistics.correlation(normal_views, normal_stock), 3)
    dates = find_dates("2015-07-01", "2019-07-01")
    print (f"Pearson coefficient: {corr_score}")
    plt.plot(dates, normal_stock, color="brown", label="Stock price")
    plt.plot(dates, normal_views, color="red", label="Pageviews")
    plt.title("Correlation between Amazon's pageviews & close stock price")
    plt.xlabel("Dates")
    plt.ylabel("Stock price ($)")
    plt.xticks(dates[::len(dates)//5])
    plt.legend()
    plt.show()
    return "Sample correlation plotted."
    
def main():
    all_stock_df = read_stock_file(STOCK_FOLDER)
    wiki_name_return = get_all_wiki_names(S_P_500, all_stock_df)
    all_wiki_names = wiki_name_return[0]
    symbol_name_mapping = wiki_name_return[1]
    all_views_data = request_views_api(all_wiki_names)
    working_stock_df = get_working_stock_views(all_views_data, all_stock_df)[0]
    working_views = get_working_stock_views(all_views_data, all_stock_df)[1]
    stock_n_wrong_key = fill_stock_data(working_stock_df)
    working_stock_df = stock_n_wrong_key[0]
    all_wrong_key = stock_n_wrong_key[1]
    working_views = fill_views_data(working_views)
    merged_dct = merge_stock_views(working_views, working_stock_df, \
                                    all_wrong_key)
    all_quantiles = categorize_quantiles(merged_dct)
    all_models = create_lin_regres(merged_dct, all_quantiles)
    all_test_views = request_test_views(merged_dct)
    all_predicted_price = predict_stock_price(all_test_views, \
                                                  all_quantiles, all_models)
    predicted_stock_plot = plot_predict_stock(all_predicted_price, \
                                              all_quantiles, "price")
    corr_sample_plot = plot_sample_corr(SAMPLE_FILE, merged_dct)
    print (corr_sample_plot)
    print (predicted_stock_plot)
    merged_dct = percent_return(merged_dct)
    return_models = create_lin_regres(merged_dct, all_quantiles)
    all_predicted_return = predict_stock_price(all_test_views, \
                                                  all_quantiles, return_models)
    return_plot = plot_predict_stock(all_predicted_return, all_quantiles, \
                                     "return")
    print (return_plot)
    
if __name__ == "__main__":
    main()