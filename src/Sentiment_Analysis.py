#Import Bing Liu's dictionary
from nltk.corpus import opinion_lexicon
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def count_pos_neg(data, positive_dict, negative_dict):
    #Count of positive and negative words that appeared in each movie review
    #Net count which is calculated by positive count subtracting negative count. 
    
    poscnt = []
    negcnt = []
    netcnt = []

    #Running a loop for every row of text data
    for nrow in range(0,len(data)):
        text = data[nrow]
        text = str(text) if pd.notnull(text) else ''
        pos_count = 0
        neg_count = 0

    #looping every word in positive or negative dictionary
    #Checking if the word is in the text file - if so the count will go up
        for word in positive_dict :
            if (word in text) :
                pos_count = pos_count + 1

        for word in negative_dict :
            if (word in text) :
                neg_count = neg_count + 1

        net_count = pos_count - neg_count

        poscnt.append(pos_count)
        negcnt.append(neg_count)
        netcnt.append(net_count)

    return (poscnt, negcnt, netcnt)


#Writing a function to read the text data file for the dictionaries

def read_local_dictionary(file):
    #Create dictionary list
    words_dict = []
    
    #Opening and reading the text file
    with open(file, "r") as f: 
        for line in f:
    #lowering the words in the text file
            t = line.strip().lower()
            words_dict.append(t)
    return words_dict



def count_obs(data, obs_dict):
    #Count of obscene words that appeared in each dialogue
    
    obscnt = []

    #Running a loop for every row of text data
    for nrow in range(0,len(data)):
        text = data[nrow]
        text = str(text) if pd.notnull(text) else ''
        obs_count = 0
        
    #looping every word in obscene words dictionary
    #Checking if the word is in the text file - if so the count will go up
        for word in obs_dict :
            if (word in text) :
                obs_count = obs_count + 1

        obscnt.append(obs_count)

    return (obscnt)


def vader_analyzer(series):
    
    analyzer = SentimentIntensityAnalyzer()
    
    #parsing the data(the dataframe from before) into the analyzer
    data = series["dialogue"].str.lower()
    scores = [analyzer.polarity_scores(sentence) for sentence in data]

    neg_s = [i["neg"] for i in scores]
    neu_s = [i["neu"] for i in scores]
    pos_s = [i["pos"] for i in scores]
    compound_s = [i["compound"] for i in scores]

    series['negscore_Vader'] = series["dialogue"].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    series['neuscore_Vader'] = series["dialogue"].apply(lambda x: analyzer.polarity_scores(x)['neu'])
    series['posscore_Vader'] = series["dialogue"].apply(lambda x: analyzer.polarity_scores(x)['pos'])
    series['compound_Vader'] = series["dialogue"].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    return series
    

