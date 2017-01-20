import pandas as pd 
from bs4 import BeautifulSoup
import numpy as np
import os
"""Function that takes in a review, and cleans the same
1. Keeps only alphabets
2. lower cases it
3. Removes stopwords"""
def review_to_words(review):
	example1 = BeautifulSoup(review, "lxml")
	import re 	# Use regular expressions to do a find-and-replace
	letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
	                      " ",                   # The pattern to replace it with
	                      example1.get_text() )  # The text to search

	lower_case = letters_only.lower()        # Convert to lower case
	words = lower_case.split()     

	import nltk
	from nltk.corpus import stopwords
	words = [w for w in words if not w in stopwords.words("english")]
	return " ".join(words)

#main function execution starts here
if __name__ == "__main__":
	#ISSUE:Had a problem with read_csv and moved to file read since I couldnt get that working. Was some unicode issues. 
	#train = pd.read_csv("classify.txt", error_bad_lines=False)
	#open the file and place its content into the arr2 variable for processing
	arr2 = []
	with open("classify_upd.csv", "r") as f:
		strval = f.read()
		strval = strval.replace("\n","\r")
		arr = strval.split("\r")
		for line in arr:
			arr2.append(line.split(","))

	#train is the dataframe that we are using to train the model , columns of which are "text" and "label"
	train = pd.DataFrame(arr2)
	train.columns = ['text', 'label']
	print train.shape
	print train.head(3)
	print train.columns.values #verify of the columns have been placed in the dataframe

	#iterate through ll the items, and cleanse the reviews 
	num_reviews = train["label"].size
	clean_train_reviews = []
	for i in range(0, num_reviews):
		if i % 20 == 0:
			print "we are in item " + str(i)
		clean =  review_to_words(train["text"][i])
		print "   -- " + clean[0: 70]
		clean_train_reviews.append(clean)

	#this is where the guts really start. We create a bag of words. and have each line tagged to that bag of words. 
	#E.g. hotel staff good woudl have hotel (1) staff (1) good(1). Words that appear in the sentence would decide which bucket to classify the same in
	print "Creating the bag of words...\n"
	from sklearn.feature_extraction.text import CountVectorizer

	# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool. 
	#basically, we are saying that we want to track words, the top 5000 of them. 
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             tokenizer = None,    \
	                             preprocessor = None, \
	                             stop_words = None,   \
	                             max_features = 1000) 
	# fit_transform() does two functions: First, it fits the model and learns the vocabulary;
	# second, it transforms our training data into feature vectors. The input to fit_transform should be a list of strings
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	vocab = vectorizer.get_feature_names()
	print clean_train_reviews[4] #display the review number 4
	print train_data_features[4] #show me the features (in the bag of words) that come up in the review #4. Would be a list of numbers
	print vocab[247] #display those 4 items from the bag of words. Should be the same as the words in the ewview
	print vocab[77]
	print vocab[331]
	print vocab[35]

	dist = np.sum(train_data_features, axis=0) # Sum up the counts of each vocabulary word. How many times do they appear
	# For each, print the vocabulary word and the number do each of the words show up. 


	for ct in range(0, len(vocab)):
		print vocab[ct], dist[0, ct]
	
	print "Training the random forest..."
	from sklearn.ensemble import RandomForestClassifier

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators = 10) 

	# Fit the forest to the training set, using the bag of words as 
	# features and the sentiment labels as the response variable
	# This may take a few minutes to run. 
	#So this is actually the most important line. Can just take a bOW, compare it to sentiment, and classify it accordingly. Can be used for any classification?
	forest = forest.fit( train_data_features, train['label'] )
	#--------------save the classifier
	import cPickle
	with open('my_dumped_classifier.pkl', 'wb') as fid:
		cPickle.dump(forest, fid)    
	with open('my_dumped_classifier.pkl', 'rb') as fid:
		forest_file = cPickle.load(fid)
    #--------------
	# Read the test data
	hid = "2248"
	f2 = open(hid + ".txt")
	#clean_test_reviews = ['dont recommend room', 'food was crappy', 'breakfast was bad', 'avoid food here', 'awsum bath room']
	clean_test_reviews = []
	for line in f2:
		line = line.strip().replace("\n","").replace("/n","").strip().replace(",",";")
		clean_test_reviews.extend(line.split("."))
	#end read test data
	test_data_features = vectorizer.transform(clean_test_reviews)
	test_data_features = test_data_features.toarray()

	# Use the random forest to make sentiment label predictions. Print out not just the values, but also the confidence with which it is making the preds. 
	result = forest_file.predict(test_data_features)
	prob = forest_file.predict_proba(test_data_features)
	f = open(hid + "_classified.csv","w+")
	f.write("test, label")
	for t_ in forest_file.classes_:
		f.write("," + t_)
	f.write("\n")
	for txt, res, pro in zip(clean_test_reviews, result, prob):
		if txt.strip() != "":
			f.write(txt + "," + str(res))
			for pp in pro:
				f.write("," + str(pp))
			f.write("\n")
	f.close()
	print "done... check the file '" + hid + "_classified.csv'"
	os.system("open " + hid + "_classified.csv")

