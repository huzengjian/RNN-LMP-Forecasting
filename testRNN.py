# python testRNN.py 0 100 100 10 1000 MISO

# features: dim 1: hours (1-24) dim 2: days (1-1000+). dim 3: features (1-10+)
# targets: dim 1: hours (1-24) dim 2: days (1-1000+). dim 3: features (1-10+)


from rnn_minibatch import MetaRNN
import pandas as pd
import numpy as np
from dateutil import parser
import time
from datetime import date
import os,sys
from os import listdir
from os.path import isfile, join
import cPickle as pickle

hours_in_day = 24
WeekDayToInt = dict(zip(
	["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], 
	range(7)))

WinterMonths = [11,12,1,2]
	

# Market Study Parameters. 1. Name of the target column 2. list of columns to be dropped 3. True = use MAPE False = use MAE
MarketStudyParams = dict(zip(
	['MISO', 'PJM', 'MISO2','PJM2','Malin-NP'],
	[['Price', ['Date','HE','Day.of.Week', 'Prev.Week.Price','Total.RTO.Load','Actual.Load','Wtd.Avg.Index'], True],
	 ['Price', ['Date','HE','Day.of.Week', 'Prev.Week.Price','Total.RTO.Load','Actual.Load','Wtd.Avg.Index'], True],
	 #['Price', ['Date','HE','Day.of.Week'], True],
	 ['Price', ['Date','HE','Weekday'], True],
	 ['Price', ['Date','HE','Weekday'], True],
	 ['MalinNP15Spread', ['MalinPrice','NP15Price','Date','HourlyLaggedSpread'], False]])) 
	 
# ['Price', ['Date','Total.RTO.Load','Actual.Load','Prev.Day.Avg.Price','Prev.Day.Price','Prev.Week.Price'], True],
# ['Price', ['Date','HE','Prev.Week.Price','Total.RTO.Load','Actual.Load','Prev.Day.Avg.Load','Prev.Week.Load','Wtd.Avg.Index','Prev.Day.Gas.Price','Wind.Gen'], True],	 
def normalize(x):
	maxX = max(x)
	minX = min(x)
	return np.array([(float(i) - minX)/(maxX - minX) for i in x])

def load(file, market):
	df = pd.read_csv(file)
	df['Date'] = [parser.parse(d) for d in df.Date]
	df['Month'] = [d.month for d in df.Date]
	df['Day'] = [d.day for d in df.Date]
	dropped_columns = MarketStudyParams[market][1]
	df.drop(dropped_columns, axis=1, inplace=True, errors='ignore')
	
	df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
	#df.dropna(inplace = True)
	
	target_column = MarketStudyParams[market][0]
	if market != 'Malin-NP':
		df = df[df[target_column] > 5]
		#df['Day.of.Week'] = [WeekDayToInt[x] for x in df['Day.of.Week']]
		
	#train_columns = set(['Total.RTO.Load','Prev.Week.Load','Wind.Gen','is.Hol.Wknd','Day.of.Week','Prev.Day.Price','Price'])
	#columns = set(list(df))-set(['Prev.Day.Avg.Price','Prev.Day.Price','Prev.Week.Price'])
	#df = df[list(columns)]
	
	
	#df = df[~df['Month'].isin(WinterMonths)]
	return df
	
def prepareAllData(df, target_column, n_steps = 24):
	train_columns = set(list(df))-set([target_column])
	print train_columns
	nrows = len(df.index)

	allX = df.as_matrix(train_columns).transpose() 
	allX = np.asmatrix(np.array([normalize(x) for x in allX])).transpose() # Normalize all columns
	
	#print allX
	allY = [[x] for x in np.asarray(df[target_column])]	#target column
	
	targets = []
	features = []
	
	for hour in xrange(n_steps):
		training_row = []
		testing_row = []
		for i in xrange(nrows/n_steps):
			idx = i*n_steps + hour
			training_row.append(np.squeeze(np.asarray(allX[idx])))
			testing_row.append(allY[idx])
		features.append(training_row)
		targets.append(testing_row)

	#print features
	#print targets
		
	return np.asarray(features),np.asarray(targets)

# Test the trained model
def testRNN(test_start, test_end, model, features, targets, use_mape = True):
	mapes = []
	maes = []
	all_predict =[]
	all_target = []
	for idx in xrange(test_start, test_end):
		guess = model.predict(features[:, idx, :][:, np.newaxis, :])
		guess = [j for x in guess for j in x]
		error = abs(targets[:,idx,:] - guess)
		maes.append(error)
		mapes.append(np.mean(error/abs(targets[:,idx,:])))
		all_predict += [j for x in guess for j in x]
		all_target += [j for x in targets[:,idx,:] for j in x]

	d = {'target': all_target, 'predict': all_predict}
	avg_mape = np.mean(mapes)
	avg_mae = np.mean(maes)
	print 'Avg {0} from Day {1} to {2} = {3}'.format("MAPE" if use_mape else "MAE", test_start, test_end, avg_mape if use_mape else avg_mae)
	return pd.DataFrame(data=d), avg_mape if use_mape else avg_mae

# Perform whole round of training + testing for a particular market like PJM or MISO
def train_and_test(market, start_hour, training_days, testing_days, n_hidden, n_epochs):
	dir = "data\\{0}\\".format(market)
	files = [f for f in listdir(dir) if isfile(join(dir, f))]	# all the testing files

	result_dir = '{0}Result\\{1}'.format(dir, time.time()) # result dir with timestamp
	os.mkdir(result_dir)
	avg_errs = []
	use_mape = MarketStudyParams[market][2]
	target_column = MarketStudyParams[market][0]
	
	for file in files:
		print "Testing file {0}...".format(file)
		errs = []
		result_df = pd.DataFrame()
		rawdata_df = load(dir+file, market)
		
		features,targets = prepareAllData(rawdata_df, target_column, n_steps = hours_in_day)
		total_days = features.shape[1]
		total_features = len(rawdata_df.columns)-1
		
		print 'Total days = {0}, # of features = {1}'.format(total_days,total_features)
		model = MetaRNN(n_in=total_features, n_hidden=n_hidden, n_out=1,
							learning_rate=0.01, learning_rate_decay=0.99,
							n_epochs=n_epochs, batch_size=100, activation='sigmoid', L1_reg = 0,
							L2_reg=0)
		#Training
		
		training_start = start_hour
		training_end = training_start + training_days-1
		model.fit(features[:,training_start:training_end,:], targets[:,training_start:training_end,:], validate_every=100, optimizer='bfgs')
		#pickle.dump(model, open( "{0}\\model.p".format(result_dir), "wb" ) )
		
		df, avg_mape = testRNN(training_start, training_end, model, features, targets, use_mape)
		df.to_csv('{0}\\{1}_training_result.csv'.format(result_dir, file))
		
		for testing_start in xrange(start_hour+training_days, total_days, testing_days):
			#training_start = testing_start - training_days
			#training_end = testing_start-1
			
			#print "\nTraining..."
			#model.fit(features[:,training_start:training_end,:], targets[:,training_start:training_end,:], validate_every=100, optimizer='bfgs')
			#training_df, training_mae = testRNN(training_start, training_end, model, features, targets, use_mape)

			testing_end = min(testing_start + testing_days - 1, total_days)
			print "Testing..."
			df, err = testRNN(testing_start, testing_end, model, features, targets, use_mape)
			result_df = pd.concat([result_df, df])
			errs.append(err)
		
		avg_err = np.mean(errs)
		print 'For {0}, avg {1} = {2}'.format(file, "MAPE" if use_mape else "MAE", avg_err)
		
		avg_errs.append(avg_err)
		result_df.to_csv('{0}\\{1}_result.csv'.format(result_dir, file))
			
	print 'Average {0} for {1} = {2}'.format("MAPE" if use_mape else "MAE", market, np.mean(avg_errs))

	
def main(argv):
	if len(argv) < 7:
		print '\nSorry - invalid parameters.\nUsage:\n==========Params=========\nStart Hour \n# of training days \n# of testing days \n# of hidden nodes \n# of iterations \nMarket(PJM|MISO|SPP)\n====================='
		return
	start_hour = int(argv[1])
	training_days = int(argv[2])
	testing_days = int(argv[3])
	n_hidden = int(argv[4])
	n_epochs = int(argv[5])
	print '\n==========Params=========\nStart Hour = {0}\n# of training days = {1} \n# of testing days = {2} \n# of hidden nodes = {3} \n# of iterations = {4}\n=========================\n'.format(start_hour, training_days, testing_days, n_hidden, n_epochs)
	for market in argv[6:]:
		train_and_test(market, start_hour, training_days, testing_days, n_hidden, n_epochs)

if __name__ == "__main__":
    main(sys.argv)