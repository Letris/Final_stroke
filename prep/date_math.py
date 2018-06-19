
from datetime import date,timedelta,datetime
import random
import dateparser

class PatientInterval():
	'''contains the functions to generate random or non-random patient intervals'''

	def __init__(self, seed):
		'''init object by setting the random seed'''
		random.seed(seed)

	# generates a random patient interval betwee either the specified dates or, 
	# if more applicable, the prespecified, actual start- and enddates of 
	# the received dataset (used for non-stroke cases)
	def randomize(self, patient_begin, patient_end, interval, verbose=False):
		fixed_begin = date(2006, 1, 1)
		fixed_end = date(2011, 12, 31)
		begin_measurement = timedelta(days=interval[1])
		end_measurement = timedelta(days=interval[0])

		min_begin = max(patient_begin, fixed_begin) # the latest of the two is our earliest starting point
		max_begin = min(patient_end, fixed_end) - begin_measurement # the earliest of the two minus the interval is our latest starting point

		if min_begin > max_begin: # data in the specified interval is not available
			# print '...............'
			# print 'skipping a non-stroke patient: it lacks ' + str((min_begin - max_begin).days) + " days"
			# print patient_begin, patient_end
			# print fixed_begin, fixed_end
			return False

		choose_from = max_begin - min_begin	
		random_day = timedelta(days=int(random.random()*(choose_from.days+1)))

		begin = min_begin + random_day
		end = begin + (begin_measurement-end_measurement)

		if verbose:
			print ('--------------')
			print (patient_begin, patient_end)
			print (fixed_begin, fixed_end)
			print (begin_measurement)
			print (end_measurement)
			print (min_begin, max_begin)
			print (choose_from)
			print (random_day)
			print (begin, end)

		return begin, end

	# generates a patient interval, when the final date of this interval is known (used for stroke cases)
	def calculate(self, enrollment, stroke_diagnosis_date, interval, verbose=False):
		one_day = timedelta(days=1)

		begin_measurement = timedelta(days=interval[1])
		end_measurement = timedelta(days=interval[0])

		min_begin = enrollment # the latest of the two is our earliest starting point
		print(stroke_diagnosis_date, begin_measurement, one_day)
		begin = stroke_diagnosis_date - begin_measurement - one_day # subtract with 1 day to exclude the day of diagnosis

		if min_begin > begin: # the specified interval is not available in the data
			# print 'skipping a stroke patient: it lacks ' + str((min_begin - begin).days) + " days"
			return False
		
		return begin, (stroke_diagnosis_date - end_measurement - one_day)

def four_weeks():
	return timedelta(weeks=4)

# convert date string in the format yyyy-mm-dd to date object
def str2date(date_str, ymd=False, yy=False, mdy=False, give_default_begin=False, give_default_end=False):
	

	if type(date_str) == date:
		return date_str 

	if type(date_str) == datetime:
		return date_str.date()

	if date_str == None or date_str == "" or (type(date_str) == str and date_str.strip() == ''):
		if give_default_begin:
			# return str2date("1850-01-01")
			return str2date("01-01-1850")

		elif give_default_end:
			# return str2date("2050-12-31")
			return str2date("31-12-2015")

	else:
			# print 'str2date encountered a value "{}" of type "{}" which it cannot handle'.format(date_str, type(date_str))

			date_lst = dateparser.parse(date_str)

			if yy:
				year = date_lst[2]
				year = '20'+year if int(year) < 70 else '19'+year
				return date(int(year), int(date_lst[1]), int(date_lst[0]))

			else:
				# print(date(date_lst.year, date_lst.month, date_lst.day))
				return date(date_lst.year, date_lst.month, date_lst.day)
	

def get_default_dates():
	return str2date("1850-01-01"), str2date("2050-12-31")

def get_dates(b, e):
	min_date, max_date = get_default_dates()
	b = b.date() if b != None else min_date
	e = e.date() if e != None else max_date
	return b, e

if __name__ == "__main__":
	patient_begin = str2date("2003-07-03")
	patient_end = str2date("2007-05-15")

	result = generate_random_patient_interval(patient_begin, patient_end, interval=365/2, verbose=False)
	print (result)
	result = generate_patient_interval(patient_begin, patient_end, interval=365/2, verbose=False)
	print (result)


