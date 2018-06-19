from datetime import timedelta

def get_trends(measurement_id, time_points):
	
	# separate dates and values (x, y)
	dates = list(zip(*time_points))[0]
	vals = list(zip(*time_points))[1]

	# calculate the boundary where 'steady' becomes 'in-' or 'decreasing'
	min_val, max_val = min(vals), max(vals)
	thresh = abs(max_val - min_val) * 0.1

	# calculate the date intervals and corresponding value differences
	intervals = zip(dates[:-1], dates[1:])
	vals_diff = zip(vals[:-1], vals[1:])

	# representation of a single day, to play with the end date of interval
	one_day = timedelta(days=1)

	# for each segment, get the begin/end dates, 
	# generate the state, add to list
	abstractions = []
	for i, v in zip(intervals, vals_diff):
		b = i[0] # begin date
		e = i[1] - one_day	 # end date

		# generate state
		v = v[1] - v[0]
		if v > thresh:
			s = 'I_'+measurement_id
		elif v < -thresh:
			s = 'D_'+measurement_id
		else:
			s = 'S_'+measurement_id

		# if the state has the same tonicity as the previous
		# simply change the endtime of the previous segment
		# else add a new segment
		if len(abstractions) > 0 and abstractions[-1][0] == s:
			abstractions[-1][2] = e
		else:
			abstractions.append([s,b,e])

	# change the final abstraction to include the last day
	final_abstr = abstractions[-1]
	final_abstr[2] = final_abstr[2] + one_day

	return abstractions