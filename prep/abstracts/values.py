from prep.StateInterval import StateInterval

def get_value(val, min_val, max_val, lab):
	if val < min_val:
		value_abstraction = 'L'
	elif val > max_val:
		value_abstraction = 'H'
	else:
		value_abstraction = 'N'
	
	return value_abstraction + '_' + lab

def get_value_SI(val, min_val, max_val, lab, date):
	s = get_value(val, min_val, max_val, lab)
	b = e = date
	return StateInterval(s,b,e)