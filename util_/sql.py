def make_filling_query(tbl, first, last, values):
	'''build the query with which to fill the created tables'''
	q_head = 'INSERT INTO {} '.format(tbl)
	
	q_body = []
	for value in values:
		data = value['data']
		row = ["'" + str(el) + "'" if el != '' else 'NULL' for el in ([data[0]] + data[first:last])]
		row_str = ','.join(row)
		q_body.append('SELECT {} FROM dual'.format(row_str))

	q_body = ' UNION ALL '.join(q_body)
	q = q_head + q_body
	return q

def new_table(cursor, init_query, fill_query):
	'''make a new table in sql and fill it'''
	cursor.execute(init_query)
	cursor.execute(fill_query)

