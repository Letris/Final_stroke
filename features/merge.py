import util_.in_out as io
import util_.util as util
from util_.util import make_dir
from collections import defaultdict

def execute(in_dir, delim, out_file, age_gender=False, counts_med=False, 
				counts_med_enrich=False, counts_consult=False, counts_consult_enrich=False,
				counts_referral=False, counts_lab=False, all_counts=False, tmprl=False,
				enriched_tmprl=False, knowledge_driven=False, counts_no_knowledge=False, tmprl_no_knowledge=False,
				separate=False, HISes=[]):
	'''merge the in files to produce the out file'''
	merged = defaultdict(list)
	headers = ['ID']

	# we may not need this.
	ID2HIS = {}
	merged_test = defaultdict(list)

	# if we wish to separate, get dictionary of patient HIS sources using SQL.
	if separate:
		c = util.sql_connect().cursor()
		HISes_str = "','".join(HISes)
		q = '''SELECT patientnummer 
				FROM patienten
				WHERE PRAKTIJKCODE IN ('{}')'''.format(HISes_str)
		c.execute(q)
		
		ID2HIS = {row[0] : row[0] for row in c}

	if age_gender:
		headers = merge_file(in_dir+'/AG.csv', merged, headers, delim, separate, ID2HIS, merged_test)
		
	if counts_med:
		headers = merge_file(in_dir+'/C_M.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if counts_med_enrich:
		headers = merge_file(in_dir+'/C_M_enrich.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if counts_consult:
		headers = merge_file(in_dir+'/C_C.csv', merged, headers, delim, separate, ID2HIS, merged_test)
	
	if counts_consult_enrich:
		headers = merge_file(in_dir+'/C_C_enrich.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if counts_referral:
		headers = merge_file(in_dir+'/C_R.csv', merged, headers, delim, separate, ID2HIS, merged_test)
	
	if counts_lab:
		headers = merge_file(in_dir+'/C_L.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if tmprl:
		headers = merge_file(in_dir+'/T.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if enriched_tmprl:
		headers = merge_file(in_dir+'/T_enrich.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if knowledge_driven:
		headers = merge_file(in_dir+'/K.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if counts_no_knowledge:
		headers = merge_file(in_dir+'/C_NK.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if tmprl_no_knowledge:
		headers = merge_file(in_dir+'/T_NK.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	if all_counts:
		print('ja')
		headers = merge_file(in_dir+'/counts.csv', merged, headers, delim, separate, ID2HIS, merged_test)

	headers = merge_file(in_dir+'/stroke.csv', merged, headers, delim, separate, ID2HIS, merged_test)
	
	# now write to new file (also check whether all results have same length)
	make_dir(out_file)
	out = io.write_csv(out_file)

	out.writerow(headers)
	skip=0

	for key in merged:
		if len(headers) != 1+len(merged[key]):
			print ('unequal to header amount ({} vs {})! watch out.'.format(len(headers),len(merged[key])))
			# skip+=1
			# continue
		out.writerow([key] + merged[key])

	if separate:
		out_file_test = out_file[:out_file.rfind('/')+1] + 'test' + out_file[out_file.rfind('/'):]
		make_dir(out_file_test)

		out = io.write_csv(out_file_test)
		
		out.writerow(headers)
		for key in merged_test:
			if len(headers) != 1+len(merged_test[key]):
				print ('unequal to header amount ({} vs {})! watch out.'.format(len(headers),len(merged_test[key])))
				# skip+=1
				# continue
			out.writerow([key] + merged_test[key])

	print ('## Done Merging ##')

def merge_file(f, merged, headers, delim, separate, ID2HIS, merged_test):
	try:
		rows = io.read_csv2(f, delim)
	except:
		print ('{} does not exist, choose a different directory or exclude the specified file from merging. Skipped for now.'.format(f))
		return headers
		
	headers = headers + next(rows)[1:]
	print(headers)
	if not separate:
		for row in rows:
			ID = row[0]
			merged[ID] = merged[ID] + row[1:]
			# print(merged)
		return headers

	# we separate test and training sets
	else:
		for row in rows:
			ID = row[0]
			if ID in ID2HIS:
				merged[ID] = merged[ID] + row[1:]
			else:
				merged_test[ID] = merged_test[ID] + row[1:] 
		return headers


