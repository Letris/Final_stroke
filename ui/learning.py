from ui.Tab import PipelineTab
from tkinter import LEFT, BooleanVar, W, NORMAL, DISABLED
from ttk import Label, Checkbutton
from learn.learn import *
# import learn.learn
import time
import util_.util as util
from collections import defaultdict

class LearningTab(PipelineTab):

	def __init__(self, parent, title, master):
		'''initialize object'''
		self.buttons = defaultdict(dict)
		PipelineTab.__init__(self, parent, title, master)

	def init_components(self):
		'''inits learning frame's components (buttons, fields, labels, underlying methods)'''

		dct = self.user_input

		# I/O components
		dct['in_dir'] = self.button_component('Browse', 'input folder', 0, 0)
		dct['out_dir'] = self.button_component('Browse', 'output folder', 1, 0)

		# input the record and target id (e.g. 'patientnummer' and 'stroke')
		dct['record_id'] = self.general_component('ID column name', 2, 0)

		# checkbox for activating survival mode
		self.survival, self.survival_btn = self.make_checkbutton(self, 'survival', 3, 0)
		self.user_input['survival'] = self.survival
		self.buttons['survival'] = self.survival_btn

		# checkbox for activating oversampling
		self.oversampling, self.oversampling_btn = self.make_checkbutton(self, 'oversampling', 4, 0)
		self.user_input['oversampling'] = self.oversampling
		self.buttons['oversampling'] = self.oversampling_btn

		self.undersampling, self.undersampling_btn = self.make_checkbutton(self, 'undersampling', 5, 0)
		self.user_input['undersampling'] = self.undersampling
		self.buttons['undersampling'] = self.undersampling_btn

		self.aggregation, self.aggregation_btn = self.make_checkbutton(self, 'aggregation', 6, 0)
		self.user_input['aggregation'] = self.aggregation
		self.buttons['aggregation'] = self.aggregation_btn

		# checkboxes allowing for algorithm selection
		self.DT, self.DT_btn = self.make_checkbutton(self, 'CART', 7, 0)
		self.user_input['DT'] = self.DT
		self.buttons['DT'] = self.DT_btn

		self.LR, self.LR_btn = self.make_checkbutton(self, 'LR', 8, 0)
		self.user_input['LR'] = self.LR
		self.buttons['LR'] = self.LR_btn

		self.RF, self.RF_btn = self.make_checkbutton(self, 'RF (100 Trees)', 9, 0)
		self.user_input['RF'] = self.RF
		self.buttons['RF'] = self.RF_btn

		self.RF, self.RF_btn = self.make_checkbutton(self, 'RF (10 Trees)', 10, 0)
		self.user_input['RFsmall'] = self.RF
		self.buttons['RFsmall'] = self.RF_btn

		self.SVM, self.SVM_btn = self.make_checkbutton(self, 'SVM', 11, 0)
		self.user_input['SVM'] = self.SVM
		self.buttons['SVM'] = self.SVM_btn

		self.XGBoost, self.XGBoost_btn = self.make_checkbutton(self, 'XGBoost', 12, 0)
		self.user_input['XGBoost'] = self.XGBoost
		self.buttons['XGBoost'] = self.XGBoost_btn

		self.COX, self.COX_btn = self.make_checkbutton(self, 'COX', 13, 0)
		self.user_input['COX'] = self.COX
		self.buttons['COX'] = self.COX_btn

		self.survSVM, self.survSVM_btn = self.make_checkbutton(self, 'survSVM', 14, 0)
		self.user_input['survSVM'] = self.survSVM
		self.buttons['survSVM'] = self.survSVM_btn

		self.GBS, self.GBS_btn = self.make_checkbutton(self, 'GBS', 15, 0)
		self.user_input['GBS'] = self.GBS
		self.buttons['GBS'] = self.GBS_btn

		Label(self, text='   ').grid(row=16, column=0, columnspan=2)
		
		self.FS, self.FS_btn = self.make_checkbutton(self, 'apply feature selection', 17, 0)
		self.user_input['FS'] = self.FS
		self.buttons['FS'] = self.FS_btn

		Label(self, text='   ').grid(row=18, column=0, columnspan=2)
		Label(self, text='If you want a separate testset, \nfill in the following fields.').grid(row=19, column=0, columnspan=2, sticky=W)

		self.sep_test, self.sep_test_btn = self.make_checkbutton(self, 'separate testset (do not use; untested)', 20, 0)
		self.user_input['sep_test'] = self.sep_test
		self.buttons['sep_test'] = self.sep_test_btn

		Label(self, text='input folder for test').grid(row=21, column=0, columnspan=2, sticky=W)
		dct['in_dir_test'] = self.button_component('Browse', 'input folder', 21, 0)
		
		# Label(self, text='HISes used for training (rest = testing)').grid(row=19, column=0, columnspan=2, sticky=W)
		# self.setup_HIS_choice()
		
		# setup algorithm launcher button (incl defaults button)
		self.setup_launcher()

		# compile
		self.pack()

	def make_checkbutton(self, f, s, r, c):
		v = BooleanVar()
		ch = Checkbutton(f, text=s, variable=v)
		ch.grid(row=r, column=c, columnspan=2, sticky=W)
		return v, ch

	def defaults(self):
		'''set the user_input dict to default values'''
		dct = self.user_input

		dct['in_dir'].set('/Users/Tristan/Downloads/merged/non-survival/nonsurvnonall')
		dct['out_dir'].set('/Users/Tristan/Downloads/merged/')
		dct['record_id'].set('ID')
		dct['survival'].set(False)
		dct['oversampling'].set(False)
		dct['undersampling'].set(False)
		dct['aggregation'].set(False)
		dct['DT'].set(False)
		dct['LR'].set(False)
		dct['RF'].set(False)
		dct['RFsmall'].set(False)
		dct['XGBoost'].set(False)
		dct['SVM'].set(False)
		dct['COX'].set(False)
		dct['survSVM'].set(False)
		dct['GBS'].set(False)
		dct['FS'].set(False)
		dct['in_dir_test'].set('./out/combined/test')

	def go(self, button):
		'''initiates the associated algorithms '''
		button.config(text='Running', state=DISABLED)
		self.master.update_idletasks()

		dct = self.user_input
		
		in_dir = dct['in_dir'].get()

		now = util.get_current_datetime()
		out_dir = dct['out_dir'].get() + '/' + now
		
		survival = dct['survival'].get()
		oversampling = dct['oversampling'].get()
		undersampling = dct['undersampling'].get()
		aggregation = dct['aggregation'].get()

		algorithms = []
		if dct['DT'].get(): algorithms.append('DT')
		if dct['LR'].get(): algorithms.append('LR')
		if dct['RF'].get(): algorithms.append('RF')
		if dct['RFsmall'].get(): algorithms.append('RFsmall')
		if dct['SVM'].get(): algorithms.append('SVM')
		if dct['XGBoost'].get(): algorithms.append('XGBoost')
		if dct['COX'].get(): algorithms.append('COX')
		if dct['survSVM'].get(): algorithms.append('survSVM')
		if dct['GBS'].get():algorithms.append('GBS')

		record_id = dct['record_id'].get().lower()
		target_id = 'target'
		feature_selection = dct['FS'].get()

		separate_testset = dct['sep_test'].get()
		# train_HISes = [dct['PMO'].get(), dct['MDM'].get(), dct['LUMC'].get(), 
					 # dct['VUMH'].get(), dct['VUMD'].get(), dct['VUSC'].get()]
		in_dir_test = dct['in_dir_test'].get()

		execute(in_dir, out_dir, record_id, target_id, algorithms, feature_selection, separate_testset, in_dir_test, survival, oversampling, undersampling, aggregation)

		button.config(text='Done')
		self.master.update_idletasks()
		time.sleep(0.5)	
		button.config(text='Run!', state=NORMAL)	
