import time
import util_.util as util
from ui.Tab import PipelineTab
from tkinter import DISABLED, NORMAL, StringVar, Radiobutton, W
from ttk import Label
import features.merge as combine

class MergeTab(PipelineTab):

	def init_components(self):
		'''inits merge frame's components (buttons, fields, labels, underlying methods)'''
		self.setup_IO_dirs()
		self.setup_general()

		Label(self, text='HISes as train (rest is test).\nOnly does something if "separate"\n button checked').grid(row=17, column=0, columnspan=2, sticky=W)
		self.user_input['separate'] = self.checkbutton_component('Separate train/test', 18, 0)
		self.setup_HIS_choice()

		self.setup_launcher()
		self.pack()

	def setup_IO_dirs(self):
		'''add I/O part'''
		dct = self.user_input

		dct['in_dir'] = self.button_component('Browse', 'input folder', 0, 0)
		dct['delimiter'] = self.general_component('Delimiter', 1, 0, init_val=',')
		dct['out_dir'] = self.button_component('Browse', 'output folder', 2, 0)
		dct['output_id'] = self.general_component('Output Name', 3, 0)

	def setup_general(self):
		'''add options part'''
		dct = self.user_input
		
		dct['age+gender'] = self.checkbutton_component('age+gender', 4, 0)
		dct['counts_med'] = self.checkbutton_component('counts (medication)', 5, 0)
		dct['counts_med_enrich'] = self.checkbutton_component('counts (medication enriched)', 6, 0)
		dct['counts_consult'] = self.checkbutton_component('counts (consults)', 7, 0)
		dct['counts_consult_enrich'] = self.checkbutton_component('counts (consults enriched)', 8, 0)
		dct['counts_referral'] = self.checkbutton_component('counts (actions)', 9, 0)
		dct['counts_lab'] = self.checkbutton_component('counts (lab)', 10, 0)
		dct['all_counts'] = self.checkbutton_component('all counts',11, 0 )
		dct['tmprl'] = self.checkbutton_component('temporal (excl. enriched)', 12, 0)
		dct['enriched_tmprl'] = self.checkbutton_component('temporal (incl. enriched)', 13, 0)
		dct['knowledge_driven'] = self.checkbutton_component('knowledge driven (Marshall)', 14, 0)
		dct['anti_knowledge_driven'] = self.checkbutton_component('EXCLUDE Marshall, counts', 15, 0)
		dct['anti_knowledge_driven_tmprl'] = self.checkbutton_component('EXCLUDE Marshall, temporal', 16, 0)
		dct['target'] = self.checkbutton_component('target', 17, 0, init_val=True, mode=DISABLED)
	
	def defaults(self):
		'''set the user_input dict to default values'''
		dct = self.user_input

		dct['in_dir'].set('./out/segments')
		dct['delimiter'].set(',')
		dct['out_dir'].set('./out/combined')
		dct['output_id'].set('counts')
		dct['age+gender'].set(True)
		dct['counts_med'].set(True)
		dct['counts_med_enrich'].set(False)
		dct['counts_consult'].set(True)
		dct['counts_consult_enrich'].set(False)	
		dct['counts_referral'].set(True)
		dct['counts_lab'].set(True)
		dct['all_counts'].set(False)
		dct['tmprl'].set(False)
		dct['enriched_tmprl'].set(False)
		dct['knowledge_driven'].set(False)
		dct['anti_knowledge_driven'].set(False)
		dct['anti_knowledge_driven_tmprl'].set(False)

	def go(self, button):
		'''initiates the associated algorithms '''
		dct = self.user_input

		button.config(text='Running', state=DISABLED)
		if dct['in_dir'].get() == 'input folder':
			dct['in_dir'].set('/Users/Reiny/Documents/UI_stroke/playground')
		if dct['delimiter'].get() == '':
			dct['delimiter'].set(',')
		if dct['out_dir'].get() == 'output folder':
			dct['out_dir'].set('/Users/Reiny/Documents/UI_stroke/out')

		self.master.update_idletasks()

		util.make_dir(dct['out_dir'].get() + '/')

		HISes = [dct['PMO'].get(), dct['MDM'].get(), dct['LUMC'].get(), 
			 dct['VUMH'].get(), dct['VUMD'].get(), dct['VUSC'].get()]

		args = [dct['in_dir'].get(), 
				dct['delimiter'].get(),
				dct['out_dir'].get() + '/' + dct['output_id'].get() + '.csv', 
				dct['age+gender'].get(), 
				dct['counts_med'].get(), 
				dct['counts_med_enrich'].get(), 
				dct['counts_consult'].get(), 
				dct['counts_consult_enrich'].get(), 
				dct['counts_referral'].get(), 
				dct['counts_lab'].get(),
				dct['all_counts'].get(),
				dct['tmprl'].get(), 
				dct['enriched_tmprl'].get(),
				dct['knowledge_driven'].get(),
				dct['anti_knowledge_driven'].get(),
				dct['anti_knowledge_driven_tmprl'].get(),
				dct['separate'].get(),
				HISes
		]

		# merge
		combine.execute(*args)

		button.config(text='Done')
		self.master.update_idletasks()
		time.sleep(0.5)	
		button.config(text='Run!', state=NORMAL)

