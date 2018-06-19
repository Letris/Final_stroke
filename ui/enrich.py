import time

import util_.util as util
import util_.in_out as io

from ui.Tab import PipelineTab
from tkinter import LEFT, BooleanVar, DISABLED, W, E, NORMAL
from ttk import Button, Label

from enrichment import ATC, ICPC, LAB, FrequencyCounter

class EnrichTab(PipelineTab):

	def init_components(self):
		'''inits enrich frame's components (buttons, fields, labels, underlying methods)'''
		self.setup_IO_dirs()
		self.setup_query_segment()
		self.setup_general()
		self.setup_launcher()
		self.setup_checkbuttons()
		self.setup_HIS_choice()
		self.pack()

	def setup_query_segment(self):
		'''set up SPARQL query segment'''
		dct = self.user_input

		# dct['out_dir'] = self.button_component('Browse', 'N/A', 0, 0, mode=DISABLED)

		sparql_btn = Button(self,text='Execute SPARQL Enrichment')
		sparql_btn.grid(row=4, column=0, columnspan=2, sticky=W)
		sparql_btn.configure(command=lambda : self.sparql_extraction(sparql_btn))

	def setup_IO_dirs(self):
		'''add I/O part'''
		dct = self.user_input

		dct['in_dir'] = self.button_component('Browse', 'input folder', 1, 0)
		dct['mapping_dir'] = self.button_component('Browse', 'SPARQL result dir', 2, 0)
		dct['delimiter'] = self.general_component('Delimiter', 3, 0, init_val=',')

	def setup_HIS_choice(self):
		dct = self.user_input

		Label(self, text='HISes to consider (only when using SQL):').grid(row=18, column=0, columnspan=2, sticky=W)
		dct['PMO'] = self.checkbutton_component('Utrecht Promedico', 19, 0, init_val='PMO', onvalue='PMO', offvalue='')
		dct['MDM'] = self.checkbutton_component('Utrecht Medicom', 20, 0, init_val='MDM', onvalue='MDM', offvalue='')
		dct['LUMC'] = self.checkbutton_component('Leiden', 21, 0, init_val='LUMC', onvalue='LUMC', offvalue='')
		dct['VUMH'] = self.checkbutton_component('Amsterdam MicroHIS', 22, 0, init_val='VUMH', onvalue='VUMH', offvalue='')
		dct['VUMD'] = self.checkbutton_component('Amsterdam Medicom', 23, 0, init_val='VUMD', onvalue='VUMD', offvalue='')
		dct['VUSC'] = self.checkbutton_component('Amsterdam Scipio', 24, 0, init_val='VUSC', onvalue='VUSC', offvalue='')

	def setup_general(self):
		'''add options part'''
		dct = self.user_input

		dct['min_age'] = self.general_component('Minimum age', 5, 0)
		dct['max_age'] = self.general_component('Maximum age', 6, 0)
		dct['begin_interval'] = self.general_component('First interval day', 7, 0, help_txt='')
		dct['end_interval'] = self.general_component('Last interval day', 8, 0, help_txt='')
		dct['ID_column'] = self.general_component('ID column name', 9, 0, init_val='patientnummer')
		dct['alpha'] = self.general_component('Alpha for test', 10, 0, init_val=0.2)

	def setup_checkbuttons(self):
		'''set up buttons to choose which source to enrich'''
		dct = self.user_input

		dct['ATC'] = self.checkbutton_component('ATC Enrichment', 11, 0)
		dct['ICPC'] = self.checkbutton_component('ICPC Enrichment', 12, 0)
		dct['Lab'] = self.checkbutton_component('Lab Enrichment (N/A)', 13, 0, mode=DISABLED)

	def defaults(self):
		'''set the user_input dict to default values'''
		dct = self.user_input

		# dct['out_dir'].set('out/enrichment/')
		dct['in_dir'].set('sql')
		dct['mapping_dir'].set('../out/semantics/')
		dct['delimiter'].set(',')
		dct['min_age'].set(30)
		dct['max_age'].set(150)
		dct['begin_interval'].set(int(365./52*26+1))
		dct['end_interval'].set(int(365./52*0+1))
		dct['ID_column'].set('patientnummer')
		dct['alpha'].set(0.2)
		dct['ATC'].set(True)
		dct['ICPC'].set(False)
		dct['Lab'].set(False)

	def sparql_extraction(self,button):
		''''''
		dct = self.user_input
		if dct['in_dir'].get() == 'input folder':
			dct['in_dir'].set('/Users/Reiny/Documents/UI_stroke/playground')
		if dct['mapping_dir'].get() == 'SPARQL result dir':
			dct['mapping_dir'].set('out/enrichment/')
		if dct['delimiter'].get() == '':
			dct['delimiter'].set(',')
		if dct['min_age'].get() == '':
			dct['min_age'].set(18)
		if dct['max_age'].get() == '':
			dct['max_age'].set(150)
		if dct['begin_interval'].get() == '':
			dct['begin_interval'].set(int(365./52*38))
		if dct['end_interval'].get() == '':
			dct['end_interval'].set(int(365./52*12))
		if dct['ID_column'].get() == '':
			dct['ID_column'].set('patientnummer')
		if dct['alpha'].get() == '':
			dct['alpha'].set(0.2)

		button.config(text='Running', state=DISABLED)
		self.master.update_idletasks()

		## enrich ##
		data_src = dct['in_dir'].get()
		export_dir = dct['mapping_dir'].get()		

		if dct['ATC'].get():
			atc = ATC()
			atc.enrich(data_src=data_src)
			atc.export(export_dir + '/atc/')
		if dct['ICPC'].get():
			icpc = ICPC()
			icpc.enrich(data_src=data_src)
			icpc.export(export_dir + '/icpc/')
		if dct['Lab'].get():
			lab = LAB()
			lab.enrich(data_src=data_src)
			lab.export(export_dir + '/lab/')
		print ('## SPARQL queries done & exported ##')
			
		button.config(text='Done')
		self.master.update_idletasks()
		time.sleep(0.5)	
		button.config(text='Execute SPARQL Enrichment', state=NORMAL)

	def go(self, button):
		'''initiates the associated algorithms '''
		dct = self.user_input
		if dct['in_dir'].get() == 'input folder':
			dct['in_dir'].set('/Users/Reiny/Documents/UI_stroke/playground')
		if dct['mapping_dir'].get() == 'SPARQL result dir':
			dct['mapping_dir'].set('out/enrichment/')
		if dct['delimiter'].get() == '':
			dct['delimiter'].set(',')
		if dct['min_age'].get() == '':
			dct['min_age'].set(18)
		if dct['max_age'].get() == '':
			dct['max_age'].set(150)
		if dct['begin_interval'].get() == '':
			dct['begin_interval'].set(int(365./52*38))
		if dct['end_interval'].get() == '':
			dct['end_interval'].set(int(365./52*12))
		if dct['ID_column'].get() == '':
			dct['ID_column'].set('patientnummer')
		
		button.config(text='Running', state=DISABLED)
		self.master.update_idletasks()
		
		## select frequent enrichments ##
		data_dir = dct['in_dir'].get()
		export_dir = dct['mapping_dir'].get()		
		subdirs = []
		if dct['ICPC'].get(): subdirs.append('icpc')
		if dct['ATC'].get(): subdirs.append('atc')
		if dct['Lab'].get(): subdirs.append('lab')
		skip = ['ingredients']
		
		HISes = [dct['PMO'].get(), dct['MDM'].get(), dct['LUMC'].get(), 
				 dct['VUMH'].get(), dct['VUMD'].get(), dct['VUSC'].get()]

		preprocess_args = [dct['in_dir'].get(), 
			dct['delimiter'].get(),
			dct['mapping_dir'].get(), 
			dct['ID_column'].get(),
			int(dct['min_age'].get()),
			int(dct['max_age'].get()),
			[int(dct['end_interval'].get()), int(dct['begin_interval'].get())],
			True if dct['in_dir'].get().lower() == 'sql' else False,
			HISes, 
			dct['mapping_dir'].get()]
		
		needs_processing = {'icpc': 0,
	                         'consults': dct['ICPC'].get(),
	                         'knowledge-driven': 0,
	                         'lab_results': dct['Lab'].get(),
	                         'medication': dct['ATC'].get(),
	                         'actions': 0}

		fc = FrequencyCounter(data_dir, export_dir, float(dct['alpha'].get()))
		fc.execute(subdirs, preprocess_args, needs_processing)

		button.config(text='Done')
		self.master.update_idletasks()
		time.sleep(0.5)	
		button.config(text='Run!', state=NORMAL)
