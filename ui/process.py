from datetime import datetime
import time

import util_.util as util
import util_.in_out as io

from prep.StandardProcess import StandardProcess
from prep.SequenceProcess import SequenceProcess
from prep.MarshallProcess import MarshallProcess
from prep.NonMarshallSequenceProcess import NonMarshallSequenceProcess
from prep.NonMarshallProcess import NonMarshallProcess
from prep.EnrichProcesses import StandardEnrichProcess, SequenceEnrichProcess
from prep.generate_pattern_occurrences_per_patient import generate_pattern_occurrences_per_patient

from ui.Tab import PipelineTab
from tkinter import LEFT, BooleanVar, DISABLED, W, E, NORMAL, Scrollbar, Listbox, RIGHT, Y, LEFT, BOTH, Canvas, VERTICAL
from ttk import Label, Checkbutton, Radiobutton, Button
from ui.context_sensitive.raw2attributes import Raw2Attributes
from ui.context_sensitive.raw2patterns import Raw2Patterns

class ProcessTab(PipelineTab):

	def init_components(self):
		'''inits process frame's components (buttons, fields, labels, underlying methods)'''
		self.setup_IO_dirs()
		self.setup_general()
		self.setup_radio_buttons()
		# Label(self, text='HISes to consider (only when using SQL):').grid(row=19, column=0, columnspan=2, sticky=W)
		# self.setup_HIS_choice()
		self.setup_launcher()
		self.pack()

		

	def setup_IO_dirs(self):
		'''add I/O part'''
		dct = self.user_input

		dct['in_dir'] = self.button_component('Browse', 'input folder', 0, 0)
		dct['delimiter'] = self.general_component('Delimiter', 1, 0, init_val=',')
		dct['out_dir'] = self.button_component('Browse', 'output folder', 2, 0)

	def setup_general(self):
		'''add options part'''
		dct = self.user_input

		dct['min_age'] = self.general_component('Minimum age', 3, 0)
		dct['max_age'] = self.general_component('Maximum age', 4, 0)
		dct['begin_interval'] = self.general_component('First interval day', 5, 0, help_txt='')
		dct['end_interval'] = self.general_component('Last interval day', 6, 0, help_txt='')
		dct['ID_column'] = self.general_component('ID column name', 7, 0, init_val='patientnummer')

		enrich_val = BooleanVar()
		dct['enrich'] = enrich_val
		Checkbutton(self,text='semantic enrichment', variable=enrich_val).grid(row=8, column=0, columnspan=2, sticky=W)
		dct['mapping_dir'] = self.button_component('Browse', 'semantic enrichment dir', 9, 0)

		# verbose_val = BooleanVar()
		# dct['verbose'] = verbose_val
		# Checkbutton(self,text='verbose (N/A)', variable=verbose_val, state=DISABLED).grid(row=10, column=0, columnspan=2, sticky=W)

		survival_val = BooleanVar()
		dct['survival'] = survival_val
		Checkbutton(self,text='activate survival', variable=survival_val).grid(row=10, column=0, columnspan=2, sticky=W)

		already_processed_val = BooleanVar()
		dct['already_processed'] = already_processed_val
		Checkbutton(self,text='already_processed', variable=already_processed_val).grid(row=11, column=0, columnspan=2, sticky=W)

	def setup_radio_buttons(self):
		'''add atemporal vs temporal choice part'''
		dct = self.user_input

		temporal_processing_flag = BooleanVar()

		# get context dependent frame (regular)
		regular = Raw2Attributes()
		regular_frame = regular.make_frame(self)
		reg_button = Radiobutton(self, text='raw2attributes', value=False, variable=temporal_processing_flag)
		reg_button.grid(row=12, column=0, sticky=W)

		# get context dependent frame (temporal)
		temporal = Raw2Patterns()
		temporal_frame = temporal.make_frame(self)
		tmprl_button = Radiobutton(self, text='raw2patterns', value=True, variable=temporal_processing_flag)
		tmprl_button.grid(row=12, column=1, sticky=W)

		# configure events, invoke one by default
		reg_button.configure(command=lambda : self.set_frame(regular_frame, temporal_frame))
		tmprl_button.configure(command=lambda : self.set_frame(temporal_frame, regular_frame))
		reg_button.invoke() # default
		
		dct['process_temporal'] = temporal_processing_flag
		dct['a-temporal_specific'] = regular.get_values()
		dct['temporal_specific'] = temporal.get_values()

	def set_frame(self, new_f, old_f):
		'''set the context dependent frame, initiated by a push on a radio button'''
		old_f.grid_forget()
		new_f.grid(row=13, column=0, rowspan=6, columnspan=2, sticky=W)

	def defaults(self):
		'''set the user_input dict to default values'''
		dct = self.user_input

		dct['in_dir'].set('/Users/Tristan/Downloads/DWH TABELLEN/')
		dct['delimiter'].set(';')
		dct['out_dir'].set('/Users/Tristan/Downloads/EMR-pre-processing-pipeline-master/output folder')
		dct['min_age'].set(30)
		dct['max_age'].set(150)
		dct['begin_interval'].set(int(365./52*26+1))
		dct['end_interval'].set(int(365./52*0+1))
		dct['ID_column'].set('pseudopatnummer')
		dct['temporal_specific']['support'].set(0.1)
		dct['mapping_dir'].set('../out/semantics/')

		# dct['PMO'].set('PMO')
		# dct['MDM'].set('MDM')
		# dct['LUMC'].set('LUMC')
		# dct['VUMH'].set('VUMH')
		# dct['VUMD'].set('VUMD')
		# dct['VUSC'].set('VUSC')

	def go(self, button):
		'''initiates the associated algorithms '''
		dct = self.user_input

		button.config(text='Running', state=DISABLED)
		if dct['in_dir'].get() == 'input folder':
			dct['in_dir'].set('sql')
		if dct['delimiter'].get() == '':
			dct['delimiter'].set(',')
		if dct['out_dir'].get() == 'output folder':
			dct['out_dir'].set('./out')
		if dct['min_age'].get() == '':
			dct['min_age'].set(30)
		if dct['max_age'].get() == '':
			dct['max_age'].set(150)
		if dct['begin_interval'].get() == '':
			dct['begin_interval'].set(int(365./52*26+1))
		if dct['end_interval'].get() == '':
			dct['end_interval'].set(int(365./52*0+1))
		if dct['ID_column'].get() == '':
			dct['ID_column'].set('patientnummer')
		if dct['temporal_specific']['support'].get() == '':
			dct['temporal_specific']['support'].set(0.1)
		# if dct['mapping_dir'].get() == 'semantic enrichment dir':
		# 	dct['mapping_dir'].set('./out/semantics/')


		self.master.update_idletasks()

		now = util.get_current_datetime()
		util.make_dir(dct['out_dir'].get() + '/' + now + '/')

		# HISes = [dct['PMO'].get(), dct['MDM'].get(), dct['LUMC'].get(), 
		# 		 dct['VUMH'].get(), dct['VUMD'].get(), dct['VUSC'].get()]

		args = [dct['in_dir'].get(), 
				dct['delimiter'].get(),
				dct['out_dir'].get() + '/' + now, 
				dct['ID_column'].get(),
				int(dct['min_age'].get()),
				int(dct['max_age'].get()),
				[int(dct['end_interval'].get()), int(dct['begin_interval'].get())],
				True if dct['in_dir'].get().lower() == 'sql' else False,
				False, dct['survival'].get(), dct['already_processed'].get()]
			
		if dct['process_temporal'].get(): # process temporally
			self.temporal(dct, now, args)
		else: # process atemporally
			self.regular(dct, now, args)

		pretty_dct = util.tkinter2var(dct)
		try:
			io.pprint_to_file(dct['out_dir'].get() + '/' + now + '/settings.txt', pretty_dct)
		except IOError as e:
			print (e)

		print ('### Done processing ###')
		button.config(text='Done')
		self.master.update_idletasks()
		time.sleep(0.5)	
		button.config(text='Run!', state=NORMAL)

	def temporal(self, dct, now, args):
		needs_processing = {k : bool(v.get()) for k, v in dct['temporal_specific'].items()}

		out_dir = dct['out_dir'].get() + '/' + now + '/data/'
		util.make_dir(out_dir)
		# minimal support is set here
		min_sup = float(dct['temporal_specific']['support'].get())
		
		# if there are no sequences available
		if not dct['temporal_specific']['sequences_available'].get():
			# if enrichment is enabled, we create a different object instance than usual

			# if enriched
			# if dct['enrich'].get():
			# 	seq_p = SequenceEnrichProcess(*args, mapping_files_dir=dct['mapping_dir'].get())
			# 	name = 'sequences_enriched'
			# if not enriched and no marshall predictors
			if dct['temporal_specific']['anti-knowledge-driven'].get():
				seq_p = NonMarshallSequenceProcess(*args)
				name = 'sequences_excl_marshall'				
			else:
				seq_p = SequenceProcess(*args)
				name = 'sequences'

			seq_p.process(needs_processing)
			seq_p.sort_sequences()
			seq_p.save_output(sequence_file=True, sub_dir='data/tmprl', name=name)

			generate_pattern_occurrences_per_patient(out_dir, seq_p.id2data, min_sup, dct['mapping_dir'].get())
			sequence_f = out_dir + '/tmprl/{}.csv'.format(name)
		else:
			sequence_f = dct['temporal_specific']['sequence_file'].get()
			generate_pattern_occurrences_per_patient(out_dir, sequence_f, min_sup, dct['mapping_dir'].get())

	def regular(self, dct, now, args):	
		# non-temporally
		needs_processing = {k : bool(v.get()) for k, v in dct['a-temporal_specific'].items()}
		survival = dct['survival']
		knowledge_driven = dct['a-temporal_specific']['knowledge-driven'].get()

		# if only marshall predictors
		if knowledge_driven:
			std_p = MarshallProcess(*args)
			std_p.process(needs_processing)
			std_p.save_output(name='counts_knowledge_driven', sub_dir='data')
		# if enriched			
		# elif dct['enrich'].get():
		# 	std_p = StandardEnrichProcess(*args, mapping_files_dir=dct['mapping_dir'].get())
		# 	std_p.process(needs_processing)
		# 	std_p.save_output(name='counts_enriched', sub_dir='data')
		# if not enriched and no marshall predictors
		elif dct['a-temporal_specific']['anti-knowledge-driven'].get():
			std_p = NonMarshallProcess(*args)
			std_p.process(needs_processing)
			std_p.save_output(name='counts_excl_marshall', sub_dir='data')			
		else:
			std_p = StandardProcess(*args)
			std_p.process(needs_processing)
			std_p.save_output(name='counts', sub_dir='data')
			std_p.save_statistics(sub_dir='data', name='statistics')

		std_p.save_output(benchmark=True, sub_dir='data', name='age+gender')
		std_p.save_output(target=True, sub_dir='data', name='target')
