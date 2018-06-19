from ui.context_sensitive.raw2processed import Raw2Processed

from tkinter import *
import tkinter.filedialog
from ttk import *

class Raw2Patterns(Raw2Processed):

	def make_frame(self, parent):
		f = Frame(parent)
		
		self.knowledge, self.knowledge_btn = self.make_checkbutton(f, 'knowledge driven only (N/A)', 0, 0)
		self.values['knowledge-driven'] = self.knowledge
		self.buttons['knowledge-driven'] = self.knowledge_btn
		self.knowledge_btn.configure(command=lambda : self.toggle_other_buttons('knowledge-driven'), state=DISABLED)

		self.antiknowledge, self.antiknowledge_btn = self.make_checkbutton(f, 'exclude usual suspects', 1, 0)
		self.values['anti-knowledge-driven'] = self.antiknowledge
		self.buttons['anti-knowledge-driven'] = self.antiknowledge_btn

		self.medication, self.medication_btn = self.make_checkbutton(f, 'include medication', 2, 0)
		self.values['medication'] = self.medication
		self.buttons['medication'] = self.medication_btn
		# self.medication_btn.configure(command=lambda : self.no_knowledge(self.medication))

		self.consults, self.consults_btn = self.make_checkbutton(f, 'include consults', 3, 0)
		self.values['consults'] = self.consults
		self.buttons['consults'] = self.consults_btn
		# self.consults_btn.configure(command=lambda : self.no_knowledge(self.consults))

		self.actions, self.actions_btn = self.make_checkbutton(f, 'include actions', 4, 0)
		self.values['actions'] = self.actions
		self.buttons['actions'] = self.actions_btn
		# self.actions_btn.configure(command=lambda : self.no_knowledge(self.actions))

		self.icpc, self.icpc_btn = self.make_checkbutton(f, 'include icpc', 5, 0)
		self.values['icpc'] = self.icpc
		self.buttons['icpc'] = self.icpc_btn
		# self.icpc_btn.configure(command=lambda : self.no_knowledge(self.icpc))

		self.lab_results, self.lab_results_btn = self.make_checkbutton(f, 'include lab results', 6, 0)
		self.values['lab_results'] = self.lab_results
		self.buttons['lab_results'] = self.lab_results_btn
		# self.lab_results_btn.configure(command=lambda : self.no_knowledge(self.lab_results))

		self.smoking, self.smoking_btn = self.make_checkbutton(f, 'include smoking', 7, 0)
		self.values['smoking'] = self.smoking
		self.buttons['smoking'] = self.smoking_btn

		self.bmi, self.bmi_btn = self.make_checkbutton(f, 'include bmi', 8, 0)
		self.values['bmi'] = self.bmi
		self.buttons['bmi'] = self.bmi_btn

		self.allergies, self.allergies_btn = self.make_checkbutton(f, 'include allergies', 9, 0)
		self.values['allergies'] = self.allergies
		self.buttons['allergies'] = self.allergies_btn

		self.blood_pressure, self.blood_pressure_btn = self.make_checkbutton(f, 'include blood pressure', 10, 0)
		self.values['blood_pressure'] = self.blood_pressure
		self.buttons['blood_pressure'] = self.blood_pressure_btn

		self.alcohol, self.alcohol_btn = self.make_checkbutton(f, 'include alcohol', 11, 0)
		self.values['alcohol'] = self.alcohol
		self.buttons['alcohol'] = self.alcohol

		self.renal_function, self.renal_function_btn = self.make_checkbutton(f, 'include renal function', 12, 0)
		self.values['renal_function'] = self.renal_function
		self.buttons['renal_function'] = self.renal_function

		self.cardiometabolism, self.cardiometabolism_btn = self.make_checkbutton(f, 'include cardiometabolism', 13, 0)
		self.values['cardiometabolism'] = self.cardiometabolism
		self.buttons['cardiometabolism'] = self.cardiometabolism

		self.lab_blood, self.lab_blood_btn = self.make_checkbutton(f, 'include lab blood', 14, 0)
		self.values['lab_blood'] = self.lab_blood
		self.buttons['lab_blood'] = self.lab_blood

		self.lung_function, self.lung_function_btn = self.make_checkbutton(f, 'include lung function', 15, 0)
		self.values['lung_function'] = self.lung_function
		self.buttons['lung_function'] = self.lung_function

		self.support_val, self.support = self.make_label(f, 'Min. support', 16, 0)
		self.values['support'] = self.support_val
		# self.buttons['lab_results'] = self.lab_results_btn
		# self.lab_results_btn.configure(command=lambda : self.no_knowledge(self.lab_results))

		self.patterns, self.patterns_btn = self.make_checkbutton(f, 'I already have the sequences', 17, 0)
		self.values['sequences_available'] = self.patterns
		self.buttons['sequences_available'] = self.patterns_btn
		self.patterns_btn.configure(command=lambda : self.sequence_mode(f, 'sequences_available'))

		self.add_file_browser(f, 'Browse', '', 9, 0, mode=DISABLED)

		return f

	def sequence_mode(self, f, button_key):
		self.toggle_other_buttons(button_key)

		enabled = self.values[button_key].get()
		if enabled:
			for w in self.sequence_widgets:
				w.config(state=NORMAL)
		else:
			for w in self.sequence_widgets:
				w.config(state=DISABLED)

	def add_file_browser(self, f, button_txt, init_txt , r, c, help_txt='', mode=NORMAL):
		b = Button(f,text=button_txt)
		b.config(state=DISABLED)
		b.grid(row=r, column=c, sticky=W+E)
		f_var = StringVar()
		f_var.set(init_txt)
		e = Entry(f, textvariable=f_var)
		e.config(state=DISABLED)
		e.grid(row=r, column=c+1)
		b.configure(command=lambda: self.browse_file(f_var, b))

		self.sequence_widgets = [b, e]
		self.values['sequence_file'] = f_var

	def browse_file(self, dir_var, button):
		s = filedialog.askopenfilename(parent=button)
		dir_var.set(s)


