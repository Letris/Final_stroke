from ui.context_sensitive.raw2processed import Raw2Processed
from tkinter import *
from ttk import *

class Raw2Attributes(Raw2Processed):

	def make_frame(self, parent):
		f = Frame(parent)
		
		
		self.knowledge, self.knowledge_btn = self.make_checkbutton(f, 'knowledge driven only', 0, 0)
		self.values['knowledge-driven'] = self.knowledge
		self.buttons['knowledge-driven'] = self.knowledge_btn
		self.knowledge_btn.configure(command=self.knowledge_only)

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

		return f