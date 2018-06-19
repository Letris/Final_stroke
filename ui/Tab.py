from tkinter import * # W, E, StringVar, BooleanVar, NORMAL, Scrollbar, Listbox, RIGHT, Y, LEFT, BOTH
from ttk import Frame, Entry, Label, Button, Checkbutton
from tkinter.filedialog import askdirectory, askopenfilename
from collections import defaultdict

class Tab(Frame):
	
	def __init__(self, parent, title, master):
		'''sets self to Frame with parent parent, initializes visuals and interactions, adds tab to parent notebook'''
		Frame.__init__(self, parent)
		self.master = master
		self.init_components()
		self.add_to(parent, title)


	def init_components(self):
		'''abstract method to be implemented by the derived class'''

	def add_to(self,parent, title):
		'''add self to parent with title title'''
		parent.add(self, text=title)

class PipelineTab(Tab):
	
	def __init__(self, parent, title, master):
		'''initialize object'''
		self.user_input = defaultdict(dict)
		Tab.__init__(self, parent, title, master)


	### methods which add often occurring combinations of interface components using the grid layout ###

	def setup_HIS_choice(self):
		dct = self.user_input

		dct['PMO'] = self.checkbutton_component('Utrecht Promedico', 20, 0, init_val='PMO', onvalue='PMO', offvalue='')
		dct['MDM'] = self.checkbutton_component('Utrecht Medicom', 21, 0, init_val='MDM', onvalue='MDM', offvalue='')
		dct['LUMC'] = self.checkbutton_component('Leiden', 22, 0, init_val='LUMC', onvalue='LUMC', offvalue='')
		dct['VUMH'] = self.checkbutton_component('Amsterdam MicroHIS', 23, 0, init_val='VUMH', onvalue='VUMH', offvalue='')
		dct['VUMD'] = self.checkbutton_component('Amsterdam Medicom', 24, 0, init_val='VUMD', onvalue='VUMD', offvalue='')
		dct['VUSC'] = self.checkbutton_component('Amsterdam Scipio', 25, 0, init_val='VUSC', onvalue='VUSC', offvalue='')

	def button_component(self, button_txt, init_txt, r, c, help_txt='', mode=NORMAL):
		'''adds a button component associated with an entry (the label only activates when help_txt != '')'''
		b = Button(self, text=button_txt)
		b.grid(row=r, column=c, sticky=W+E)
		b.config(state=mode)
		dir_var = StringVar()
		dir_var.set(init_txt)
		Entry(self, textvariable=dir_var).grid(row=r, column=c+1)
		Label(self, text=help_txt).grid(row=r, column=c+2, sticky=W)
		b.configure(command=lambda: self.browse_dir(dir_var, b))
		return dir_var

	def checkbutton_component(self, s, r, c, init_val=False, mode=NORMAL, onvalue=-1, offvalue=-1):
		'''adds a checkbutton, and its associated variable'''
		if onvalue == -1:
			var = BooleanVar()
			var.set(init_val)
			btn = Checkbutton(self,text=s, variable=var)
		else:
			var = StringVar()
			var.set(init_val)
			btn = Checkbutton(self,text=s, variable=var, onvalue=onvalue, offvalue=offvalue)			
		btn.grid(row=r, column=c, columnspan=2, sticky=W)	
		btn.config(state=mode)
		return var		

	def browse_dir(self, dir_var, button):
		'''called to browse a directory'''
		s = askdirectory(parent=button)
		dir_var.set(s)

	def general_component(self, s, r, c, init_val='', help_txt=''):
		'''adds a label, plus an entry and its associated variable'''
		Label(self, text=s).grid(row=r, column=c, sticky=W)
		var = StringVar()
		var.set(init_val)
		Entry(self, textvariable=var).grid(row=r, column=c+1)	
		Label(self, text=help_txt).grid(row=r, column=c+2)
		return var		

	def setup_launcher(self):
		'''create buttons to execute the job and for default values'''

		def_button = Button(self,text='Defaults')
		def_button.grid(row=30, column=0, padx=5, pady=5, sticky=W)
		def_button.configure(command=self.defaults)

		go_button = Button(self,text="Run!")
		go_button.grid(row=30, column=1, padx=5, pady=5, sticky=E)
		go_button.configure(command=lambda: self.go(go_button))

	def defaults(self):
		'''abstract method called when the 'defaults' button is pressed'''
	
	def go(self, button):
		'''abstract method called when the 'go' button is pressed'''

