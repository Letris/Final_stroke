from ui.Tab import Tab
from tkinter import LEFT
from ttk import Label

class AboutTab(Tab):

	def init_components(self):
		'''add disclaimer-ish kind of text and contact info'''
		disclaimer = '''This software may be used to obtain knowledge\nfrom medical data. Though it was tested and\nsuccessfully used by myself, usage is at your\nown risk. For any questions, contact me at\nr.kop@vu.nl.\n\n- Reinier Kop'''
		l = Label(self, justify=LEFT, text=disclaimer)
		l.pack()
		self.pack()

