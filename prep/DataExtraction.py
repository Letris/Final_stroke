import re
from prep.date_math import PatientInterval, str2date
from datetime import date,timedelta,datetime
import numpy as np
# from date_math import generate_patient_interval, generate_random_patient_interval, str2date
from util_.in_out import write_csv, save_obj, load_obj
import util_.util as util
import util_.sql as sql
from tqdm import *
import pandas as pd
from collections import defaultdict
from numba import jit
import logging

class DataExtraction:

    def process_medication(self):
        med_f = util.select_file(self.files, 'medicatie') #selects 'medicatie' file
        rows, fields = util.import_data(med_f, delim=self.delim) #imports med_f data and separates headers from data (rows, fields(headers))
        #rows hier zijn vergelijkbaar met cursor bij SQL
        med_headers, self.num_med, self.num_med_pos, suf = self.insert_data(
                    rows, fields,
                    'atc',
                    ['dprescdate', 'dprescdate'],
                    '[A-Z][0-9][0-9]', 3,
                    suffix=['atc']
                    ) #insert_data
                                                                    
        self.headers = self.headers + med_headers


    def process_consults(self):
        consult_f = util.select_file(self.files, 'journaal') #selects 'journaal' file
        rows, fields = util.import_data(consult_f, delim=self.delim) #imports consult_f data and separates headers (fields) from data (rows)

        consult_headers, self.num_cons, self.num_cons_pos, suf = self.insert_data(
									rows, fields,
									'icpcprobleem',
									['regdatum', 'regdatum'],
									'[A-Z][0-9][0-9]', 3,
									suffix = ['consults']
									)

        self.headers = self.headers + consult_headers


    def process_actions(self):
        ref_f = util.select_file(self.files, 'verrichtingen') #selects 'verwijzingen' file
        rows, fields = util.import_data(ref_f, delim=self.delim) #imports ref_f data and separates headers (fields) from data (rows)

        ref_headers,_,_,_ = self.insert_data(
									rows, fields,
									'prestatiecode',
									['dverrdate', 'dverrdate'],
									'[0-9][0-9[0-9][0-9][0-9]', None,
									suffix=['actions']) #verrichtcode was prestatiecode
        self.headers = self.headers + ref_headers


    def process_icpc(self):
        comor_f = util.select_file(self.files, 'icpc')
        rows, fields = util.import_data(comor_f, delim=self.delim) #rows en fields zien er goed uit
        comor_headers,_,_,_ = self.insert_data(
									rows, fields,
									'icpc_cat',
									['dicpc_startdate', 'dicpc_enddate'],
									'[A-Z][0-9][0-9]', 3,
									suffix=['icpc'])
									#insert_data van sequenceprocess bij temporal, en van standardprocess bij regular
									#bij sequenceprocess gaat het dus fout!!!
        self.headers = self.headers + comor_headers	

    def process_labresults(self):
        lab_f = util.select_file(self.files, 'meetwaarden')
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'.+', None,
									suffix=['lab_results'])
        self.headers = self.headers + lab_headers

    def process_smoking(self):
        lab_f = util.select_file(self.files, 'roken')

        # gather PAKJAQ data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'PAKJAQ', None, 
									suffix=['smoking', 'lab_results'])

        self.headers = self.headers + lab_headers
        
        # gather smoking information
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'ROOKAQ', None,
									suffix=['smoking'], counter=1)

        self.headers = self.headers + lab_headers

    def process_bmi(self):
        lab_f = util.select_file(self.files, 'bmi')

		# gather length data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'LNG', 3,
									suffix=['bmi', 'lab_results'])

        self.headers = self.headers + lab_headers

         # gather weight data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'GEW', 3,
									suffix=['bmi', 'lab_results'], counter=1)
			
        self.headers = self.headers + lab_headers

		# gather bmi data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'QUE', 3,
									suffix=['bmi', 'lab_results'], counter=2)

        self.headers = self.headers + lab_headers

    def process_allergies(self):
        lab_f = util.select_file(self.files, 'allergie')

        # gather allergie data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'.+', None,
									suffix=['allergies'])

        self.headers = self.headers + lab_headers


    def process_bloodpressure(self):
        lab_f = util.select_file(self.files, 'bloeddruk')

		# gather allergie data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'RRD', 3,
									suffix=['blood_pressure', 'lab_results'])

        self.headers = self.headers + lab_headers

        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'RRS', 3,
									suffix=['blood_pressure', 'lab_results'])

        self.headers = self.headers + lab_headers


    def process_alcohol(self):
        lab_f = util.select_file(self.files, 'ggzanamnese')

        # gather allergie data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'ALCO', 4,
									suffix= ['alcohol', 'lab_results'])

        self.headers = self.headers + lab_headers


    def process_renalfunction(self):
        lab_f = util.select_file(self.files, 'klaring')

        # gather renal function data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'KREM', 4,
									suffix= ['renal_function', 'lab_results'])

        self.headers = self.headers + lab_headers


    def process_cardiometabolism(self):
        lab_f = util.select_file(self.files, 'cardiometabool')

        # gather renal function data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'[A-Z][A-Z][A-Z][A-Z][A-Z][A-Z]', None,
									suffix= ['cardiometabolism'])

        self.headers = self.headers + lab_headers

    
    def process_lab_blood(self):
        lab_f = util.select_file(self.files, 'labbloed')

        # gather renal function data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'.+', None,
									suffix= ['lab_blood', 'lab_results'])

        self.headers = self.headers + lab_headers

    def process_lung_function(self):
        lab_f = util.select_file(self.files, 'longfunctie')

        # gather renal function data
        rows, fields = util.import_data(lab_f, delim=self.delim)
        lab_headers, self.num_lab, self.num_lab_pos, suf = self.insert_data(
									rows, fields,
									'dmemo',
									['dtestdate', 'dtestdate'],
									'.+', None,
									suffix= ['lung_function', 'lab_results'])

        self.headers = self.headers + lab_headers