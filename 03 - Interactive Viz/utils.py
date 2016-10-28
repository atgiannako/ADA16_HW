# utils.py contains helper functions to be used in hw03.ipynb

import geocoder
import numpy as np

def extract_canton(university):
	'''
	Extracts the canton of a university
	NOTE: not all universities have the right canton ID
	
	Parameters
	----------
	@university : str
		university name
	
	Returns
	-------
		str: the canton of the (input param) university
	'''
	# keep canton of university
	university_canton = university.split(' - ')[-1]
	return university_canton


def find_canton_from_google(university):
	'''
	Extracts the canton in which a university is located.
	
	Parameters
	----------
	@university: str
		university name
	
	Returns
	-------
		str: the canton in which the  (input param) university is located or
			the str NaN if the university is not found
	'''
	g = geocoder.google(university)
	try:
		return g.geojson['properties']['state']
	except:
		return np.nan

def modify_university_name(university):
	'''
	Extracts university name.
	
	Parameters
	----------
	@university: str
		str in format '<uni_name> - <canton abriviation>'
	
	Returns
	-------
		str: <uniname> taken from the parameter @university
	'''
	return university.split(' - ')[0]

def fill_unknown_cantons(swiss_cantons, uni_to_cantons, university, canton):
	'''
	Gets a name of a university and tries to guess in which canton the university can be assigned
	
	Parameters
	----------
	@swiss_cantons: dictionary
		maps Swiss canton abbreviations to original Swiss canton names
		key -> canton abbreviation
		value -> original Swiss canton name
	@uni_to_cantons: dictionary
		maps Swiss universities to Swiss cantons
		key -> Swiss university name
		value -> Swiss canton abbreviation
	@university: str
		name of the university
	@canton: str
		canton abbreviation to be returned if the canton abbreviation of the university is in the @swiss_cantons dictionary
	
	Returns
	-------
		str: abbreviation of the canton that the @university belongs to. 
	'''
	if canton in swiss_cantons.keys():
		return canton
	try:
		return uni_to_cantons[university]
	except:
		return np.nan

def get_university_location(university):
	'''
	Extracts university name.
	NOTE: uses geocoder library.
	
	Parameters
	----------
	@university: str
		university name
	
	Returns
	-------
		list: list with two values [latitude, longitude] which represent
		the exact location of a Swiss university in a map
	'''
	g = geocoder.google(university)
	return g.latlng

def split_switzerland(canton, french_part, german_part):
	'''
	Assigns university to french, german, or other part of Switzerland.
	
	Parameters
	----------
	@canton: str
		canton abbreviation
	@french_part: list
		list of str with canton abbreviations that belong to the french part
	@german_part: list
		list of str with canton abbreviations that belong to the german part    
	
	Returns
	-------
		str: 'french', 'german', or 'other' according to canton id
	'''
	if canton in french_part:
		return 'french'
	elif canton in  german_part:
		return 'german'
	return 'other'