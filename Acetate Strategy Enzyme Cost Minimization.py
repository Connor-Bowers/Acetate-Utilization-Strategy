#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:17:44 2025

@author: Connor Bowers

Perform enzyme cost minimization for reverse beta-oxidation producing butyrate,
hexanoate and octanoate with complete acetate assimilation or recycling.
"""

from equilibrator_api import ComponentContribution, Q_
cc = ComponentContribution()
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Acetate_Strategy_ECM_Functions import (reactions_from_excel, 
                                            init_thermo_model, 
                                            MDF_lite, 
                                            RBO_ECM,
                                            ECM_energies, 
                                            MDF_plot_conc)

# TODO: Directory for reading excel model files and saving images

ECM_read = ''

ECM_save = ''

colors = ["xkcd:grey", "xkcd:red", "xkcd:blue"]

np.set_printoptions(legacy='1.25')

# Simulation Configuration

config_dict = {'flux_unit': 'Unitless',
               'version': '3',
               'kcat_source': 'fwd',
               'denominator': 'CM',
               'regularization': 'volume', 
               'objective': 'enzyme',
               'standard_concentration': '1 M', 
               'solver': 'CLARABEL', 
               'algorithm': 'ECM', 
               'p_h': '6',
               'ionic_strength': '250 mM',
               'p_mg': '3',
               'dg_confidence' : '0'}

# Standard Concentration Bounds

standard_lb = 0.001
standard_ub = 10

# Specify reduction potentials (V)

E_fe = -0.400 # Ferredoxin
E_rub = -0.075 # Rubredoxin
E_cyt = 0.254 # Cytochrome C
E_nadh = -0.280 # NADH
E_nadph = -0.370 # NADPH

lam_atp_dict = {} # Container for enzyme costs per ATP flux 
atp_dict = {} # Container for ATP yields on lactate

def wrap(title, 
         model_file, # Excel file path
         pmf, # Proton motive force in Volts
         ATP, # Mole ATP per mole lactate
         ):
    
    """ Wraps model construction from excel file, thermodynamic model initialization,
    MDF and ECM. Creates MDF graphs, plots free energies of enzyme cost minimum,
    plots enzyme costs, saves thermodynamic model and ECM solution as tsv,
    return enzyme cost per unit ATP flux (g/mmol ATP/hr) and enzyme cost relative
    to a standard """
    
    compounds = pd.read_excel(model_file, sheet_name = 'Compounds', index_col = 0)
    reactions = pd.read_excel(model_file, sheet_name = 'Reactions')
        
    reaction_dict, redox_list, proton_list = reactions_from_excel(reactions, compounds, cc)
    compound_dict = {name : cc.get_compound(compounds['ID'].loc[name]) for name in compounds.index}
    lower_bounds = compounds['Lower Bound'].fillna(standard_lb).apply(lambda x: Q_(x, 'mM'))
    upper_bounds = compounds['Upper Bound'].fillna(standard_ub).apply(lambda x: Q_(x, 'mM'))
    fluxes = reactions['Flux'].to_list()

    thermo_model = init_thermo_model(reaction_dict, 
                                     compound_dict, 
                                     redox_list,
                                     proton_list,
                                     fluxes,
                                     lower_bounds,
                                     upper_bounds,
                                     E_fe,
                                     E_rub,
                                     E_cyt,
                                     E_nadh,
                                     E_nadph,
                                     pmf,
                                     config_dict,
                                     cc
                                     )

    mdf_sol = MDF_lite(thermo_model)
    
    MDF_plot_conc(mdf_sol, title, ECM_save)

    Km_units = 'mM'
    crc_units = '1/s'
    protein_mass_units = 'kDa'
    compound_mass_units = 'Da'
    crc_array = reactions['Forward Catalytic Constant'].to_list()
    protein_mass_array = reactions['Protein Mass'].to_list()
    compound_mass_array = compounds['Compound Mass'].to_list()
    Km_df = pd.read_excel(model_file, sheet_name = 'Michaelis Constants')
    new_dgs = mdf_sol.standard_dg_prime

    protein_cost, protein_cost_per_ATP, ecm_sol = RBO_ECM(thermo_model,
                                                          new_dgs,
                                                          Km_df,
                                                          Km_units,
                                                          crc_array,
                                                          crc_units,
                                                          protein_mass_array,
                                                          protein_mass_units,
                                                          compound_mass_array,
                                                          compound_mass_units,
                                                          ATP,
                                                          title+' ECM', 
                                                          ECM_save, 
                                                          colors
                                                          )
    
    ecm_sol = ECM_energies(ecm_sol, thermo_model, title+' ∆Gs', ECM_save, 0.8)
    
    return protein_cost, protein_cost_per_ATP 

### Without RNF and HYD1

### Lactate to Butyrate IMF

title = 'Butyrate (Acetate Assimilation)'
model_file = ECM_read+'Butyrate Assimilation.xlsx'

pmf = 0.100 # Chemical potential of proton crossing membrane (in to out) in Volts
ATP = 0.5 # ATP yield per lactate

protein_cost, protein_cost_per_ATP = wrap(title, model_file, pmf, ATP)

atp_dict['Butyrate IMF'] = ATP
lam_atp_dict['Butyrate IMF'] = protein_cost_per_ATP
### Lactate to Butyrate SLP

title = 'Butyrate (Acetate Recycling)'
model_file = ECM_read+'Butyrate Recycling.xlsx'

pmf = 0.200 # Chemical potential of proton crossing membrane (in to out) in Volts
ATP = 0.25 # ATP yield per lactate

protein_cost, protein_cost_per_ATP = wrap(title, model_file, pmf, ATP)

atp_dict['Butyrate SLP'] = ATP
lam_atp_dict['Butyrate SLP'] = protein_cost_per_ATP

### Lactate to Caproate IMF

title = 'Hexanoate (Acetate Assimilation)'
model_file = ECM_read+'Hexanoate Assimilation.xlsx'

pmf = 0.100 # Chemical potential of proton crossing membrane (in to out) in Volts
ATP = 0.5 # ATP yield per lactate

protein_cost, protein_cost_per_ATP = wrap(title, model_file, pmf, ATP)

atp_dict['Hexanoate IMF'] = ATP
lam_atp_dict['Hexanoate IMF'] = protein_cost_per_ATP

### Lactate to Caproate SLP

title = 'Hexanoate (Acetate Recycling)'
model_file = ECM_read+'Hexanoate Recycling.xlsx'

pmf = 0.100 # Chemical potential of proton crossing membrane (in to out) in Volts
ATP = 0.33 # ATP yield per lactate

protein_cost, protein_cost_per_ATP = wrap(title, model_file, pmf, ATP)

atp_dict['Hexanoate SLP'] = ATP
lam_atp_dict['Hexanoate SLP'] = protein_cost_per_ATP

### Lactate to Octanoate IMF

title = 'Octanoate (Acetate Assimilation)'
model_file = ECM_read+'Octanoate Assimilation.xlsx'

# Specify chemical potential of proton crossing membrane (in to out)

pmf = 0.100 # Chemical potential of proton crossing membrane (in to out) in Volts
ATP = 0.5 # ATP yield per lactate

protein_cost, protein_cost_per_ATP = wrap(title, model_file, pmf, ATP)

atp_dict['Octanoate IMF'] = ATP
lam_atp_dict['Octanoate IMF'] = protein_cost_per_ATP

### Lactate to Octanoate SLP

title = 'Octanoate (Acetate Recycling)'
model_file = ECM_read+'Octanoate Recycling.xlsx'

# Specify chemical potential of proton crossing membrane (in to out)

pmf = 0.100 # Chemical potential of proton crossing membrane (in to out) in Volts
ATP = 0.375 # ATP yield per lactate

protein_cost, protein_cost_per_ATP = wrap(title, model_file, pmf, ATP)

atp_dict['Octanoate SLP'] = ATP
lam_atp_dict['Octanoate SLP'] = protein_cost_per_ATP

    
