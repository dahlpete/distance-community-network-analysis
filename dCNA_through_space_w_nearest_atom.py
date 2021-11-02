"""
-------------------------------------------------------------------------------------------------------

DISTANCE COMMUNITY NETWORK ANALYSIS (dCNA) -- Electron Transfer Through Space

This script runs a distance analysis on aromatic amino acids from a molecular dynamics trajectory. The
script takes as an input, a .pdb file containing only aromatic residues (PHE, TYR, TRP, HIS). The 
geometric centers of the residue side chains is computed, and a matrix (Dist_matrix), with entries d_ij
equal to sum(1) or np.exp(-beta*r), is determined (for distinction, see first comment). In the case of the 
exponential, these entries are proportional to the probability of electron transfer, as determined by a 
tunnelling mechanism. The current version of the code is written for Python 2.7.

Code written by:
		Peter Dahl (1)(2)
		Malvankar Lab (1)(2)
		Batista Lab (3)

		(1) Yale University, Department of Molecular Biophysics and Biochemistry, New Haven, CT 06511
		(2) Yale University, Microbial Sciences Institute, West Haven, CT 06516
		(3) Yale University, Department of Chemistry, New Haven, CT 06511
-------------------------------------------------------------------------------------------------------		
"""

import numpy as np
import csv

threshold_mode = True      # This will determine how the distance matrix is built. If threshold mode is on
tunneling_mode = False     # the distance matrix elements will be a sum of 1's resulting from every instance
                           # the distance between a pair of residues is less than or equal to the threshold
                           # value (thresh). If tunnelling mode is on, the distance matrix elements will be
                           # the mean of exponentials of the form exp(-beta*r), where beta determines the
                           # rate at which the probability of tunnelling decays, and r is the distance
                           # between a pair of residues.

thresh_low = 0.0           # These thresholds are used in threshold mode as the criteria for building the
thresh_high = 10.0         # frequency matrix

ex_thresh = 14             # This threshold is used to compute the exclusion list
beta = 0.63                # This parameter is used in the exponential function to build the tunneling matrix

hist_thresh = 10           # This threshold is used as a cutoff when adding distances to the file
                           # dist_list_for_hist.txt

#read PDB file
aro_pdb = open('/gpfs/loomis/project/fas/batista/pd455/pathogen_work/wt/nm_wt_100ns.pdb')

"""
FUNCTIONS FOR DISTANCE CALCULATION
"""

def distance(coord_i,coord_j):
	"""
	Purpose: This function computes the distance between two residues. There is a special case when either
			 one or both the the residues in the calculation are a TRP residue. In this case, the fucntion
			 computes the distance between all aromatic rings (TRP has two) and selects the shortest.

	Inputs:
		coord_i         A list containing the coordinates of the center of the aromatic ring of the i-th 
						residue.
		coord_j         A list containing the coordinates of the center of the aromatic ring of the j-th 
						residue.
		iTRP			A boolean indicating if the i-th residue is a TRP.
		jTRP			A boolean indicating if the j-th residue is a TRP.

	Outputs:
		dist            The distance between the centers of the two aromatic rings.
	"""
	coord_i = coord_i[0]; coord_j = coord_j[0]
	i_length = len(coord_i); j_length = len(coord_j)

	tabulate_distances = np.zeros([i_length,j_length])
	for i in range(0,i_length):
		for j in range(0,j_length):
			tabulate_distances[i,j] = np.sqrt((coord_j[j][0] - coord_i[i][0])**2 + (coord_j[j][1] - coord_i[i][1])**2 + (coord_j[j][2] - coord_i[i][2])**2) 

	dist = min(min(tabulate_distances))

	return dist


def atom_positions(position,res_identity,type_of_atom,residue):
	"""
	Purpose: This function determines the coordinates of the geometric centers of each of the 
			 aromatic rings. TRP is a special case for which two sets of coordinates are calculated 
			 (one for each of the rings of the residue).

	Inputs:       
		position          A numpy array of the coordinates of each atom
		res_identity      An array of integer values indicating the identity of each aromatic residue. 
						  (e.g. PHE 1 chain A = 1; TYR 24 chain A = 2; TYR 27 chain A = 3; ...)
		type_of_atom      An array of values 1 through 6 that are used to assign atoms to particular 
						  positions in the aromatic ring. These are used to compute midpoints to identify 
						  the geometric centers of the rings.
		residue  		  A list of residue names (e.g. PHE, TYR, TRP, HIS). This function uses these to 
						  determine by which means it should calculate the center.

	Outputs: 
		residue           A rewritten list of residue names. Now there is only one instance of the residue 
						  type per actual residue in the stucture (before there was one instance per atom 
						  in the residue).
		center_pos        A list of coordinates of the centers of the aromatic rings. Since TRP has two 
						  aromatics rings, two sets of coordinates are included for the positions in the 
						  list cooresponding to a TRP residue.
	"""
	residue = residue[type_of_atom == 1,:]; 
	atomic_positions = np.zeros((int(max(res_identity)),1)); atomic_positions = atomic_positions.tolist()
	for i in range(0, int(max(res_identity))):
		res_indices = res_identity == i+1
		res_pos = position[res_indices,:]
		atomic_type = type_of_atom[res_indices]
		if residue[i,0] == 'PHE': 
			atom_pos_1 = res_pos[atomic_type == 1,:]
			atom_pos_2 = res_pos[atomic_type == 2,:]
			atom_pos_3 = res_pos[atomic_type == 3,:]
			atom_pos_4 = res_pos[atomic_type == 4,:]
			atom_pos_5 = res_pos[atomic_type == 5,:]
			atom_pos_6 = res_pos[atomic_type == 6,:]
			atom_pos_7 = res_pos[atomic_type == 7,:]
			
			atomic_positions[i] = [atom_pos_1,atom_pos_2,atom_pos_3,atom_pos_4,atom_pos_5,atom_pos_6,atom_pos_7]
		elif residue[i,0] == 'TYR':
			atom_pos_1 = res_pos[atomic_type == 1,:]
			atom_pos_2 = res_pos[atomic_type == 2,:]
			atom_pos_3 = res_pos[atomic_type == 3,:]
			atom_pos_4 = res_pos[atomic_type == 4,:]
			atom_pos_5 = res_pos[atomic_type == 5,:]
			atom_pos_6 = res_pos[atomic_type == 6,:]
			atom_pos_7 = res_pos[atomic_type == 7,:]
			atom_pos_8 = res_pos[atomic_type == 8,:]

			atomic_positions[i] = [atom_pos_1,atom_pos_2,atom_pos_3,atom_pos_4,atom_pos_5,atom_pos_6,atom_pos_7,atom_pos_8]

		elif residue[i,0] == 'TRP':
			atom_pos_1 = res_pos[atomic_type == 1,:]
			atom_pos_2 = res_pos[atomic_type == 2,:]
			atom_pos_3 = res_pos[atomic_type == 3,:]
			atom_pos_4 = res_pos[atomic_type == 4,:]
			atom_pos_5 = res_pos[atomic_type == 5,:]
			atom_pos_6 = res_pos[atomic_type == 6,:]
			atom_pos_7 = res_pos[atomic_type == 7,:]
			atom_pos_8 = res_pos[atomic_type == 8,:]
			atom_pos_9 = res_pos[atomic_type == 9,:]
			atom_pos_10 = res_pos[atomic_type == 10,:]
			
			atomic_positions[i] = [atom_pos_1,atom_pos_2,atom_pos_3,atom_pos_4,atom_pos_5,atom_pos_6,atom_pos_7,atom_pos_8,atom_pos_9,atom_pos_10]
#		elif residue[i,0][0] == 'HIS':
#			atom_pos_1 = res_pos[atomic_type == 1,:]
#			atom_pos_2 = res_pos[atomic_type == 2,:]
#			atom_pos_3 = res_pos[atomic_type == 3,:]
#			atom_pos_4 = res_pos[atomic_type == 4,:]
#			atom_pos_5 = res_pos[atomic_type == 5,:]
#			atom_pos_6 = res_pos[atomic_type == 6,:]
#
#			atomic_positions[i] = [atom_pos_1,atom_pos_2,atom_pos_3,atom_pos_4,atom_pos_5,atom_pos_6]

	return residue, atomic_positions

def distance_matrix(Dist_matrix,lt_thresh,position,res_identity,type_of_atom,residue,frame_num,thresh_low,thresh_high,ex_thresh,beta,tunneling_mode,threshold_mode,dist_list_for_hist,dist_sum,dist_sum_sqr):
	"""
	Purpose: This functions computes the matrix that describes the distance dependent interactions
			 between each aromatic residue with every other aromatic residue for a single time point.
			 However, it takes as an input the previous output of the function and adds on the matrix
			 computed in the curent step. This program computes a proxy for the tunnelling probability
			 given by np.exp(-beta*r) where r is the distance between the centers of the aromatics.

	Inputs:
		Dist_matrix          A matrix that describes the distance dependent relationships of each 
							 aromatic residue with every other aromatic residue. Each entry D(i,j)
							 is a sum of terms determined via the exponential relation np.exp(-beta*r),
							 a relation that is proportional to the tunnelling probability. The number
							 of terms summed is equal to the number of time points in the trajectory
							 file. Initially input as a numpy array of a single zero.
		lt_thresh            Initially a numpy array of a single zero, but is initialzed to a numpy 
							 array of zeros of dimension equal to the number of residues when 
							 frame_num is equal to 2. 
		position 			 A numpy array of the coordinates of each atom
		res_identity         An array of integer values indicating the identity of each aromatic residue. 
						     (e.g. PHE 1 chain A = 1; TYR 24 chain A = 2; TYR 27 chain A = 3; ...)
		type_of_atom         An array of values 1 through 6 that are used to assign atoms to particular 
						     positions in the aromatic ring. These are used to compute midpoints to identify 
						     the geometric centers of the rings.
		residue              A list of residue names (e.g. PHE, TYR, TRP, HIS).
		frame_num			 The time point being analyzed in frame numbers (1 frame per 5 ps).
		thresh               The distance threshold used to determine the exclusion list (in angstroms)
		beta                 The decay factor in the exponential used to determine D(i,j). Beta is a value
							 characteristic of the material being studied.

	Outputs:
		Dist_matrix          A matrix that describes the distance dependent relationships of each 
							 aromatic residue with every other aromatic residue. Each entry D(i,j)
							 is a sum of terms dependent on the specified functional form (see first 
							 comment). The number of terms summed is equal to the number of time 
							 points in the trajectory file.
		residue              Residue remains unchanged; thus, it remains the same as the input of the 
							 same name
		lt_thresh            Each entry (i,j) of this array indicates whether the given residue pair
							 has ever been within the defined distance threshold (ex_thresh)
	"""
	[residue, atomic_positions] = atom_positions(position,res_identity,type_of_atom,residue)
	if frame_num == 2:
		Dist_matrix = np.zeros((len(atomic_positions),len(atomic_positions)))
		dist_sum = np.zeros((len(atomic_positions),len(atomic_positions)))
		dist_sum_sqr = np.zeros((len(atomic_positions),len(atomic_positions)))
		lt_thresh = np.zeros((len(atomic_positions),len(atomic_positions)))
	
	for i in range(len(atomic_positions)):
		for j in range(len(atomic_positions)):
			if residue[i,0] == 'TRP':
				iTRP = True
			else:
				iTRP = False

			if residue[j,0] == 'TRP':
				jTRP = True
			else:
				jTRP = False

			if (residue[i,0] == 'PHE' or residue[i,0] == 'TYR' or residue[i,0] == 'TRP'): # or residue[i,0] == 'HIS'):
				r = distance(atomic_positions[i],atomic_positions[j])
				if r <= hist_thresh and r > 0:
					dist_list_for_hist.append(r)

				if threshold_mode == True:
					if r >= thresh_low and r <= thresh_high:
						Dist_matrix[i,j] += 1

				elif tunneling_mode == True:
					Dist_matrix[i,j] += np.exp(-beta*r)

				dist_sum[i,j] += r
				dist_sum_sqr[i,j] += r**2

				if r < ex_thresh and lt_thresh[i,j] == 0:
					lt_thresh[i,j] = 1


	return Dist_matrix, residue, lt_thresh, dist_list_for_hist, dist_sum, dist_sum_sqr


"""
MAIN LOOP
"""
if tunneling_mode == True:
	print('Starting Tunneling Analysis... \n')
elif threshold_mode == True:
	print('Starting Distance Threshold Analysis... \n')

atom_type = 0
residue_identity = ['RESNAME', str(0), 'CHAIN']; res_id = 0
frame_num = 0; loop_count = 0
Dist_matrix = np.array([0]); lt_thresh = np.array([0])
dist_sum = np.array([0]); dist_sum_sqr = np.array([0])
dist_list_for_hist = []

#loop through rows of PDB file to generate arrays of required data
for line in aro_pdb:
	list = line.split()
	id = list[0]
	
	if id == 'ATOM' and (list[3] == 'PHE' or list[3] == 'TYR' or list[3] == 'TRP'): # or list[3] == 'HIS'):
		if list[1] == '1':
			res_id = 0
			loop_count = 0
			frame_num += 1	
			if frame_num >= 2:
				residue = residue[residue != ['X', 'X', 'X']]; 
				residue = np.reshape(residue,(len(residue)/3,3))
				type_of_atom = type_of_atom[np.isnan(type_of_atom) == False]
				res_identity = res_identity[np.isnan(res_identity) == False]
				position = position[np.isnan(position[:,0]) == False]

                   	        if frame_num % 100 == 0:
					print('Frame Number '+str(frame_num))  

				"""
				Call function for computation of the distance matrix
				"""
				[Dist_matrix, residue2, lt_thresh, dist_list_for_hist, dist_sum, dist_sum_sqr] = distance_matrix(Dist_matrix,lt_thresh,position,res_identity,type_of_atom,residue,frame_num,thresh_low,thresh_high,ex_thresh,beta,tunneling_mode,threshold_mode,dist_list_for_hist,dist_sum,dist_sum_sqr)

			# re-initialize the data arrays for the next frame
			residue = np.chararray((300000,3), itemsize=3); type_of_atom = np.empty((300000,1))
			res_identity = np.empty((300000,1)); position = np.empty((300000,3));
			residue[:,:] = 'X'; 
			type_of_atom[:] = np.nan; res_identity[:] = np.nan; position[:,:] = np.nan

		if [list[3], str(list[5]), list[4]] != residue_identity:
			res_id += 1
			residue_identity = [list[3], str(list[5]), list[4]]

		if residue_identity[0] == 'PHE':
			if atom_type < 7:
				atom_type += 1
			else:
				atom_type = 1

			type = list[2]
			if (type == 'CB'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 7
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]  
			if (type == 'CG'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 1
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]  
			elif (type == 'CD1'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 2
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CD2'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 6
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CE1'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 3
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CE2'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 5
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CZ'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 4
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			loop_count += 1

		elif residue_identity[0] == 'TYR':
			if atom_type < 8:
				atom_type += 1
			else:
				atom_type = 1

			type = list[2]
			if (type == 'CB'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 7
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]  
			if (type == 'CG'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 1
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]  
			elif (type == 'CD1'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 2
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CD2'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 6
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CE1'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 3
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CE2'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 5
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CZ'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 4
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			elif (type == 'OH'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 8
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			loop_count += 1

		elif residue_identity[0] == 'TRP':
			if atom_type < 10:
				atom_type += 1
			else:
				atom_type = 1

			type = list[2]
			if (type == 'CB'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 10
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]  
			if (type == 'CG'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 1
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			elif (type == 'CD1'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 2
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			elif (type == 'NE1'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 3
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			elif (type == 'CE2'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 4
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9] 
			elif (type == 'CD2'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 5
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			elif (type == 'CE3'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 6
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			elif (type == 'CZ3'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 7
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			elif (type == 'CZ2'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 9
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			elif (type == 'CH2'):
				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 8
				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
			loop_count += 1

#		elif residue_identity[0] == 'HIS':
#			if atom_type < 6:
#				atom_type += 1
#			else:
#				atom_type = 1
#
#			type = list[2]
#			if (type == 'CB'):
#				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 6
#				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]  
#			if (type == 'CG'):
#				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 1
#				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
#			elif (type == 'ND1'):
#				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 2
#				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
#			elif (type == 'CE1'):
#				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 3
#				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
#			elif (type == 'NE2'):
#				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 4
#				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
#			elif (type == 'CD2'):
#				residue[loop_count,:] = residue_identity; type_of_atom[loop_count] = 5
#				res_identity[loop_count] = res_id; position[loop_count,:] = list[6:9]
#			loop_count += 1

if tunneling_mode == True:
	Dist_matrix = Dist_matrix / frame_num #average value of np.exp(-r)

Dist_matrix_norm = (Dist_matrix - np.amin(Dist_matrix)) / (np.amax(Dist_matrix) - np.amin(Dist_matrix))
Dist_matrix_norm2 = (Dist_matrix - np.amin(Dist_matrix)) / (np.amax(Dist_matrix) - np.amin(Dist_matrix))
Dist_matrix_norm[Dist_matrix_norm < 1e-6] = 2.0

dist_mean = dist_sum / frame_num
dist_variance = dist_sum_sqr / frame_num - dist_mean**2
dist_std = np.sqrt(dist_variance)

residue = residue2[residue2 != ['X', 'X', 'X']]; 
residue = np.reshape(residue,(len(residue)/3,3))
type_of_atom = type_of_atom[np.isnan(type_of_atom) == False]
res_identity = res_identity[np.isnan(res_identity) == False]
position = position[np.isnan(position[:,0]) == False]

"""
CALCULATE EXCLUSION MATRIX
"""
length = np.shape(Dist_matrix)[0]
total_ix_array = np.arange(1,length+1)
exclusion = np.zeros((length,1)); exclusion = exclusion.tolist()
ex_ix = np.zeros(length); ex_ix = ex_ix.tolist()
for i in range(0,length):
	exclusion[i] = []
	rowlist = exclusion[i]
	for j in range(0,length):
		if (lt_thresh[i,j] == 0 or i == j or Dist_matrix_norm[i,j] == 2.0):                 # A residue j is added to the exclusion matrix if it  
			rowlist.append(j)                                                           # has never been within the distance threshold, if i 
	if len(rowlist) == length:                                                                  # is equal to j (same residue), or if the distance 
		ex_ix[i] = False                                                                    # matrix entry is equal to 2.0 (indicates value < 1e-6)                                                                        
	else:                                                                                   
		ex_ix[i] = True

ex_ix = np.array(ex_ix, dtype=bool)
residue_exclude = residue[ex_ix,:]
dist_mat_ex = Dist_matrix_norm[ex_ix,:]
dist_mat_ex = dist_mat_ex[:,ex_ix]
dist_mat_ex2 = Dist_matrix_norm2[ex_ix,:]
dist_mat_ex2 = dist_mat_ex2[:,ex_ix]
dist_mean_ex = dist_mean[ex_ix,:]
dist_mean_ex = dist_mean_ex[:,ex_ix]
dist_std_ex = dist_std[ex_ix,:]
dist_std_ex = dist_std_ex[:,ex_ix]
lt_thresh_ex = lt_thresh[ex_ix,:]
lt_thresh_ex = lt_thresh_ex[:,ex_ix]

# Now recalculate the exclusion list
length2 = np.shape(dist_mat_ex)[0]
exclusion2 = np.zeros((length2,1)); exclusion2 = exclusion2.tolist()
for i in range(0,length2):
	exclusion2[i] = []
	rowlist = exclusion2[i]
	for j in range(0,length2):
		if (lt_thresh_ex[i,j] == 0 or i == j or dist_mat_ex[i,j] == 2.0):
			rowlist.append(j)

"""
WRITE TEXT FILES
"""
print('\nWriting text files...\n')

filename = 'nottunneling.txt'
with open(filename,'w') as f:
	csv_writer = csv.writer(f, delimiter=' ')
	csv_writer.writerows(exclusion2)

filename = 'matrix.in'
dist = np.zeros((length2+1,1)); dist = dist.tolist()
for i in range(0,length2+1):
	dist[i] = []
	row = dist[i]
	if i == 0:
		row.append(length2)
	else:
		entry = dist_mat_ex[i-1,:]
		row.extend(np.float32(entry))


with open(filename,'w') as f:
	csv_writer = csv.writer(f, delimiter=' ')
	csv_writer.writerows(dist)


filename = 'dist_matrix_for_centrality.txt'
dist = np.zeros((length2+1,1)); dist = dist.tolist()
for i in range(0,length2+1):
	dist[i] = []
	row = dist[i]
	if i == 0:
		row.append(length2)
	else:
		entry = dist_mat_ex2[i-1,:]
		row.extend(np.float32(entry))


with open(filename,'w') as f:
	csv_writer = csv.writer(f, delimiter=' ')
	csv_writer.writerows(dist)

filename = 'dist_means.txt'
dist_means = np.zeros([length2,1]); dist_means = dist_means.tolist()
for i in range(0,length2):
	dist_means[i] = []
	row_mean = dist_means[i]
	entry_mean = dist_mean_ex[i,:]
	row_mean.extend(np.float32(entry_mean))

with open(filename,'w') as f:
	csv_writer = csv.writer(f, delimiter=' ')
	csv_writer.writerows(dist_means)


filename = 'dist_stds.txt'
dist_stds = np.zeros([length2,1]); dist_stds = dist_stds.tolist()
for i in range(0,length2):
	dist_stds[i] = []
	row_stds = dist_stds[i]
	entry_stds = dist_std_ex[i,:]
	row_stds.extend(np.float32(entry_stds))

with open(filename,'w') as f:
	csv_writer = csv.writer(f, delimiter=' ')
	csv_writer.writerows(dist_stds)

#residue id list_after exclusion
residue_numbs = np.arange(0,length2).reshape(length2,1)
residue_list = np.zeros((length2,1)); residue_list = residue_list.tolist()
resid_conv = np.zeros((length2,2))
for row in range(0,length2):
	residue_list[row] = [row,residue_exclude[row,0],residue_exclude[row,1],residue_exclude[row,2]]
	for col in range(0,2):
		if col == 0:
			entry2 = residue_numbs[row]
			resid_conv[row,col] = int(float(entry2))
		elif col == 1:
			entry2 = residue_exclude[row,1]
			resid_conv[row,col] = int(float(entry2))

filename = 'residue_list_ex.txt'
with open(filename,'w') as f:
	csv_writer = csv.writer(f, delimiter = '\t')
	csv_writer.writerows(residue_list)

filename = 'residue_list_for_conv.txt'
with open(filename,'w') as f:
	csv_writer = csv.writer(f,delimiter = ' ')
	csv_writer.writerows(resid_conv)

filename = 'dist_list_for_hist.txt'
with open(filename,'w') as f:
	csv_writer = csv.writer(f, delimiter = '\t')
	csv_writer.writerow(dist_list_for_hist)


"""
REFORMAT FOR GNUPLOT
"""

dim = ((length**2) + (length-1))
gnu_dist_matrix = np.empty((dim,3),dtype='str')
gnu_dist_matrix[:,:] = ' '; gnu_dist_matrix = gnu_dist_matrix.tolist()
index = -1; count = 0;
for j in range(0,length):
	index += 1
	for i in range(0,length):
		ix = count + index
		gnu_dist_matrix[ix][0] = str(i)
		gnu_dist_matrix[ix][1] = str(j)
		gnu_dist_matrix[ix][2] = str(Dist_matrix_norm[i,j])

		count += 1
		
gnu_dist_matrix = np.array(gnu_dist_matrix)
filename = 'tunneling_matrix_gnu_norm.txt'
with open(filename,'w') as f:
	csv_writer = csv.writer(f, delimiter='\t')
	csv_writer.writerows(gnu_dist_matrix)

print('Done')
