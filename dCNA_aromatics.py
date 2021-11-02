import numpy as np
import networkx as nx
import MDAnalysis as mda
import MDAnalysis.analysis.distances as distances
import csv
import time

start = time.time()  

structure_file = 'testing/nm_wt_10chains_ph7_wb_ionized.psf'
dcd_trajectory = 'testing/nm_wt_test.dcd'


u = mda.Universe(structure_file,dcd_trajectory)

aro_resnames = ['PHE','TYR','TRP']

selection = 'resname ' + ' '.join(aro_resnames)
aro_atomgroup = u.select_atoms(selection).residues

resids = aro_atomgroup.resids
segids = aro_atomgroup.segids
resnames = aro_atomgroup.resnames
atomnames = aro_atomgroup.names

print(len(aro_atomgroup))
#print((resnames[1],resids[1],segids[1]))

mean_dist_array = np.zeros([len(aro_atomgroup),len(aro_atomgroup)])
std_dist_array = np.zeros([len(aro_atomgroup),len(aro_atomgroup)])
for i in range(len(aro_atomgroup)):
	loop_start = time.time()
	for j in range(i+1,len(aro_atomgroup)):

		sel1 = 'resname %s and resid %s and segid %s and not name N C O H* CA CB' % (resnames[i],resids[i],segids[i])
		res1 = u.select_atoms(sel1)
		pos1 = np.array([res1.atoms.positions for ts in u.trajectory])
		#pos1 = res1.atoms.positions
		#print(np.shape(pos1))

		sel2 = 'resname %s and resid %s and segid %s and not name N C O H* CA CB' % (resnames[j],resids[j],segids[j])
		res2 = u.select_atoms(sel2)
		pos2 = np.array([res2.atoms.positions for ts in u.trajectory])
		#pos2 = res2.atoms.positions
		#print(np.shape(pos2))

		d = np.array([np.min(distances.distance_array(pos1[k,:,:],pos2[k,:,:])) for k in range(len(u.trajectory))])
		#d = map(lambda k: np.min(distances.distance_array(pos1[k,:,:],pos2[k,:,:])),range(len(u.trajectory)))
		dlist = np.array(list(d))
		mean_dist_array[i,j] = np.mean(dlist)
		std_dist_array[i,j] = np.std(dlist)
		#print(d[0:10])

	#	min_dists = np.array([np.min(distances.distance_array(pos1,pos2)) for ts in u.trajectory])
	#	print(min_dists[100])

		#print(np.shape(pos1))

	line_time = time.time() - loop_start
	print('line %s: time = %.3d seconds' % (i,line_time))


#filename = 'dist_means.txt'
#with open(filename,'w') as f:
#        csv_writer = csv.writer(f, delimiter=' ')
#        csv_writer.writerows(mean_dist_array)
#
#filename = 'dist_means.txt'
#with open(filename,'w') as f:
#	csv_writer = csv.writer(f, delimiter=' ')
#	csv_writer.writerows(std_dist_array)

stop = time.time()
wallclock = stop - start
print('\nWallclock: %.3d seconds' % wallclock)
