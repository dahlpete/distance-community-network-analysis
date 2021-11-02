import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.distances as distances
from multiprocessing import Pool, cpu_count
import sys
import time

structure_file = 'testing/nm_wt_10chains_ph7_wb_ionized.psf'
dcd_trajectory = 'testing/nm_wt_test.dcd'

ncores = int(sys.argv[1])

u = mda.Universe(structure_file,dcd_trajectory)

aro_resnames = ['PHE','TYR','TRP']

selection = 'resname ' + ' '.join(aro_resnames)
aro_atomgroup = u.select_atoms(selection).residues

resids = aro_atomgroup.resids
segids = aro_atomgroup.segids
resnames = aro_atomgroup.resnames
atomnames = aro_atomgroup.names

mean_dist_array = np.zeros([len(aro_atomgroup),len(aro_atomgroup)])
std_dist_array = np.zeros([len(aro_atomgroup),len(aro_atomgroup)])

def pair_distance(res_pair):

	l = res_pair[0]; m = res_pair[1]

	sel1 = 'resname %s and resid %s and segid %s and not name N C O H* CA CB' % (resnames[l],resids[l],segids[l])
	res1 = u.select_atoms(sel1)
	pos1 = np.array([res1.atoms.positions for ts in u.trajectory])

	sel2 = 'resname %s and resid %s and segid %s and not name N C O H* CA CB' % (resnames[m],resids[m],segids[m])
	res2 = u.select_atoms(sel2)
	pos2 = np.array([res2.atoms.positions for ts in u.trajectory])
	
	d = np.array([np.min(distances.distance_array(pos1[k,:,:],pos2[k,:,:])) for k in range(len(u.trajectory))])
	dlist = np.array(list(d))
	mean_dist = np.mean(dlist)
	std_dist = np.std(dlist)

	return (mean_dist,std_dist)


def main():
	start = time.time()
	print(f'starting computations on {ncores} cores')

	residue_pairs = [(i,j) for i in range(len(aro_atomgroup)) for j in range(i+1,len(aro_atomgroup))]

	with Pool(ncores) as pool:
		dist_vals = pool.map(pair_distance,residue_pairs)

	dist_mean_array = np.zeros([len(aro_atomgroup),len(aro_atomgroup)])
	dist_std_array = np.zeros([len(aro_atomgroup),len(aro_atomgroup)])
	for k in range(len(residue_pairs)):
		i,j = residue_pairs[k]

		dist_mean_array[i,j] = dist_vals[k][0]
		dist_mean_array[j,i] = dist_vals[k][0]

		dist_std_array[i,j] = dist_vals[k][1]
		dist_std_array[j,i] = dist_vals[k][1]

	stop = time.time()
	wallclock = stop - start
	print('\nWallclock: %.3f seconds' % wallclock)

if __name__ == '__main__':
	main()

