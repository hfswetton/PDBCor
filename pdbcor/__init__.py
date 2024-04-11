import argparse
import json
import os
import shutil
from copy import copy

import matplotlib
import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser  # pdb extraction
from Bio.PDB.Polypeptide import is_aa
from matplotlib import pyplot as plt  # Plotting library
from matplotlib.cm import get_cmap
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, AutoMinorLocator
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from .estimators import DistanceCor, AngleCor

matplotlib.use('agg')


# correlation extraction wrapper class
class CorrelationExtraction:
    # constructor
    def __init__(self,
                 path,
                 mode='backbone',
                 nstates=2,
                 therm_fluct=0.5,
                 therm_iter=5,
                 loop_start=-1,
                 loop_end=-1):
        # HYPERVARIABLES
        directory = 'correlations'
        self.mode = mode
        self.savePath = os.path.join(os.path.dirname(path), directory)
        self.PDBfilename = os.path.basename(path)
        self.therm_iter = therm_iter
        self.resid = []
        self.aaS = 0
        self.aaF = 0
        self.nstates = nstates  # number of states
        # CREATE CORRELATION ESTIMATORS WITH STRUCTURE ANG CLUSTERING MODEL
        self.structure = PDBParser().get_structure('test', path)
        self.chains = []
        for chain in self.structure[0].get_chains():
            self.chains.append(chain.id)
        clust_model = GaussianMixture(n_components=self.nstates, n_init=25, covariance_type='diag')
        pose_estimator = GaussianMixture(n_init=25, covariance_type='full')
        self.distCor = DistanceCor(self.structure,
                                   mode,
                                   nstates,
                                   clust_model,
                                   pose_estimator,
                                   therm_fluct,
                                   loop_start,
                                   loop_end)
        self.angCor = AngleCor(self.structure, mode, nstates, clust_model)

    # calculate information gain between 2 clustering sets
    def calc_ami(self, clusters, banres):
        # calculate mutual information
        ami_list = []
        for i in tqdm(range(clusters.shape[0])):
            for j in range(i + 1, clusters.shape[0]):
                cor = adjusted_mutual_info_score(clusters[i, 1:], clusters[j, 1:])
                ami_list += list(clusters[i, :1]) + list(clusters[j, :1]) + [cor]
        ami_list = np.array(ami_list).reshape(-1, 3)
        # construct correlation matrix
        ami_matrix = np.zeros((int(self.aaF - self.aaS + 1), int(self.aaF - self.aaS + 1)))
        for i in range(self.aaS, int(self.aaF + 1)):
            if i not in self.resid or i in banres:
                for j in range(self.aaS, int(self.aaF + 1)):
                    ami_matrix[int(i - self.aaS), int(j - self.aaS)] = None
                    ami_matrix[int(j - self.aaS), int(i - self.aaS)] = None
            else:
                ami_matrix[int(i - self.aaS), int(i - self.aaS)] = 1
        for i in range(ami_list.shape[0]):
            ami_matrix[int(ami_list[i, 0] - self.aaS), int(ami_list[i, 1] - self.aaS)] = ami_list[i, 2]
            ami_matrix[int(ami_list[i, 1] - self.aaS), int(ami_list[i, 0] - self.aaS)] = ami_list[i, 2]
        return ami_list, ami_matrix

    def calc_cor(self, graphics=True):
    	# write readme file
        shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.txt'),
        		 os.path.join(self.savePath, 'README.txt'))
        for chain in self.chains:
            chainPath = os.path.join(self.savePath, 'chain' + chain)
            try:
                os.mkdir(chainPath)
            except:
                pass
            self.resid = []
            for res in self.structure[0][chain].get_residues():
                if is_aa(res, standard=True):
                    self.resid.append(res._id[1])
            self.aaS = min(self.resid)
            self.aaF = max(self.resid)
            print()
            print()
            print('################################################################################')
            print('#################################   CHAIN {}   #################################'.format(chain))
            print('################################################################################')
            self.calc_cor_chain(chain, chainPath, self.resid, graphics)

    # execute correlation extraction
    def calc_cor_chain(self, chain, chainPath, resid, graphics):
        # extract angle correlation matrices
        print()
        print()
        print('#############################   ANGLE CLUSTERING   ############################')
        ang_clusters, ang_banres = self.angCor.clust_cor(chain, resid)
        print('ANGULAR MUTUAL INFORMATION EXTRACTION:')
        ang_ami, ang_hm = self.calc_ami(ang_clusters, ang_banres)
        # Run a series of thermally corrected distance correlation extractions
        print()
        print()
        print('############################   DISTANCE CLUSTERING   ##########################')
        for i in range(self.therm_iter):
            dist_clusters, dist_banres = self.distCor.clust_cor(chain, resid)
            print('DISTANCE MUTUAL INFORMATION EXTRACTION, RUN {}:'.format(i + 1))
            dist_ami_loc, dist_hm_loc = self.calc_ami(dist_clusters, dist_banres)
            if i == 0:
                dist_ami = dist_ami_loc
                dist_hm = dist_hm_loc
            else:
                dist_ami = np.dstack((dist_ami, dist_ami_loc))
                dist_hm = np.dstack((dist_hm, dist_hm_loc))
        # Average them
        if self.therm_iter > 1:
            dist_ami = np.mean(dist_ami, axis=2)
            dist_hm = np.mean(dist_hm, axis=2)
        # round correlations down to 3 significant numbers
        dist_ami[:, 2] = np.around(dist_ami[:, 2], 4)
        ang_ami[:, 2] = np.around(ang_ami[:, 2], 4)
        # Calculate best coloring vector
        ami_sum = np.nansum(dist_hm, axis=1)
        best_res = [i for i in range(len(ami_sum)) if ami_sum[i] == np.nanmax(ami_sum)]
        best_res = best_res[0]
        best_clust = dist_clusters[best_res, 1:]
        print()
        print()
        print('############################       FINALIZING       ###########################')
        print('PROCESSING CORRELATION MATRICES')
        df = pd.DataFrame(ang_ami, columns=['ID1',
                                            'ID2',
                                            'Correlation'])
        df['ID1'] = df['ID1'].astype('int')
        df['ID2'] = df['ID2'].astype('int')
        df.to_csv(chainPath + '/ang_ami_' + self.mode + '.csv',
                  index=False)
        df = pd.DataFrame(dist_ami, columns=['ID1',
                                             'ID2',
                                             'Correlation'])
        df['ID1'] = df['ID1'].astype('int')
        df['ID2'] = df['ID2'].astype('int')
        df.to_csv(chainPath + '/dist_ami_' + self.mode + '.csv',
                  index=False)
        # write correlation parameters
        self.write_correlations(dist_ami, ang_ami, best_clust, chainPath)
        # create a chimera executable
        self.color_pdb(best_clust, chainPath)
        # plot everything
        if graphics:
            print('PLOTTING')
            self.plot_heatmaps(dist_hm, os.path.join(chainPath, 'heatmap_dist_' + self.mode + '.png'))
            self.plot_heatmaps(ang_hm, os.path.join(chainPath, 'heatmap_ang_' + self.mode + '.png'))
            self.plot_hist(dist_ami, os.path.join(chainPath, 'hist_dist_' + self.mode + '.png'))
            self.plot_hist(ang_ami, os.path.join(chainPath, 'hist_ang_' + self.mode + '.png'))
            self.custom_bar_plot(dist_hm, os.path.join(chainPath, 'seq_dist_' + self.mode + '.png'))
            self.custom_bar_plot(ang_hm, os.path.join(chainPath, 'seq_ang_' + self.mode + '.png'))
            shutil.make_archive(self.savePath, 'zip', self.savePath + '/')
        print('DONE')
        print()

    # write a file with correlation parameters
    def write_correlations(self, dist_ami, ang_ami, best_clust, path):
        # correlation parameters
        dist_cor = (np.mean(dist_ami[:, 2]))
        ang_cor = (np.mean(ang_ami[:, 2]))
        with open(os.path.join(path, 'correlations_' + self.mode + '.txt'), "w") as f:
            f.write('Distance correlations: {}\nAngle correlations: {} \n'.format(dist_cor, ang_cor))
            for i in range(self.nstates):
                pop = len([j for j in best_clust if j == i]) / len(best_clust)
                f.write('State {} population: {} \n'.format(i + 1, pop))
        result_dist = {
            'dist_cor': np.around(dist_cor, 4),
            'ang_cor': np.around(ang_cor, 4)
        }
        with open(os.path.join(path, 'results.json'), 'w') as outfile:
            json.dump(result_dist, outfile)

    # construct a chimera executive to view a colored bundle
    def color_pdb(self, best_clust, path):
        state_color = ['#00FFFF', '#00008b', '#FF00FF', '#FFFF00', '#000000']
        chimera_path = os.path.join(path, 'bundle_vis_' + self.mode + '.py')
        with open(chimera_path, 'w') as f:
            f.write('from chimera import runCommand as rc\n')
            f.write('rc("open ../../{}")\n'.format(self.PDBfilename))
            f.write('rc("background solid white")\n')
            f.write('rc("~ribbon")\n')
            f.write('rc("show :.a@ca")\n')
            # bundle coloring
            for i in range(len(self.structure)):
                f.write('rc("color {} #0.{}")\n'.format(state_color[int(best_clust[i])], int(i + 1)))

    # plot the correlation matrix heatmap
    def plot_heatmaps(self, hm, path):
        # change color map to display nans as gray
        cmap = copy(get_cmap("viridis"))
        cmap.set_bad('gray')
        # plot distance heatmap
        fig, ax = plt.subplots()
        pos = ax.imshow(hm,
                        origin='lower',
                        extent=[self.aaS, self.aaF, self.aaS, self.aaF],
                        cmap=cmap,
                        vmin=0,
                        vmax=1)
        fig.colorbar(pos, ax=ax)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        plt.xlabel('Residue ID')
        plt.ylabel('Residue ID')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    # plot histogram of correlation parameter
    def plot_hist(self, ami, path):
        # plot distance correlation histogram
        start = np.floor(min(ami[:, 2]) * 50) / 50
        nbreaks = int((1 - start) / 0.02) + 1
        fig, ax = plt.subplots()
        ax.hist(ami[:, 2], bins=np.linspace(start, 1, nbreaks), density=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Density')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    # create subplot
    def custom_bar_subplot(self, cor_seq, ax, ind):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        range_loc = range(self.aaS + 50 * ind, min(self.aaF + 1, self.aaS + 50 * (ind + 1)))
        cor_seq_loc = cor_seq[(50 * ind):min(len(cor_seq), 50 * (ind + 1))]
        if min(cor_seq) < 0:
            cor_range = max(cor_seq) - min(cor_seq)
            ax.set_ylim(min(cor_seq) - cor_range / 8, max(cor_seq) + cor_range / 8)
        else:
            ax.set_ylim(0, max(cor_seq))
            cor_range = max(cor_seq)
        ax.bar(range_loc, cor_seq_loc, width=0.8)
        for ind2, cor in enumerate(cor_seq_loc):
            if cor >= 0:
                ax.text(range_loc[ind2] - 0.25, cor + cor_range / 50, '{:.3f}'.format(cor), fontsize=10, rotation=90)
            else:
                ax.text(range_loc[ind2] - 0.25, cor - cor_range / 11, '{:.3f}'.format(cor), fontsize=10, rotation=90)
        if len(cor_seq) < 50:
            ax.set_xlim(self.aaS + 50 * ind - 1, self.aaF + 1)
        else:
            ax.set_xlim(self.aaS + 50 * ind - 1, self.aaS + 50 * (ind + 1))
        ax.set_xticks(range_loc)
        ax.set_xticklabels([i for i in range_loc], fontsize=10, rotation=90)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel('Correlation')
        if ind == np.ceil(len(cor_seq) / 50) - 1:
            ax.set_xlabel('Residue ID')

    # custom bar plot
    def custom_bar_plot(self, hm, path):
        cor_seq = np.mean(np.nan_to_num(hm), axis=0)
        fig, axs = plt.subplots(nrows=int(np.ceil(len(cor_seq) / 50)),
                                ncols=1,
                                figsize=(16, 4 * int(np.ceil(len(cor_seq) / 50))))
        if len(cor_seq) < 50:
            self.custom_bar_subplot(cor_seq, axs, 0)
        else:
            for ind, ax in enumerate(axs):
                self.custom_bar_subplot(cor_seq, ax, ind)
        plt.subplots_adjust(left=0.125,
                            bottom=-0.8,
                            right=0.9,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.2)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


def cli():
    """CLI script for __main__"""
    parser = argparse.ArgumentParser(description='Correlation extraction from multistate protein bundles')
    parser.add_argument('bundle', type=str,
                        help='protein bundle file path')
    parser.add_argument('--nstates', type=int,
                        default=2,
                        help='number of states')
    parser.add_argument('--graphics', type=bool,
                        default=True,
                        help='generate graphical output')
    parser.add_argument('--mode', type=str,
                        default='backbone',
                        help='correlation mode')
    parser.add_argument('--therm_fluct', type=float,
                        default=0.5,
                        help='Thermal fluctuation of distances in the protein bundle')
    parser.add_argument('--therm_iter', type=int,
                        default=5,
                        help='Number of thermal simulations')
    parser.add_argument('--loop_start', type=int,
                        default=-1,
                        help='Start of the loop')
    parser.add_argument('--loop_end', type=int,
                        default=-1,
                        help='End of the loop')
    args = parser.parse_args()
    # create correlations folder
    corPath = os.path.join(os.path.dirname(args.bundle), 'correlations')
    try:
        os.mkdir(corPath)
    except:
        pass
    # write parameters of the correlation extraction
    args_dict = vars(args)
    args_path = os.path.join(corPath, 'args.json')
    with open(args_path, 'w') as outfile:
        json.dump(args_dict, outfile)
    # correlation mode
    if args.mode == 'backbone':
        modes = ['backbone']
    elif args.mode == 'sidechain':
        modes = ['sidechain']
    elif args.mode == 'combined':
        modes = ['combined']
    elif args.mode == 'full':
        modes = ['backbone', 'sidechain', 'combined']
    else:
        parser.error('Mode has to be either backbone, sidechain, combined or full')
    for mode in modes:
        print('###############################################################################')
        print('############################   {} CORRELATIONS   ########################'.format(mode.upper()))
        print('###############################################################################')
        print()
        a = CorrelationExtraction(args.bundle,
                                  mode=mode,
                                  nstates=args.nstates,
                                  therm_fluct=args.therm_fluct,
                                  therm_iter=args.therm_iter,
                                  loop_start=args.loop_start,
                                  loop_end=args.loop_end)
        a.calc_cor(graphics=args.graphics)

