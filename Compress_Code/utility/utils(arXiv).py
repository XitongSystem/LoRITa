import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.linalg import svdvals
from scipy.interpolate import interp1d
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.ticker import MaxNLocator

def svd(A, full_matrices=True):
    U, s, VT = np.linalg.svd(A, full_matrices=full_matrices)
    return U, s, VT.T

def compose(f, g):
    def h(x):
        return f(g(x))
    return h

def compute_angle(U, V):
    if len(U.shape) < 2:
        U = U.reshape(-1, 1)
    if len(V.shape) < 2:
        V = V.reshape(-1, 1)

    s = np.clip(svdvals(U.T@V), 0, 1)
    theta = np.arccos(s)
    # remove the top theta
    # theta = theta[:-6]

    # return np.linalg.norm(theta)

    # if len(U.shape) < 2:
    #     U = U.reshape(-1, 1)
    # if len(V.shape) < 2:
    #     V = V.reshape(-1, 1)

    # s = np.clip(svdvals(np.dot(U.T, V)), 0, 1)
    # theta = np.arccos(s)

    return np.max(theta)

def compute_svd_series(weights, rank):
    input_dim = weights[0].shape[1]
    m = input_dim - 2 * rank


    Uinit, _, Vinit = svd(weights[1])

    right_series = [[ *(input_dim * [0]) ]]
    left_series = [[ *(input_dim * [0]) ]]
    sval_series = [svdvals(weights[0])/np.max(svdvals(weights[0]))]

    # for w in tqdm(weights[1:]):
    for w in  weights[1:]:
    # for w in tqdm(weights[-1:]):
        Ut, st, Vt = svd(w)

        #
        # print(st[:2*rank])

        st = st/np.max(st)
        print(st[:4*rank])
        sval_series.append(st)


        Ut_top = Ut[:, :rank]
        Ut_mid = Ut[:, rank:-rank]
        Ut_bot = Ut[:, -rank:]

        Uinit_top = Uinit[:, :rank]
        Uinit_mid = Uinit[:, rank:-rank]
        Uinit_bot = Uinit[:, -rank:]

        Vt_top = Vt[:, :rank]
        Vt_mid = Vt[:, rank:-rank]
        Vt_bot = Vt[:, -rank:]

        Vinit_top = Vinit[:, :rank]
        Vinit_mid = Vinit[:, rank:-rank]
        Vinit_bot = Vinit[:, -rank:]

        right = []

        # for k in range(rank):
        #     right.append(compute_angle(Vt_top[:, k], Vinit_top[:, k]))

        right += rank * [compute_angle(Vt_top, Vinit_top)]
        # right += rank * [compute_angle(np.concatenate((Vt_top, Vt_bot),axis=1), np.concatenate((Vinit_top, Vinit_bot),axis=1))]

        # for k in range(m):
        #     right.append(compute_angle(Vt_mid[:, k], Vinit_mid[:, k]))
        

        right += m * [compute_angle(Vt_mid, Vinit_mid)]
        # print(compute_angle(Vt_mid, Vinit_mid))

        # for k in range(rank):
            # right.append(compute_angle(Vt_bot[:, k], Vinit_bot[:, k]))

        right += rank * [compute_angle(Vt_bot, Vinit_bot)]

        right_series.append(right)

        left = []

        # for k in range(rank):
        #     left.append(compute_angle(Ut_top[:, k], Uinit_top[:, k]))

        left += rank * [compute_angle(Ut_top, Uinit_top)]
        
        # for k in range(m):
        #     left.append(compute_angle(Ut_mid[:, k], Uinit_mid[:, k]))

        left += m * [compute_angle(Ut_mid, Uinit_mid)]

        # for k in range(rank):
        #     left.append(compute_angle(Ut_bot[:, k], Uinit_bot[:, k]))

        left += rank * [compute_angle(Ut_bot, Uinit_bot)]

        left_series.append(left)

    sval_series = np.array(sval_series)
    right_series = np.array(right_series)
    left_series = np.array(left_series)
    
    return (sval_series, right_series, left_series)


def plot_sv_series(ax, series, color='viridis', spec_step=2, sample_step=1):
    # subsample the series 
    series = series[:, ::sample_step]

    n_time_indices, n_sval_indices = series.shape
    time_indices = np.arange(n_time_indices)
    sval_indices = np.arange(n_sval_indices)

    spectrum_verts = []

    for idx in time_indices[::spec_step]:
        spectrum_verts.append([
            (0, np.min(series)-0.05), *zip(sval_indices, series[idx, :]), (n_sval_indices, np.min(series)-0.05)
        ])

    path_verts = []

    for idx in sval_indices:
        path_verts.append([
            *zip(time_indices, series[:, idx])
        ])

    spectrum_poly = PolyCollection(spectrum_verts)
    spectrum_poly.set_alpha(0.8)
    spectrum_poly.set_facecolor(plt.colormaps[color](np.linspace(0, 0.7, len(spectrum_verts))))
    spectrum_poly.set_edgecolor('black')

    path_line = LineCollection(path_verts)
    path_line.set_linewidth(1)
    path_line.set_edgecolor('black')
    
    ax.set_box_aspect(aspect=None, zoom=0.85)

    ax.add_collection3d(spectrum_poly, zs=time_indices[::spec_step], zdir='y')
    ax.add_collection3d(path_line, zs=sval_indices, zdir='x')

    ax.set_xlim(0, n_sval_indices)
    ax.set_ylim(0, n_time_indices)
    ax.set_zlim(np.min(series)-0.1, np.max(series)+0.1)

    # elev = 30
    # azim = -50
    # roll = 0
    # ax.view_init(elev, azim, roll)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))