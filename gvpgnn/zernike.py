# https://github.com/nipy/mindboggle/blob/master/mindboggle/shapes/zernike/zernike.py

import numpy as np
from scipy.special import (factorial, comb as nchoosek)
import time
import numba

import logging
LOG = logging.getLogger(__name__)


def nested_loop(stack, args):
  if len(stack) != 0:
    fn = stack.pop()
    for i in fn(*args):
      for j in nested_loop(stack, args+[i]):
        yield (i,)+j
    stack.append(fn)
  else:
    yield tuple()


def nest(*_stack):
  return nested_loop(list(reversed(_stack)), [])


def autocat(arrs, **dargs):
  axis = dargs.pop('axis', None)
  if axis is None:
    return np.concatenate(arrs, **dargs)
  ndim = arrs[0].ndim
  assert all([ a.ndim == ndim for a in arrs])
  if axis >= ndim:
    arrs = tuple([ np.expand_dims(a,axis) for a in arrs ])
  return np.concatenate(arrs, axis=axis)


IMAG_CONST = np.imag(1j) # scipy.sqrt(-1)
PI_CONST = np.pi
NAN_CONST = np.NaN


def geometric_moments_exact(points_array, faces_array, N):
  n_facets, n_vertices = faces_array.shape[:2]
  assert n_vertices == 3
  moments_array = np.zeros([N + 1, N + 1, N + 1])
  monomial_array = monomial_precalc(points_array, N)
  for face in faces_array:
    vertex_list = [points_array[_i, ...] for _i in face]
    Cf_list = [monomial_array[_i, ...] for _i in face]
    Vf = facet_volume(vertex_list)
    t0 = time.time()
    moments_array += Vf * term_Sijk(Cf_list, N)
  return factorial_scalar(N) * moments_array


def factorial_scalar(N):
  i, j, k = np.mgrid[0:N + 1, 0:N + 1, 0:N + 1]
  return factorial(i) * factorial(j) * factorial(k) / (factorial(i + j + k + 2) * (i + j + k + 3))


def monomial_precalc(points_array, N):
  n_points = points_array.shape[0]
  monomial_array = np.zeros([n_points, N + 1, N + 1, N + 1])
  tri_array = trinomial_precalc(N)
  for point_indx, point in enumerate(points_array):
    monomial_array[point_indx, ...] = mon_comb(
      point, tri_array, N)
  return monomial_array


def mon_comb(vertex, tri_array, N, out=None):
  x, y, z = vertex
  c = np.zeros([N + 1, N + 1, N + 1])
  for i, j, k in nest(lambda: range(N + 1),
            lambda _i: range(N - _i + 1),
            lambda _i, _j: range(N - _i - _j + 1),
            ):
    c[i, j, k] = tri_array[i, j, k] * \
      np.power(x, i) * np.power(y, j) * np.power(z, k)
  return c


@numba.njit
def term_Sijk(Cf_list, N):
  S = np.zeros((N + 1, N + 1, N + 1))
  C0, C1, C2 = Cf_list
  Dabc = term_Dabc(C1, C2, N)

  for i in range(N + 1):
    for j in range(N - i + 1):
      for k in range(N - i - j + 1):
        for ii in range(i + 1):
          for jj in range(j + 1):
            for kk in range(k + 1):
              S[i, j, k] += C0[ii, jj, kk] * Dabc[i - ii, j - jj, k - kk]
  return S


def trinomial_precalc(N):
  tri_array = np.zeros([N + 1, N + 1, N + 1])
  for i, j, k in nest(lambda: range(N + 1),
            lambda _i: range(N - _i + 1),
            lambda _i, _j: range(N - _i - _j + 1)
            ):
    tri_array[i, j, k] = trinomial(i, j, k)
  return tri_array


def trinomial(i, j, k):
  return factorial(i + j + k) / (factorial(i) * factorial(j) * factorial(k))


def facet_volume(vertex_list):
  return np.linalg.det(autocat(vertex_list, axis=1))


@numba.njit
def term_Dabc(C1, C2, N: int):
  D = np.zeros((N + 1, N + 1, N + 1))
  for i in range(N + 1):
    for j in range(N + 1):
      for k in range(N + 1):
        for ii in range(i + 1):
          for jj in range(j + 1):
            for kk in range(k + 1):
              D[i, j, k] += C1[ii, jj, kk] * C2[i - ii, j - jj, k - kk]
  return D


def zernike(G, N):
  V = np.zeros((N + 1, N + 1, N + 1), dtype=complex)
  for a, b, c, alpha in nest(lambda: range(int(N / 2) + 1),
                  lambda _a: range(N - 2 * _a + 1),
                  lambda _a, _b: range(N - 2 * _a - _b + 1),
                  lambda _a, _b, _c: range(_a + _c + 1),
                  ):
    V[a, b, c] += np.power(IMAG_CONST, alpha) * \
      nchoosek(a + c, alpha) * G[2 * a + c - alpha, alpha, b]

  W = np.zeros((N + 1, N + 1, N + 1), dtype=complex)
  for a, b, c, alpha in nest(lambda: range(int(N / 2) + 1),
                  lambda _a: range(N - 2 * _a + 1),
                  lambda _a, _b: range(N - 2 * _a - _b + 1),
                  lambda _a, _b, _c: range(_a + 1),
                  ):
    W[a, b, c] += np.power(-1, alpha) * np.power(2, a - alpha) * \
      nchoosek(a, alpha) * V[a - alpha, b, c + 2 * alpha]

  X = np.zeros((N + 1, N + 1, N + 1), dtype=complex)
  for a, b, c, alpha in nest(lambda: range(int(N / 2) + 1),
                  lambda _a: range(N - 2 * _a + 1),
                  lambda _a, _b: range(N - 2 * _a - _b + 1),
                  lambda _a, _b, _c: range(_a + 1),
                  ):
    X[a, b, c] += nchoosek(a, alpha) * W[a - alpha, b + 2 * alpha, c]

  Y = np.zeros((N + 1, N + 1, N + 1), dtype=complex)
  for l, nu, m, j in nest(lambda: range(N + 1),
              lambda _l: range(int((N - _l) / 2) + 1),
              lambda _l, _nu: range(_l + 1),
              lambda _l, _nu, _m: range(int((_l - _m) / 2) + 1),
              ):
    Y[l, nu, m] += Yljm(l, j, m) * X[nu + j, l - m - 2 * j, m]

  Z = np.zeros((N + 1, N + 1, N + 1), dtype=complex)
  for n, l, m, nu, in nest(lambda: range(N + 1),
                lambda _n: range(_n + 1),
                # there's an if...mod missing in this but it
                # still works?
                lambda _n, _l: range(_l + 1),
                lambda _n, _l, _m: range(int((_n - _l) / 2) + 1),
                ):
    # integer required for k when used as power in Qklnu below:
    k = int((n - l) / 2)
    Z[n, l, m] += (3 / (4 * PI_CONST)) * \
      Qklnu(k, l, nu) * np.conj(Y[l, nu, m])

  for n, l, m in nest(lambda: range(N + 1),
            lambda _n: range(n + 1),
            lambda _n, _l: range(l + 1),
            ):
    if np.mod(np.sum([n, l, m]), 2) == 0:
      Z[n, l, m] = np.real(
        Z[n, l, m]) - np.imag(Z[n, l, m]) * IMAG_CONST
    else:
      Z[n, l, m] = -np.real(Z[n, l, m]) + \
        np.imag(Z[n, l, m]) * IMAG_CONST

  return Z


def Yljm(l, j, m):
  aux_1 = np.power(-1, j) * (np.sqrt(2 * l + 1) / np.power(2, l))
  aux_2 = trinomial(
    m, j, l - m - 2 * j) * nchoosek(2 * (l - j), l - j)
  aux_3 = np.sqrt(trinomial(m, m, l - m))
  y = (aux_1 * aux_2) / aux_3
  return y


def Qklnu(k, l, nu):
  aux_1 = np.power(-1, k + nu) / np.power(4, k).astype(float)
  aux_2 = np.sqrt((2 * l + 4 * k + 3) / 3.0)
  aux_3 = trinomial(
    nu, k - nu, l + nu + 1) * nchoosek(2 * (l + nu + 1 + k), l + nu + 1 + k)
  aux_4 = nchoosek(2.0 * (l + nu + 1), l + nu + 1)
  return (aux_1 * aux_2 * aux_3) / aux_4


# @numba.jit()
def feature_extraction(Z, N):
  F = np.zeros((N + 1, N + 1)) - 1  # +NAN_CONST
  for n in range(N + 1):
    for l in range(n + 1):
      if np.mod(n - l, 2) != 0:
        continue
      aux_1 = Z[n, l, 0:(l + 1)]
      if l > 0:
        aux_2 = np.conj(aux_1[1:(l + 1)])
        for m in range(0, l):
          aux_2[m] = aux_2[m] * np.power(-1, m + 1)
        aux_2 = np.flipud(aux_2)
        aux_1 = np.concatenate((aux_2, aux_1))
      F[n, l] = np.linalg.norm(aux_1, ord=2)
  F = F.transpose()
  mask = F >= 0
  return F[mask]


def zernike_moments(points, faces, order=10, scale_input=True,
                    decimate_fraction=0, decimate_smooth=0, verbose=False):
    """
    Compute the Zernike moments of a surface patch of points and faces.

    Optionally decimate the input mesh.

    Note::
      Decimation sometimes leads to an error of "Segmentation fault: 11"
      (Twins-2-1 left label 14 gives such an error only when decimated.)

    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex
    faces : list of lists of 3 integers
        each list contains indices to vertices that form a triangle on a mesh
    order : integer
        order of the moments being calculated
    scale_input : bool
        translate and scale each object so it is bounded by a unit sphere?
        (this is the expected input to zernike_moments())
    decimate_fraction : float
        fraction of mesh faces to remove for decimation (0 for no decimation)
    decimate_smooth : integer
        number of smoothing steps for decimation
    verbose : bool
        print statements?

    Returns
    -------
    descriptors : list of floats
        Zernike descriptors

    Examples
    --------
    >>> # Example 1: simple cube (decimation results in a Segmentation Fault):
    >>> import numpy as np
    >>> from mindboggle.shapes.zernike.zernike import zernike_moments
    >>> points = [[0,0,0], [1,0,0], [0,0,1], [0,1,1],
    ...           [1,0,1], [0,1,0], [1,1,1], [1,1,0]]
    >>> faces = [[0,2,4], [0,1,4], [2,3,4], [3,4,5], [3,5,6], [0,1,7]]
    >>> order = 3
    >>> scale_input = True
    >>> decimate_fraction = 0
    >>> decimate_smooth = 0
    >>> verbose = False
    >>> descriptors = zernike_moments(points, faces, order, scale_input,
    ...     decimate_fraction, decimate_smooth, verbose)
    >>> [np.float("{0:.{1}f}".format(x, 5)) for x in descriptors]
    [0.09189, 0.09357, 0.04309, 0.06466, 0.0382, 0.04138]

    Example 2: Twins-2-1 left postcentral pial surface -- NO decimation:
               (zernike_moments took 142 seconds for order = 3 with no decimation)

    >>> from mindboggle.shapes.zernike.zernike import zernike_moments
    >>> from mindboggle.mio.vtks import read_vtk
    >>> from mindboggle.guts.mesh import keep_faces
    >>> from mindboggle.mio.fetch_data import prep_tests
    >>> urls, fetch_data = prep_tests()
    >>> label_file = fetch_data(urls['left_freesurfer_labels'], '', '.vtk')
    >>> points, f1,f2, faces, labels, f3,f4,f5 = read_vtk(label_file)
    >>> I22 = [i for i,x in enumerate(labels) if x==1022] # postcentral
    >>> faces = keep_faces(faces, I22)
    >>> order = 3
    >>> scale_input = True
    >>> decimate_fraction = 0
    >>> decimate_smooth = 0
    >>> verbose = False
    >>> descriptors = zernike_moments(points, faces, order, scale_input,
    ...     decimate_fraction, decimate_smooth, verbose)
    >>> [np.float("{0:.{1}f}".format(x, 5)) for x in descriptors]
    [0.00471, 0.0084, 0.00295, 0.00762, 0.0014, 0.00076]

    Example 3: left postcentral + pars triangularis pial surfaces:

    >>> from mindboggle.mio.vtks import read_vtk, write_vtk
    >>> points, f1,f2, faces, labels, f3,f4,f5 = read_vtk(label_file)
    >>> I20 = [i for i,x in enumerate(labels) if x==1020] # pars triangularis
    >>> I22 = [i for i,x in enumerate(labels) if x==1022] # postcentral
    >>> I22.extend(I20)
    >>> faces = keep_faces(faces, I22)
    >>> order = 3
    >>> scale_input = True
    >>> decimate_fraction = 0
    >>> decimate_smooth = 0
    >>> verbose = False
    >>> descriptors = zernike_moments(points, faces, order, scale_input,
    ...     decimate_fraction, decimate_smooth, verbose)
    >>> [np.float("{0:.{1}f}".format(x, 5)) for x in descriptors]
    [0.00586, 0.00973, 0.00322, 0.00818, 0.0013, 0.00131]

    View both segments (skip test):

    >>> from mindboggle.mio.plots import plot_surfaces # doctest: +SKIP
    >>> from mindboggle.mio.vtks import rewrite_scalars # doctest: +SKIP
    >>> scalars = -1 * np.ones(np.shape(labels)) # doctest: +SKIP
    >>> scalars[I22] = 1 # doctest: +SKIP
    >>> rewrite_scalars(label_file, 'test_two_labels.vtk', scalars,
    ...                 'two_labels', scalars) # doctest: +SKIP
    >>> plot_surfaces(vtk_file) # doctest: +SKIP

    """
    # Convert lists to numpy arrays:
    if isinstance(points, list):
      points = np.array(points)
    if isinstance(faces, list):
      faces = np.array(faces)

    # ------------------------------------------------------------------------
    # Translate all points so that they are centered at their mean,
    # and scale them so that they are bounded by a unit sphere:
    # ------------------------------------------------------------------------
    if scale_input:
      center = np.mean(points, axis=0)
      points = points - center
      maxd = np.max(np.sqrt(np.sum(points**2, axis=1)))
      points /= maxd

    # ------------------------------------------------------------------------
    # Geometric moments:
    # ------------------------------------------------------------------------
    G = geometric_moments_exact(points, faces, order)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    Z = zernike(G, order)

    # ------------------------------------------------------------------------
    # Extract Zernike descriptors:
    # ------------------------------------------------------------------------
    descriptors = feature_extraction(Z, order).tolist()

    if verbose:
      print("Zernike moments: {0}".format(descriptors))

    return descriptors