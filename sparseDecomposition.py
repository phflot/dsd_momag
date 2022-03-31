import numpy as np
from sklearn.decomposition import sparse_encode
from sklearn.utils.extmath import randomized_svd



def prox_infty(x, eta=1):
	"""
	:param x: np.ndarray(n)
	:param eta: float, size of the l_infty ball to compote the proximal operator by
	:return: np.ndarray(n), result of the l_infty proximal operator

	this method computes the l_infty proximal operator on its input with "step size" eta
	"""
	x_sorted = -np.sort(-np.abs(x))
	steps = np.cumsum(-np.diff(x_sorted, prepend=0) * np.arange(len(x_sorted)))
	tmp = (steps < eta).nonzero()[0]
	if len(tmp) > 0:
		step_ind = tmp[-1]
	else:
		step_ind = len(x_sorted) - 1
	
	slope = step_ind + 1
	s = x_sorted[step_ind] + (steps[step_ind] - eta) / slope

	return np.sign(x) * np.minimum(np.abs(x), s)


def l1_proj(x, eta=1):
	"""
	:param x: np.ndarray(n), input which is to be projected to the nearest point on the eta- l1-ball
	:param eta: float, size of the l1 ball to project to
	:return: np.ndarray(n), projected input array
	"""
	return x - prox_infty(x, eta=eta)


def l12_proj(x, eta=1):
	"""
	:param x: np.ndarray(n, m), input which should be projected to the nearest point on the eta- l21 ball
	:param eta: float, size of the l21 ball to project to

	:return: np.ndarray(n, m) projected input vector
	This method performs the projection onto the l21 ball with size eta
	"""
	norms = np.linalg.norm(x, axis=1)
	l1_norms = l1_proj(norms, eta=eta)
	x = np.where(l1_norms[:, None] > 0, x * (l1_norms / norms)[:, None], 0)
	return x


class sparsesparsePCA():
	"""
	This object implements the modified sparse dictionary algorithm where the euclidean norm constraint
	is replaced by a 2,1-norm
	"""
	def __init__(self, num_components, alpha=1, eta=1, max_iter=1000, batch_size=3):
		"""
		:param num_components: int, number of components to use for the decomposition
		:param alpha: float, sparsity regularization parameter used for the sparse coding sub-problem
		:param eta: float, norm constraint for the 2,1-norm, higher values lead to less sparse dictionaries
		:param max_iter: int, maximum number of optimization iterations
		:param batch_size: int, batch size used during the optimization procedure
		"""
		self.num_components = num_components
		self.alpha = alpha
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.eta = eta

	def fit_transform(self, X):
		"""
		:param X: np.ndarray(n, m) of float, array which should be decomposed as X=dG
		:return: np.ndarray(n, num_components) of float

		This function fits the model and returns the coding directly
		"""
		self.fit(X)
		return self.d.T

	def do_dict_loop(self, X, A, B):
		"""
		:param X: np.ndarray()
		:param A: np.ndarray()
		:param B: np.ndarray()

		This is a sub-routine used during the optimization where the dictionary entries get updated
		"""
		for k in range(self.num_components):
			# add using new component code
			if A[k, k] > 1e-6:
				self.d[k] += (B[:, k] - A[k].dot(self.d)) / A[k, k]
			else:
				newd = X[np.random.choice(len(X))]
				self.d[k] = newd + 0.01 * np.random.randn(*newd.shape)

			self.d[k] = l12_proj(self.d[k].reshape(-1, 2), eta=self.eta).flatten()

	def fit(self, X):
		"""
		:param X: np.narray(n, m), input array

		This method solves the sparse dictionary sparse coding problem constraining the l21 norm of the dictionary
		entries by iterative optimization. The results of this computation get saved in the objects attributes similar
		to the scikit-learn paradigm
		"""
		X = X.T

		self.m = X.shape[1]
		_, S, dictionary = randomized_svd(X, self.num_components, random_state=None)
		self.d = np.asfortranarray(S[:, None] * dictionary)
		A = np.zeros((self.num_components, self.num_components))
		B = np.zeros((self.m, self.num_components))
		
		for i in range(0, self.max_iter, self.batch_size):
			print(i, self.max_iter)
			x = X[np.random.choice(len(X), size=self.batch_size)]

			code = sparse_encode(x, self.d, algorithm='lasso_lars', alpha=self.alpha)

			A += code.T.dot(code)
			B += x.T.dot(code)

			self.do_dict_loop(X, A, B)

		self.code = sparse_encode(X, self.d, algorithm='lasso_lars', alpha=self.alpha)

		used_k = []
		# dropping a component if it decays to zero
		for k in range(self.num_components):
			if not np.all(self.code[:, k] == 0):
				used_k.append(k)
		print('Using {}/{} componentes'.format(len(used_k), self.num_components))
		self.d = self.d[used_k, :]
		self.code = self.code[:, used_k]

	def get_components(self):
		"""
		:return: np.ndarray(num_components, m), the sparse coding optained during the dictionary optimization
		"""
		return self.code.T