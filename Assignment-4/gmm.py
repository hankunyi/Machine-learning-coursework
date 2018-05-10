import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape
        updates = 0

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            centroids, membership, up = k_means.fit(x)
            self.means = centroids
            pi_k=[]
            variance = np.zeros((self.n_cluster,D,D))
            for i in range (self.n_cluster):
                count = 0
                cov = np.zeros((D,D))
                for j in range(N):
                    if (membership[j] == i):
                        count += 1
                        diff = x[j] - centroids[i]
                        cov = cov + np.dot(np.array( [diff] ).T, np.array([diff]))
                variance[i] = cov / count
                pi_k.append(count)


            self.variances = variance
            self.pi_k = np.array(pi_k) / N


            #raise Exception(
            #    'Implement initialization of variances, means, pi_k using k-means')
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            means = np.zeros((self.n_cluster, D))
            for n in range(self.n_cluster):
                means[n] = np.random.uniform(0,1,D)
            variance = np.zeros((self.n_cluster,D,D))
            for i in range(self.n_cluster):
                variance[i] = np.eye(D)
            self.means = means
            self.variances = variance
            self.pi_k = np.full(self.n_cluster, 1 / self.n_cluster )

            #raise Exception(
            #    'Implement initialization of variances, means, pi_k randomly')
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE

        log_sum = 0
        for iter in range (self.max_iter):

            respons = np.zeros((N, self.n_cluster))
            ##print('first cp')



            pinv_set = []
            det_set = []

            for k in range(self.n_cluster):
                while (np.linalg.det(self.variances[k]) == 0):
                    self.variances[k] += 0.001 * np.eye(D)
                det_set.append(np.linalg.det(self.variances[k]))
                pinv_set.append( np.linalg.inv(self.variances[k]))



            for i in range (N):
                for j in range (len(self.pi_k)):
                    diff = x[i] - self.means[j]
                    gauss = np.exp( -0.5 * np.dot (    np.dot( diff, pinv_set[j]  ) , np.array([diff]).T        ))/ np.sqrt( (2* np.pi) ** D * det_set[j])
                    respons[i][j] = self.pi_k[j] * gauss
                respons[i] /= np.sum(respons[i])

            num_k = np.sum(respons, axis=0)
            ##print('second cp')
            new_mean = np.zeros((self.n_cluster, D))
            for i in range (self.n_cluster):
                sum_mean = np.zeros ((1, D))
                for j in range (N):
                    sum_mean += respons[j][i] * x[j]
                new_mean[i] = sum_mean / num_k[i]


            new_variance = np.zeros((self.n_cluster,D,D))
            for i in range(self.n_cluster):
                sum_cov= np.zeros((D,D))
                for j in range(N):
                    sum_cov += respons[j][i] * np.dot(   np.array( [(x[j]-new_mean[i])]).T , np.array([(x[j]-new_mean[i])])          )
                new_variance[i] = sum_cov / num_k[i]


            ##print('third cp')
            new_pi_k = num_k / N

            self.means = new_mean
            self.variances = new_variance
            self.pi_k = new_pi_k
            updates += 1
            log_sum_new = self.compute_log_likelihood(x)
            ##print(log_sum_new)
            if(iter>=1 and abs(log_sum - log_sum_new) <=self.e):
                break

            log_sum = log_sum_new


        return updates

        #raise Exception('Implement fit function (filename: gmm.py)')
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE

        sampling = []
        sample_k = np.random.choice(self.n_cluster, size=N, p=self.pi_k)
        for i in range (N):
            samp = np.random.multivariate_normal (self.means[sample_k[i]], self.variances[sample_k[i]])

            sampling.append(samp)

        return np.array(sampling)





        #raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE

        N,D = x.shape
        log_sum = 0

        pinv_set = []
        det_set = []
        for k in range(self.n_cluster):
            while (np.linalg.det(self.variances[k]) == 0):
                        self.variances[k] += 0.001 * np.eye(D)
            det_set.append(np.linalg.det(self.variances[k]))
            pinv_set.append( np.linalg.inv(self.variances[k]))


        for i in range(N):
            log_xi = 0
            for j in range(self.n_cluster):
                gauss = np.exp( -0.5 * np.dot (    np.dot(  (x[i] - self.means[j]), pinv_set[j]  ) , np.array([x[i] - self.means[j]]).T        ))/ np.sqrt( (2* np.pi) ** D * det_set[j])
                log_xi += self.pi_k[j] * gauss
            log_sum += np.log(log_xi)
        return float(log_sum[0])





        #raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
