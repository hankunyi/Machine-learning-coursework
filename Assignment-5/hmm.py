from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  for j in range (S):
    alpha[j][0] = pi[j] * B[j][O[0]]

  for t in range (N-1):
    for j in range(S):
      forward_sum = 0
      for i in range(S):
        forward_sum += A[i][j] * alpha[i][t]
      alpha[j][t+1] = B[j][O[t+1]] * forward_sum



  ###################################################

  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  for j in range (S):
    beta[j][N-1] = 1

  for t in range (N-1):
    for i in range (S):
      for j in range (S):
        beta[i][N-2-t] += beta[j][N-1-t] * A[i][j] * B[j][O[N-1-t]]


  ###################################################
  
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  j,t = alpha.shape
  prob = np.sum(alpha[:,t-1])
  ###################################################
  
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  i,t = beta.shape
  for n in range(i):
    prob += beta[n][0] * pi[n] * B[n][O[0]]




  ###################################################
  
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  S = len(pi)
  N = len(O)
  delta = np.zeros([S, N])
  path_all = np.zeros([S,N-1],dtype=int)
  for j in range(S):
    delta[j][0] = pi[j] * B[j][O[0]]
  for t in range (N-1):
    for j in range(S):
      delta_new = np.zeros(S)
      for i in range(S):
        delta_new[i] = delta[i][t] * A[i][j] * B[j][O[t+1]]
      delta[j][t+1] = np.max(delta_new)
      path_all[j][t] = np.argmax(delta_new)

  final_state = np.argmax(delta[:,N-1])
  for n in range(N-1):
    path.insert(0,final_state)
    final_state = path_all[final_state][N-2-n]
  path.insert(0,final_state)




  ###################################################
  
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()