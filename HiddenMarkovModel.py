import torch
import numpy as np
import pdb

class HiddenMarkovModel(object):
    """
    Hidden Markov self Class

    Parameters:
    -----------
    
    - S: Number of states.
    - T: numpy.array Transition matrix of size S by S
         stores probability from state i to state j.
    - E: numpy.array Emission matrix of size S by N (number of observations)
         stores the probability of observing  O_j  from state  S_i. 
    - T0: numpy.array Initial state probabilities of size S.
    """

    def __init__(self, T, E, T0, epsilon = 0.001, maxStep = 10):
        # Max number of iteration
        self.maxStep = maxStep
        # convergence criteria
        self.epsilon = epsilon 
        # Number of possible states
        self.S = T.shape[0]
        # Number of possible observations
        self.O = E.shape[0]
        self.prob_state_1 = []
        # Emission probability
        self.E = torch.tensor(E)
        # Transition matrix
        self.T = torch.tensor(T)
        # Initial state vector
        self.T0 = torch.tensor(T0)
        
    def initialize_viterbi_variables(self, shape): 
        pathStates = torch.zeros(shape, dtype=torch.float64)
        pathScores = torch.zeros_like(pathStates)
        states_seq = torch.zeros([shape[0]], dtype=torch.int64)
        return pathStates, pathScores, states_seq
    
    def belief_propagation(self, scores):
        return scores.view(-1,1) + torch.log(self.T)
    
    def viterbi_inference(self, x): # x: observing sequence        
        self.N = len(x)
        shape = [self.N, self.S]
        # Init_viterbi_variables
        pathStates, pathScores, states_seq = self.initialize_viterbi_variables(shape) 
        # log probability of emission sequence
        obs_prob_full = torch.log(self.E[x])
        # initialize with state starting log-priors
        pathScores[0] = torch.log(self.T0) + obs_prob_full[0]
        for step, obs_prob in enumerate(obs_prob_full[1:]):
            # propagate state belief
            belief = self.belief_propagation(pathScores[step, :])
            # the inferred state by maximizing global function
            pathStates[step + 1] = torch.argmax(belief, 0)
            # and update state and score matrices 
            pathScores[step + 1] = torch.max(belief, 0)[0] + obs_prob
        # infer most likely last state
        states_seq[self.N - 1] = torch.argmax(pathScores[self.N-1, :], 0)
        for step in range(self.N - 1, 0, -1):
            # for every timestep retrieve inferred state
            state = states_seq[step]
            state_prob = pathStates[step][state]
            states_seq[step -1] = state_prob
        return states_seq, torch.exp(pathScores) # turn scores back to probabilities
    
    def initialize_forw_back_variables(self, shape):
        self.forward = torch.zeros(shape, dtype=torch.float64)
        self.backward = torch.zeros_like(self.forward)
        self.posterior = torch.zeros_like(self.forward)
        
    def _forward(model, obs_prob_seq):
        model.scale = torch.zeros([model.N], dtype=torch.float64) #scale factors
        # initialize with state starting priors
        init_prob = model.T0 * obs_prob_seq[0]
        # scaling factor at t=0
        model.scale[0] = 1.0 / init_prob.sum()
        # scaled belief at t=0
        model.forward[0] = model.scale[0] * init_prob
         # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq[1:]):
            # previous state probability
            prev_prob = model.forward[step].unsqueeze(0)
            # transition prior
            prior_prob = torch.matmul(prev_prob, model.T)
            # forward belief propagation
            forward_score = prior_prob * obs_prob
            forward_prob = torch.squeeze(forward_score)
            # scaling factor
            model.scale[step + 1] = 1 / forward_prob.sum()
            # Update forward matrix
            model.forward[step + 1] = model.scale[step + 1] * forward_prob
    
    def _backward(self, obs_prob_seq_rev):
        # initialize with state ending priors
        self.backward[0] = self.scale[self.N - 1] * torch.ones([self.S], dtype=torch.float64)
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):
            # next state probability
            next_prob = self.backward[step, :].unsqueeze(1)
            # observation emission probabilities
            obs_prob_d = torch.diag(obs_prob)
            # transition prior
            prior_prob = torch.matmul(self.T, obs_prob_d)
            # backward belief propagation
            backward_prob = torch.matmul(prior_prob, next_prob).squeeze()
            # Update backward matrix
            self.backward[step + 1] = self.scale[self.N - 2 - step] * backward_prob
        self.backward = torch.flip(self.backward, [0, 1])
        
    def forward_backward(self, obs_prob_seq):
        """
        runs forward backward algorithm on observation sequence

        Arguments
        ---------
        - obs_prob_seq : matrix of size N by S, where N is number of timesteps and
            S is the number of states

        Returns
        -------
        - forward : matrix of size N by S representing
            the forward probability of each state at each time step
        - backward : matrix of size N by S representing
            the backward probability of each state at each time step
        - posterior : matrix of size N by S representing
            the posterior probability of each state at each time step
        """        
        self._forward(obs_prob_seq)
        obs_prob_seq_rev = torch.flip(obs_prob_seq, [0, 1])
        self._backward(obs_prob_seq_rev)
    
    def re_estimate_transition(self, x):
        self.M = torch.zeros([self.N - 1, self.S, self.S], dtype = torch.float64)

        for t in range(self.N - 1):
            tmp_0 = torch.matmul(self.forward[t].unsqueeze(0), self.T)
            tmp_1 = tmp_0 * self.E[x[t + 1]].unsqueeze(0)
            denom = torch.matmul(tmp_1, self.backward[t + 1].unsqueeze(1)).squeeze()

            trans_re_estimate = torch.zeros([self.S, self.S], dtype = torch.float64)

            for i in range(self.S):
                numer = self.forward[t, i] * self.T[i, :] * self.E[x[t+1]] * self.backward[t+1]
                trans_re_estimate[i] = numer / denom

            self.M[t] = trans_re_estimate

        self.gamma = self.M.sum(2).squeeze()
        T_new = self.M.sum(0) / self.gamma.sum(0).unsqueeze(1)

        T0_new = self.gamma[0,:]

        prod = (self.forward[self.N-1] * self.backward[self.N-1]).unsqueeze(0)
        s = prod / prod.sum()
        self.gamma = torch.cat([self.gamma, s], 0)
        self.prob_state_1.append(self.gamma[:, 0]) 
        return T0_new, T_new
    
    def re_estimate_emission(self, x):
        states_marginal = self.gamma.sum(0)
        # One hot encoding buffer that you create out of the loop and just keep reusing
        seq_one_hot = torch.zeros([len(x), self.O], dtype=torch.float64)
        seq_one_hot.scatter_(1, torch.tensor(x).unsqueeze(1), 1)
        emission_score = torch.matmul(seq_one_hot.transpose_(1, 0), self.gamma)
        return emission_score / states_marginal
    
    def check_convergence(self, new_T0, new_transition, new_emission):
  
        delta_T0 = torch.max(torch.abs(self.T0 - new_T0)).item() < self.epsilon
        delta_T = torch.max(torch.abs(self.T - new_transition)).item() < self.epsilon
        delta_E = torch.max(torch.abs(self.E - new_emission)).item() < self.epsilon

        return delta_T0 and delta_T and delta_E
    
    def expectation_maximization_step(self, obs_seq):
    
        # probability of emission sequence
        obs_prob_seq = self.E[obs_seq]

        self.forward_backward(obs_prob_seq)

        new_T0, new_transition = self.re_estimate_transition(obs_seq)

        new_emission = self.re_estimate_emission(obs_seq)

        converged = self.check_convergence(new_T0, new_transition, new_emission)
        
        self.T0 = new_T0
        self.E = new_emission
        self.T = new_transition
        
        return converged
    
    def Baum_Welch_EM(self, obs_seq):
        # length of observed sequence
        self.N = len(obs_seq)

        # shape of Variables
        shape = [self.N, self.S]

        # initialize variables
        self.initialize_forw_back_variables(shape)

        converged = False

        for i in range(self.maxStep):
            converged = self.expectation_maximization_step(obs_seq)
            if converged:
                print('converged at step {}'.format(i))
                break      
        return self.T0, self.T, self.E, converged