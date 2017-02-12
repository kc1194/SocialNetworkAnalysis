import numpy as np
import itertools
import math
import cPickle
from scipy.spatial.distance import cdist
import sys
import heapq
import time
from scipy.spatial import cKDTree

home = '/home/cse/dual/cs5130287/kc/'
def create_mapping(all_users):
        inverse_map = {}
        mapping = 0
        for user in all_users:
                inverse_map[user] = mapping
                mapping = mapping + 1
        return inverse_map
def generate_aug(m,k,l_x_c,labels):
        power = m*l_x_c+k*labels #I dont need all labels!!!! Actually I do :P
        #num = 2*np.exp(-1*power)
        denom = np.exp(2*power) + 1 #actually then this becomes wrong!! Actually its probably correct :P
        aug = 2.0/denom
        return aug

def calculate_grads(l_x_c,labels,aug):
        grad_m = l_x_c*aug
        grad_m = np.sum(grad_m,axis=0)

        grad_k = labels*aug
        grad_k = np.sum(grad_k,axis=0)

        return grad_m,grad_k

print "Heloooo"
#import files
adoption_list = open(home+'data_files/adoption_np.npy')
user_dict = open(home+'data_files/c_user_vec.p')

#load from files
conf = np.load(adoption_list)
print "breakpoint 1"
user_vec = cPickle.load(user_dict)
#user_vec,conf = sys.argv[0]
print "breakpoint 2"

#generate set of all users
all_users = [int(ele) for ele in user_vec]
print "length of user_vec",len(all_users)
all_users_set = set(all_users)

#create inverse mapping from user to index
inverse_map = create_mapping(all_users)
                                           
#replace user names with vectors
all_vectors = [user_vec[ele] for ele in user_vec]
print "breakpoint 4"

#import all nearest neighbours
nearest = np.load(open(home+'data_files/nn.npy'))

timer1 = time.clock()
mytree = cKDTree(all_vectors)
timer2 = time.clock()

print "Time taken for building kdtree: ",timer2-timer1

def inverse_map_list(users):
        return [inverse_map[user] for user in users]

def neighbour_points(points):
        some = time.clock()
        neighbours = [nearest[ele] for ele in points]
        some_t = time.clock()
        neighbours_flat = [item for sublist in neighbours for item in sublist]
        return list(set(neighbours_flat)),some_t-some

def neighbourhood(points):
        some = time.clock()
        neighbours = [nearest[ele] for ele in points]
        some_t = time.clock()
        return neighbours,some_t-some

################################################
#BIG Question -> should I do on one configuration only? Is this really minibatch gradient Descent???
# Generate random permutation of configurations at least

#################################################

#initialise parameters 
k = -1.0
m = 1.0

#learning rates
alpha = 0.01
beta = 0.01

#random initial likelihood
ll = 1.0

iteration = 0
#start loop here
for config in conf:
        start = time.clock()
        adopted_users = [ele for ele in config if ele in user_vec]

        adopted_users = inverse_map_list(adopted_users)
        init_users = adopted_users[:len(adopted_users)/10]
        init_users = list(set(init_users))
        adopted_users_set = set(adopted_users)

        mid1 = time.clock()
        one_step_points,t1 = neighbour_points(init_users)
        mid2 = time.clock()
        two_step_points,t2 = neighbourhood(one_step_points)
        mid3 = time.clock()
        if (iteration%200 == 0):
                print " "
                print "------------------------------------------------------"
                print "number of init :    ", len(init_users)
                print "number of one-step: ", len(one_step_points)
                print "number of two-step: ", len(two_step_points)
                print "One step neighbour: ", mid2-mid1
                print "Two step neighbour: ", mid3-mid2

        phi = []
        for point1,points in zip(one_step_points,two_step_points):
                vec1 = all_vectors[point1]
                vecs = [all_vectors[ele] for ele in points]
                dist = cdist([vec1],vecs)
                sigma_x = 1.0 if point1 in adopted_users_set else -1.0
                sigma_y = np.array([1.0 if ele in adopted_users_set else -1.0 for ele in points])
                sigma_y[dist[0]==0.0] = 0.0
                dist[dist == 0.0] = 1.0
                phi_x_y = np.sum(sigma_x*sigma_y/dist)
                phi.append(phi_x_y)

        sigma = np.array([1.0 if ele in adopted_users_set else -1.0 for ele in one_step_points])
        phi = np.array(phi)

        #generate aug
        aug = generate_aug(m,k,phi,sigma)

        #calculate gradients
        grad_m,grad_k = calculate_grads(phi,sigma,aug)

        #perform update to parameters
        k = k + alpha*grad_k
        m = m + beta*grad_m

        end = time.clock()
        if (iteration%200 == 0):
                print " "
                print "------------------------------------------------------"
                print "Iteration",iteration
                print "time for iteration",end - start
                print "final parameters: k =",k,"m =",m
                iteration = iteration+1

finalfile = open(home+'finalparam.txt')
finalfile.write('k = '+repr(k)+' m = '+repr(m)+' itr = '+repr(iteration))
                                                                                                                                                                   162,1         Bot
