import numpy as np
import math
import cPickle
import time

#The confusion with indexing 
#3 indexes :- User name, user vector and user no in list
#user no in list is used for nearest neighbours

#Parameters
#Gotta guess these somehow :P
k = -1.0 #positive or negative??
m = 1.0 #positive or negative??

def create_mapping(all_users):
    inverse_map = {}
    mapping = 0
    for user in all_users:
            inverse_map[user] = mapping
            mapping = mapping + 1
    return inverse_map

home = '/home/cse/dual/cs5130287/kc/'

adoption_list = open(home+'data_files/adoption_np.npy')
user_dict = open(home+'data_files/c_user_vec.p')

conf = np.load(adoption_list)
print "breakpoint 1"
user_vec = cPickle.load(user_dict)
#user_vec,conf = sys.argv[0]
print "breakpoint 2"

#import all nearest neighbours
nearest = np.load(open(home+'data_files/nn.npy'))

#generate set of all users
all_users = [int(ele) for ele in user_vec]
print "length of user_vec",len(all_users)
all_users_set = set(all_users)

#create inverse mapping from user to index
inverse_map = create_mapping(all_users)

def gen_index(entry,users):
	idx = 0
	for user in entry:
		if user in users:
			idx = idx*2+1
		else:
			idx = idx*2
	return idx


def inverse_map_list(users):
	return [inverse_map[user] for user in users]

q_matrix = [0]*math.pow(2,6) ##possible space of configurations

#calculate p(x) the underlying probability - only needs to be calculated once. Q(x) will vary based on parameters.
#generates probability of 100001 110001 and so on as noticed in all topics
for config in conf:
	adopted_users = [ele for ele in config if ele in user_vec]
	adopted_users = inverse_map_list(adopted_users)
	adopted_users_set = set(adopted_users)
	for entry in nearest:
		idx = gen_index(entry,adopted_users_set)
		q_matrix[idx] = q_matrix[idx] + 1



