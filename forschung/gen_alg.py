import datetime
import sqlite3
import itertools
import random
import numpy as np
import math
import time
import copy
import os

def prog_loggy(message): 
    log = open("prog_log_2.txt","a")
    log.write(message)
    log.flush()
    os.fsync(log.fileno())

def gen_grid(r,c,p):
        return [[1 if np.random.random()<p else 0 for i in range(0, c)] for j in range(0,r)]

def generate_pop(size):
	return [[[random.randint(0,3) for i in range(0,256)],[0,0],0] for j in range(0,size)]

## plays a game with a given player, grid, and number of turns, with the additional option of saving a record of the player's positions over the course of the game, so that the game can later be graphically displayed and qualitatively analyzed

def hamming_distance(p1,p2):
	return(sum([int(p1[0][i] != p2[0][i]) for i in range(0,256)]))

def hamming_diversity(pop):
	pop_size = len(pop)
	sum_distances = 0
	for i in range(0,pop_size-1):
		for j in range(i+1,pop_size):
			sum_distances += hamming_distance(pop[i],pop[j])
	return(2*sum_distances/(pop_size**2-pop_size))

def play_game(player, original_grid, duration, log=0):
	grid = copy.deepcopy(original_grid)
	rows, cols = len(grid),len(grid[0])
	player[1] = [0,0]
	movement_record = [[0,0]]
	for k in range(0,duration):
		## looks at all of the squares surrounding the player, on which it will base its next move
		surroundings = [grid[(player[1][0] + i) % rows][(player[1][1] + j) % cols] for i,j in itertools.product([-1,0,1],[-1,0,1])]
		del surroundings[4]
		## establishes a mapping based on binary sequences between the possible states of the surroundings and the corresponding instruction in the gene sequence
		for i in range(0,8): surroundings[i] = surroundings[i] * (2 ** i)
		gene_instr = player[0][sum(surroundings)]
		## executes instruction encoded in the gene
		if gene_instr == 0 or gene_instr == 2:
			player[1][0] = (player[1][0] + gene_instr - 1) % rows
		else:
			player[1][1] = (player[1][1] + gene_instr - 2) % cols
		## gives player "points" for getting a token and removes the token
		if grid[player[1][0]][player[1][1]] == 1:
			player[2] += 1
			grid[player[1][0]][player[1][1]] = 0
		if log == 1: movement_record.append(player[1].copy())
	if log == 1: return(movement_record)

## given a population of players who have already been evaluated and assigned "scores," this function outputs a the same population sorted in order of their scores

def sort_by_scores(population):
	sorted_players = []
	for j in range(0,len(population)):
		top_player = [0,0,0]
		top_player_index = 0
		for i in range(0, len(population)):
			if population[i][2] >= top_player[2]:
				top_player = population[i]
				top_player_index = i
		sorted_players.append([top_player[0].copy(),[0,0],top_player[2]])
		del population[top_player_index]
	return(sorted_players)
	
## randomly mutates characters of a gene sequence uniformly and with a specified probability

def mutate_genes(gene_seq, mutation_prob):
	for i in range(0,len(gene_seq)):
		if random.random() < mutation_prob:
			gene_seq[i] = random.randint(1,4)
	return(gene_seq)

## generates a specified number of grids with specified parameters and evaluates all players in a given population using those grids, then selects the most fit players and allows them to reproduce using a specified reproduction mechanism; returns the new population and the score distribution in a new array

def evaluation_selection_reproduction(population, trials, r, c, p, duration, mutation_prob, reproduction_method='asexual'):
	pop_size = len(population)
	grids = [gen_grid(r,c,p) for i in range(0,trials)]
	for player, grid in itertools.product(population, grids): play_game(player, grid, duration)
	scores = [x[2] for x in population]
	gene_pool = sort_by_scores(population)[0:math.floor(pop_size/2)]
	new_pop = []
	if reproduction_method == 'AS':
		for i in range(0, pop_size): new_pop.append([mutate_genes(copy.deepcopy(random.choice(gene_pool))[0],mutation_prob),[0,0],0])
	if reproduction_method == '2P1PX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,2)]
			cutoff = random.randint(0,255)
			new_gene_seq = mutate_genes(parents[0][0][:cutoff] + parents[1][0][cutoff:],mutation_prob)
			new_pop.append([new_gene_seq,[0,0],0])
	if reproduction_method == '2P2PX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,2)]
			cuts = [random.randint(0,255) for j in range(0,2)]
			cutoff2,cutoff1 = max(cuts[0],cuts[1]),min(cuts[0],cuts[1])
			new_gene_seq = mutate_genes(parents[0][0][:cutoff1]+parents[1][0][cutoff1:cutoff2]+parents[0][0][cutoff2:],mutation_prob)
			new_pop.append([new_gene_seq,[0,0],0])
	if reproduction_method == '3P2PX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,3)]
			cuts = [random.randint(0,255) for j in range(0,2)]
			cutoff2,cutoff1 = max(cuts[0],cuts[1]),min(cuts[0],cuts[1])
			new_gene_seq = mutate_genes(parents[0][0][:cutoff1]+parents[1][0][cutoff1:cutoff2]+parents[2][0][cutoff2:],mutation_prob)
			new_pop.append([new_gene_seq,[0,0],0])
	if reproduction_method == '2PSX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,2)]
			new_gene_seq = []
			par_num = 0
			for j in range(0,256):
				new_gene_seq.append(parents[par_num][0][j])
				if random.random() < 0.2: par_num = 1-par_num
			new_pop.append([mutate_genes(new_gene_seq,mutation_prob),[0,0],0])
	if reproduction_method == '3PSX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,3)]
			new_gene_seq = []
			par_num = 0
			for j in range(0,256):
				new_gene_seq.append(parents[par_num][0][j])
				if random.random() < 0.2: par_num = (par_num+1)%3
			new_pop.append([mutate_genes(new_gene_seq,mutation_prob),[0,0],0])
	if reproduction_method == '2PUX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,2)]
			new_gene_seq = mutate_genes([parents[random.randint(0,1)][0][j] for j in range(0,256)],mutation_prob)
			new_pop.append([new_gene_seq,[0,0],0])
	if reproduction_method == '3PUX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,3)]
			new_gene_seq = mutate_genes([parents[random.randint(0,2)][0][j] for j in range(0,256)],mutation_prob)
			new_pop.append([new_gene_seq,[0,0],0])
	if reproduction_method == 'W2PUX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,2)]
			critical_prob = parents[0][2]/(parents[0][2]+parents[1][2])
			new_gene_seq = []
			for j in range(0,256):
				if random.random() < critical_prob: new_gene_seq.append(parents[0][0][j])
				else: new_gene_seq.append(parents[1][0][j])
			new_pop.append([mutate_genes(new_gene_seq,mutation_prob),[0,0],0])
	if reproduction_method == 'W3PUX':
		for i in range(0, pop_size):
			parents = [random.choice(gene_pool) for j in range(0,3)]
			critical_prob_1 = parents[0][2]/(parents[0][2]+parents[1][2]+parents[2][2])
			critical_prob_2 = parents[1][2]/(parents[0][2]+parents[1][2]+parents[2][2])
			new_gene_seq = []
			for j in range(0,256):
				rand = random.random()
				if rand < critical_prob_1: new_gene_seq.append(parents[0][0][j])
				elif rand < critical_prob_2: new_gene_seq.append(parents[1][0][j])
				else: new_gene_seq.append(parents[2][0][j])
			new_pop.append([mutate_genes(new_gene_seq,mutation_prob),[0,0],0])
	if reproduction_method == 'UUX':
		for i in range(0, pop_size):
			new_gene_seq = mutate_genes([random.choice(gene_pool)[0][j] for j in range(0,256)],mutation_prob)
			new_pop.append([new_gene_seq,[0,0],0])
	return([new_pop,scores])
	
## creates a random populations and repeatedly applies the evolution_selection_reproduction function for a specified number of generations

def generations(gens, trials, pop_size, r, c, p, duration, mutation_prob, reproduction_method='AS'):
	pop = generate_pop(pop_size)
	generational_scores = []
	generational_diversity = []
	for i in range(0,gens):
		generational_diversity.append(hamming_diversity(pop))
		results = evaluation_selection_reproduction(pop, trials, r, c, p, duration, mutation_prob, reproduction_method)
		pop = results[0]
		generational_scores.append([x/trials for x in results[1]])
	return([generational_scores,generational_diversity])

## animate_movement(gen_grid(20,40,0.5),[[0,0],[0,1],[1,1],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6]])

## Now I just need to write code to pluck some samples out of the population and graphically display some example games, and also decide which statistics I want to use to measure certain attributes of a population (overall performance, convergence speed, population diversity, etc). Of course, I also need to keep adding more reproduction mechanisms.

def collect_data(reprod, mut_prob, num_trials, num_gens):
    log = open("prog_log.txt","a")
    for trial in range(0, num_trials):
        data = generations(num_gens,50,50,10,10,0.5,100,mut_prob,reprod)
        new_data = [[max(data[0][gen]), sum(data[0][gen])/len(data[0][gen]), np.std(data[0][gen]), data[1][gen]] for gen in range(0,num_gens)]
        try:
            sqliteConnection = sqlite3.connect('datenbank.db')
            cursor = sqliteConnection.cursor()
            log.write("Successfully connected to database for trial " + str(trial))
            for gen in range(0, num_gens):
                sqlite_insert_query = """INSERT INTO 'gen_alg_tabelle' ('reprod', 'mut_prob', 'trial_num', 'gen_num', 'max_score', 'av_score', 'var_scores', 'hamm_dist') VALUES ('""" + reprod + """', """ + str(mut_prob) + """, """ + str(trial+1) + """, """ + str(gen+1) + """, """ + str(new_data[gen][0]) + """, """ + str(new_data[gen][1]) + """, """ + str(new_data[gen][2]) + """, """ + str(new_data[gen][3]) + """)"""
                count = cursor.execute(sqlite_insert_query)
                sqliteConnection.commit()
            cursor.close()
        except sqlite3.Error as error:
            log.write(str(datetime.datetime) + ": failed to collect data for trial " + str(trial+1) + ", generation " + str(gen+1) + ". Reprod = " + reprod + ", mut_prob = " + str(mut_prob) + ".")
            log.flush()
            os.fsync(log.fileno())
        finally:
            if (sqliteConnection):
                sqliteConnection.close()

reprod_methods = ['AS','2P1PX','2P2PX','3P2PX','2PSX','3PSX','2PUX','3PUX','W2PUX','W3PUX','UUX']
mut_probs = [0.005]


for x,y in itertools.product(reprod_methods,mut_probs):
    collect_data(x,y,100,100)
    log = open("prog_log_2.txt","a")
    log.write("Collected data for reprod="+x+" and mut_prob="+str(y))
    log.flush()
    os.fsync(log.fileno())
