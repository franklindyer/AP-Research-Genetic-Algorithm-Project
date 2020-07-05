import sqlite3
import numpy as np
import math
import time
import copy
import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

log = open("log_sandbox.txt","a")

reprods = ['AS','2P1PX','2P2PX','3P2PX']
mut_probs = [0.01,0.03,0.05,0.07]

def test_fxn():
    try:
        sqliteConnection = sqlite3.connect('datenbank.db')
        cursor = sqliteConnection.cursor()
        for x,y in itertools.product(reprods,mut_probs):
            cursor.execute('SELECT max_score FROM gen_alg_tabelle WHERE reprod="'+x+'" AND mut_prob='+str(y)+' AND gen_num=99')
            to_be_crunched = [row[0] for row in cursor]
            log.write('For reprod='+x+' and mut_prob='+str(y)+', the average max end score is '+str(np.average(to_be_crunched))+'\n')
        cursor.close()
    except:
        log.write('Connection failed.')
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
    log.flush()
    os.fsync(log.fileno())


def test_fxn_2():
    try:
        sqliteConnection = sqlite3.connect('datenbank.db')
        cursor = sqliteConnection.cursor()
        cursor.execute('SELECT hamm_dist FROM gen_alg_tabelle WHERE reprod="2PSX" AND mut_prob=0.01 AND gen_num=100')
        to_be_crunched = sorted([row[0] for row in cursor])
        plt.scatter(range(0,len(to_be_crunched)),to_be_crunched)
        cursor.close()
        plt.savefig('images/2PSX_0.01_endhammdist.png')
    except:
        log.write('Connection failed.')
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
    log.flush()
    os.fsync(log.fileno())


def grouped_bar_chart(mut_probs,reprods,measurement='max_score',when='100',name='sample_title'):
    plt.clf()
    plt.figure(figsize=(15,6))
    mp_dim = len(mut_probs)
    rp_dim = len(reprods)
    big_data_nest = []
    try:
        sqliteConnection = sqlite3.connect('datenbank.db')
        cursor = sqliteConnection.cursor()
        for x in range(0,mp_dim):
            mp_specific_data = []
            for y in range(0,rp_dim):
                if when == 'av': 
                    cursor.execute('SELECT id FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num=1')
                    relevant_ids = [row[0] for row in cursor]
                    rp_specific_data = []
                    for i in relevant_ids:
                        cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE id>'+str(i-1)+' AND id<'+str(i+100))
                        rp_specific_data.append(np.average([row[0] for row in cursor]))
                else:
                    cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num='+str(when))
                    rp_specific_data = [row[0] for row in cursor]
                mp_specific_data.append([np.average(rp_specific_data),np.std(rp_specific_data)])
            big_data_nest.append(mp_specific_data)
        cursor.close()
    except:
        log.write('Connection failed.')
    finally:
        if (sqliteConnection):
            sqliteConnection.close()

    bar_width = 1.5/(mp_dim+1)
    bar_heights = [[x[0] for x in y] for y in big_data_nest]
    bar_errors = [[x[1] for x in y] for y in big_data_nest]
    bar_colors = ['#ff82ab','#ee7ae9','#b23aef','#836fff','#3d59ab']
    x_positions = [[1.5*i+bar_width*j for i in range(0,rp_dim)] for j in range(0,mp_dim)]
    for i in range(0,mp_dim): 
        plt.bar(x_positions[i],bar_heights[i],color=bar_colors[i],width=bar_width,edgecolor='white',label=mut_probs[i])
        plt.errorbar(x_positions[i],bar_heights[i],bar_errors[i],fmt='None',ecolor='black')
    plt.xlabel('Crossover Operator',fontweight='bold')
    plt.xticks([1.5*r+bar_width for r in range(0,rp_dim)],reprods)
    plt.title(name)
    plt.legend(bbox_to_anchor=(1.1,1.05))
    plt.savefig('images/'+name+'.png')


def big_data_table(mut_probs,reprods,measurement='max_score',when='100',name='sample_title',stat='mean'):
    plt.clf() 
    plt.figure(figsize=(20,15))
    mp_dim = len(mut_probs)
    rp_dim = len(reprods)
    big_data_nest = []
    try:
        sqliteConnection = sqlite3.connect('datenbank.db')
        cursor = sqliteConnection.cursor()
        for x in range(0,mp_dim):
            mp_specific_data = []
            for y in range(0,rp_dim):
                if when == 'av': 
                    cursor.execute('SELECT id FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num=1')
                    relevant_ids = [row[0] for row in cursor]
                    rp_specific_data = []
                    for i in relevant_ids:
                        cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE id>'+str(i-1)+' AND id<'+str(i+100))
                        rp_specific_data.append(np.average([row[0] for row in cursor]))
                else:
                    cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num='+str(when))
                    rp_specific_data = [row[0] for row in cursor]
                if stat == 'mean':
                    mp_specific_data.append(round(np.average(rp_specific_data),5))
                elif stat == 'std':
                    mp_specific_data.append(round(np.std(rp_specific_data),5))
            big_data_nest.append(mp_specific_data)
        cursor.close()
    except:
        log.write('Connection failed.')
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
    color_list = ['#ff82ab','#ee7ae9','#b23aef','#836fff','#3d59ab']
    cell_colors = [[x]*rp_dim for x in color_list]
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(big_data_nest) 
    ax.table(big_data_nest,cellColours=cell_colors,loc='center',colLabels=reprods)
 ##   plt.title(name)
    fig.tight_layout()
    plt.savefig('images/'+name+'.png')

## note that the first entry in the "reprods" array will be interpreted at the CO to be compared against the others

def confidence_table(mut_probs,reprods,measurement='max_score',when='100',name='sample_title'):
    plt.clf() 
    plt.close('all')
    plt.figure(figsize=(20,15))
    mp_dim = len(mut_probs)
    rp_dim = len(reprods)
    big_data_nest = []
    try:
        sqliteConnection = sqlite3.connect('datenbank.db')
        cursor = sqliteConnection.cursor()
        for x in range(0,mp_dim):
            mp_specific_data = []
            for y in range(0,rp_dim):
                if when == 'av': 
                    cursor.execute('SELECT id FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num=1')
                    relevant_ids = [row[0] for row in cursor]
                    rp_specific_data = []
                    for i in relevant_ids:
                        cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE id>'+str(i-1)+' AND id<'+str(i+100))
                        rp_specific_data.append(np.average([row[0] for row in cursor]))
                else:
                    cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num='+str(when))
                    rp_specific_data = [row[0] for row in cursor]
                mp_specific_data.append([np.average(rp_specific_data),np.std(rp_specific_data)])
            ##p_array = [1-stats.t.cdf(t,198) for t in t_array]
            big_data_nest.append(mp_specific_data)
        cursor.close()
    except:
        log.write('Connection failed.')
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
    t_values = [[np.round((y[0]-x[0][0])/((x[0][1]**2)/100+(y[1]**2)/100)**(1/2),5) for y in x[1:]] for x in big_data_nest]
    color_list = ['#ff82ab','#ee7ae9','#b23aef','#836fff','#3d59ab']
    cell_colors = [[x]*(rp_dim-1) for x in color_list]
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(big_data_nest) 
    ax.table(t_values,cellColours=cell_colors,loc='center',colLabels=reprods[1:])
 ##   plt.title(name)
    fig.tight_layout()
    plt.savefig('images/'+name+'.png')

def confidence_table_2(mut_probs,reprods,measurement='max_score',when='100',name='sample_title'):
    plt.clf() 
    plt.close('all')
    plt.figure(figsize=(20,15))
    mp_dim = len(mut_probs)
    rp_dim = len(reprods)
    big_data_nest = []
    try:
        sqliteConnection = sqlite3.connect('datenbank.db')
        cursor = sqliteConnection.cursor()
        for x in range(0,mp_dim):
            mp_specific_data = []
            for y in range(0,rp_dim):
                if when == 'av': 
                    cursor.execute('SELECT id FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num=1')
                    relevant_ids = [row[0] for row in cursor]
                    rp_specific_data = []
                    for i in relevant_ids:
                        cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE id>'+str(i-1)+' AND id<'+str(i+100))
                        rp_specific_data.append(np.average([row[0] for row in cursor]))
                else:
                    cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num='+str(when))
                    rp_specific_data = [row[0] for row in cursor]
                mp_specific_data.append([np.average(rp_specific_data),np.std(rp_specific_data)])
            ##p_array = [1-stats.t.cdf(t,198) for t in t_array]
            big_data_nest.append(mp_specific_data)
        cursor.close()
    except:
        log.write('Connection failed.')
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
    t_values = [[np.abs(np.round((big_data_nest[0][i][0]-big_data_nest[1][i][0])/((big_data_nest[0][i][1]**2)/100+(big_data_nest[1][i][1]**2)/100)**(1/2),5)) for i in range(0,len(reprods))]]
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(big_data_nest) 
    ax.table(t_values,loc='center',colLabels=reprods)
 ##   plt.title(name)
    fig.tight_layout()
    plt.savefig('images/'+name+'.png')

def confidence_table_3(mut_probs,reprods,measurement='max_score',when='100',name='sample_title'):
    plt.clf() 
    plt.close('all')
    plt.figure(figsize=(10,10))
    mp_dim = len(mut_probs)
    rp_dim = len(reprods)
    big_data_nest = []
    try:
        sqliteConnection = sqlite3.connect('datenbank.db')
        cursor = sqliteConnection.cursor()
        for x in range(0,mp_dim):
            mp_specific_data = []
            for y in range(0,rp_dim):
                if when == 'av': 
                    cursor.execute('SELECT id FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num=1')
                    relevant_ids = [row[0] for row in cursor]
                    rp_specific_data = []
                    for i in relevant_ids:
                        cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE id>'+str(i-1)+' AND id<'+str(i+100))
                        rp_specific_data.append(np.average([row[0] for row in cursor]))
                else:
                    cursor.execute('SELECT '+measurement+' FROM gen_alg_tabelle WHERE reprod="'+reprods[y]+'" AND mut_prob='+str(mut_probs[x])+' AND gen_num='+str(when))
                    rp_specific_data = [row[0] for row in cursor]
                mp_specific_data.append([np.average(rp_specific_data),np.std(rp_specific_data)])
            ##p_array = [1-stats.t.cdf(t,198) for t in t_array]
            big_data_nest.append(mp_specific_data)
        cursor.close()
    except:
        log.write('Connection failed.')
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
    t_values = [[np.round((x[y][0]-x[y+1][0])/((x[y][1]**2)/100+(x[y+1][1]**2)/100)**(1/2),5) for y in [0]] for x in big_data_nest]
    color_list = ['#ff82ab','#ee7ae9','#b23aef','#836fff','#3d59ab']
    cell_colors = [[x] for x in color_list]
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(big_data_nest) 
    ax.table(t_values,cellColours=cell_colors,loc='center')
 ##   plt.title(name)
    fig.tight_layout()
    plt.savefig('images/'+name+'.png')

for x in ['2P1PX','2P2PX','3P2PX']:
    confidence_table([0.005,0.01,0.03,0.05,0.07],[x,'2PSX','3PSX','2PUX','3PUX','W2PUX','W3PUX','UUX'],measurement='hamm_dist',when=100,name='z-scores for the claim that '+x+'<S2 with respect to final hamming distance')
    confidence_table([0.005,0.01,0.03,0.05,0.07],[x,'2PSX','3PSX','2PUX','3PUX','W2PUX','W3PUX','UUX'],measurement='hamm_dist',when='av',name='z-scores for the claim that '+x+'<S2 with respect to average hamming distance')

## test_fxn_2()

## for s1,meas,when in itertools.product(['UUX'],['av_score','max_score'],[100,'av']):
##    confidence_table([0.005,0.01,0.03,0.05,0.07],[s1,'AS','2P1PX','2P2PX','3P2PX','2PSX','3PSX','2PUX','3PUX','W2PUX','W3PUX'],meas,when,'z-values for the claim '+s1+'>X wrt '+str(when)+' '+meas)

##grouped_bar_chart([0.005,0.01,0.03,0.05,0.07],['AS','2P1PX','2P2PX','3P2PX','2PSX','3PSX','2PUX','3PUX','W2PUX','W3PUX','UUX'],'hamm_dist','av','Comparison of average average hamming distance')

##for meas, when in itertools.product(['max_score','av_score','var_scores','hamm_dist'],[25,50,75,100,'av']):
##    grouped_bar_chart([0.005,0.01,0.03,0.05,0.07],['AS','2P1PX','2P2PX','3P2PX','2PSX','3PSX'],meas,when,'bars_'+str(meas)+'_'+str(when)+'_part1')
##    grouped_bar_chart([0.005,0.01,0.03,0.05,0.07],['2PUX','3PUX','W2PUX','W3PUX','UUX'],meas,when,'bars_'+str(meas)+'_'+str(when)+'_part2')



log.flush()
os.fsync(log.fileno())

