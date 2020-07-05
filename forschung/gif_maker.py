from images2gif import writeGif
import numpy as np
from PIL import Image, ImageDraw
import random
import itertools
import copy

def gen_grid(r,c,p):
	return [[1 if np.random.random()<p else 0 for i in range(0, c)] for j in range(0,r)]

def play_game(player, original_grid, duration, log=0, scoretype='av'):
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
		if gene_instr == 4: gene_instr = random.randint(0,3)
		if gene_instr == 0 or gene_instr == 2:
			player[1][0] = (player[1][0] + gene_instr - 1) % rows
		else:
			player[1][1] = (player[1][1] + gene_instr - 2) % cols
		## gives player "points" for getting a token and removes the token
		if grid[player[1][0]][player[1][1]] == 1:
			player[2] += 1
			grid[player[1][0]][player[1][1]] = 0
		if log == 1: movement_record.append([player[1][1],player[1][0]].copy())
	if log == 1: return(movement_record)

##im = Image.new('RGB',(300,300),(256,256,256))
##draw = ImageDraw.Draw(im)
##draw.line((0,0, 100,50), fill=500, width=5)
##im.show()

def render_grid(grid):
	r = len(grid)
	c = len(grid[0])
	width = 400
	height = int(np.ceil(width*r/c))
	sq_size = width/c
	im = Image.new('RGB',(width,height), (256,256,256))
	draw = ImageDraw.Draw(im)
## draw vertical lines on grid
	for i in range(0, c+1):
		approx_x = int(np.ceil(width*i/c))
		draw.line((approx_x,0, approx_x,height), fill='lightgray', width=5)
## draw horizontal lines on grid
	for i in range(0, r+1):
		approx_y = int(np.ceil(height*i/r))
		draw.line((0,approx_y, width,approx_y), fill='lightgray', width=5)
## draw tokens on grid
	for i,j in itertools.product(range(0,r), range(0,c)):
		if grid[i][j] == 1:
			draw.ellipse(((j+1/3)*sq_size,(i+1/3)*sq_size, (j+2/3)*sq_size,(i+2/3)*sq_size), fill='#e6e600')
	return [im, sq_size]

def animate_game(grid, moves, gif_name):
    smooth_frames = 5
    ch_grid = copy.deepcopy(grid)
    frames = []
    for pos in range(0,len(moves)):
        x = moves[pos][0]
        y = moves[pos][1]
        im_info = render_grid(ch_grid)
        im = im_info[0].copy()
        sq_size = im_info[1]
        draw = ImageDraw.Draw(im)
        pos_tuple = ((x+1/3)*sq_size,(y+1/3)*sq_size, (x+2/3)*sq_size,(y+2/3)*sq_size)
        draw.rectangle(pos_tuple, fill='purple')
        draw.text((sq_size/4,sq_size/4), str(pos), fill='black')
        for i in range(0,3): frames.append(im)
        if ch_grid[y][x] == 1: ch_grid[y][x] = 0
        im_info = render_grid(ch_grid)
        if pos < len(moves)-1:
            nx = moves[pos+1][0]
            ny = moves[pos+1][1]
            for dt in [(i+1)/smooth_frames for i in range(0,smooth_frames-1)]:
                im = im_info[0].copy()
                draw = ImageDraw.Draw(im)
                if x == 0 and nx > 1:
                    between_tuple = (((x+1/3)*(1-dt)+(-1+1/3)*dt)*sq_size, ((y+1/3)*(1-dt)+(ny+1/3)*dt)*sq_size, ((x+2/3)*(1-dt)+(-1+2/3)*dt)*sq_size, ((y+2/3)*(1-dt)+(ny+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                    between_tuple = (((nx+1+1/3)*(1-dt)+(nx+1/3)*dt)*sq_size, ((y+1/3)*(1-dt)+(ny+1/3)*dt)*sq_size, ((nx+1+2/3)*(1-dt)+(nx+2/3)*dt)*sq_size, ((y+2/3)*(1-dt)+(ny+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                elif nx == 0 and x > 1:
                    between_tuple = (((x+1/3)*(1-dt)+(x+1+1/3)*dt)*sq_size, ((y+1/3)*(1-dt)+(ny+1/3)*dt)*sq_size, ((x+2/3)*(1-dt)+(x+1+2/3)*dt)*sq_size, ((y+2/3)*(1-dt)+(ny+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                    between_tuple = (((-1+1/3)*(1-dt)+(nx+1/3)*dt)*sq_size, ((y+1/3)*(1-dt)+(ny+1/3)*dt)*sq_size, ((-1+2/3)*(1-dt)+(nx+2/3)*dt)*sq_size, ((y+2/3)*(1-dt)+(ny+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                elif y == 0 and ny > 1:
                    between_tuple = (((x+1/3)*(1-dt)+(nx+1/3)*dt)*sq_size, ((y+1/3)*(1-dt)+(-1+1/3)*dt)*sq_size, ((x+2/3)*(1-dt)+(nx+2/3)*dt)*sq_size, ((y+2/3)*(1-dt)+(-1+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                    between_tuple = (((x+1/3)*(1-dt)+(nx+1/3)*dt)*sq_size, ((ny+1+1/3)*(1-dt)+(ny+1/3)*dt)*sq_size, ((x+2/3)*(1-dt)+(nx+2/3)*dt)*sq_size, ((ny+1+2/3)*(1-dt)+(ny+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                elif ny == 0 and y > 1:
                    between_tuple = (((x+1/3)*(1-dt)+(nx+1/3)*dt)*sq_size, ((y+1/3)*(1-dt)+(y+1+1/3)*dt)*sq_size, ((x+2/3)*(1-dt)+(nx+2/3)*dt)*sq_size, ((y+2/3)*(1-dt)+(y+1+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                    between_tuple = (((x+1/3)*(1-dt)+(nx+1/3)*dt)*sq_size, ((-1+1/3)*(1-dt)+(ny+1/3)*dt)*sq_size, ((x+2/3)*(1-dt)+(nx+2/3)*dt)*sq_size, ((-1+2/3)*(1-dt)+(ny+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                else:
                    between_tuple = (((x+1/3)*(1-dt)+(nx+1/3)*dt)*sq_size, ((y+1/3)*(1-dt)+(ny+1/3)*dt)*sq_size, ((x+2/3)*(1-dt)+(nx+2/3)*dt)*sq_size, ((y+2/3)*(1-dt)+(ny+2/3)*dt)*sq_size)
                    draw.rectangle(between_tuple, fill='purple')
                draw.text((sq_size/4,sq_size/4), str(pos), fill='black')
                frames.append(im)

    frames[0].save('images/'+gif_name, format='GIF', save_all=True, append_images=frames, optimize=False, duration=50, loop=0)

def test_and_animate(player, grid, duration, name='game_test'):
    game_log = play_game(player, grid, duration, log=1)
    animate_game(grid, game_log, name+'.gif')

grid = gen_grid(20,20,0.5)
##animate_game(grid, [[0,0],[0,1],[0,2],[0,3],[0,4],[1,4],[2,4],[2,5],[2,6]], 'fucky_gopher.gif')
sample_player = [[2, 2, 2, 2, 4, 2, 0, 0, 1, 3, 0, 1, 1, 3, 1, 1, 3, 3, 3, 0, 3, 3, 0, 0, 3, 0, 0, 1, 4, 1, 3, 2, 2, 3, 4, 3, 0, 2, 0, 4, 1, 1, 3, 1, 4, 3, 1, 1, 3, 3, 4, 2, 3, 0, 3, 0, 1, 3, 4, 2, 0, 3, 3, 1, 2, 2, 0, 0, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 4, 2, 3, 2, 2, 3, 2, 0, 2, 1, 4, 2, 3, 3, 2, 1, 2, 2, 0, 2, 1, 2, 1, 2, 4, 1, 0, 3, 4, 1, 4, 4, 0, 2, 3, 3, 1, 2, 4, 1, 1, 1, 4, 2, 3, 0, 3, 3, 2, 3, 3, 2, 4, 3, 3, 4, 2, 3, 3, 4, 0, 4, 4, 4, 4, 3, 3, 3, 2, 3, 4, 0, 0, 3, 4, 0, 0, 3, 3, 1, 3, 3, 3, 0, 4, 4, 4, 0, 1, 3, 3, 0, 1, 3, 4, 4, 4, 3, 3, 4, 3, 3, 4, 3, 3, 3, 2, 0, 0, 0, 2, 3, 1, 2, 2, 2, 0, 2, 2, 2, 4, 2, 2, 2, 4, 2, 3, 2, 0, 2, 2, 2, 4, 2, 2, 0, 3, 2, 1, 4, 1, 4, 2, 2, 4, 2, 2, 2, 4, 2, 4, 4, 1, 1, 1, 2, 2, 1, 4, 1, 4, 3, 4, 2, 4, 2, 2, 1, 0, 1, 1, 0, 0, 0, 1, 1, 2],[0,0],0]
test_and_animate(sample_player, grid, 100, 'nativehabitat_20x20')
