import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

class Agent():
    
    def __init__(self, pos, power, speed, energy):
        
        self.pos = np.array([int(pos[0]), int(pos[1])])
        self.power = int(power)
        self.speed = int(speed)
        self.spending = int(power * speed)
        self.energy = int(energy)
        
    def set_pos(self, pos):
        
        self.pos = np.array([int(pos[0]), int(pos[1])])
       
def create_population(surface_size, n_agents, power_mean, power_sd, 
                                              speed_mean, speed_sd,
                                              energy_mean, energy_sd):
    
    pos = np.random.rand(n_agents, 2) * surface_size
    powers = np.random.normal(power_mean, power_sd, n_agents)
    speeds = np.random.normal(speed_mean, speed_sd, n_agents)
    energies = np.random.normal(energy_mean, energy_sd, n_agents)
    
    agents = np.array([])
    
    for i in range(n_agents):
        
        agents = np.append(agents, Agent(pos[i], 
                                         powers[i], 
                                         speeds[i], 
                                         energies[i]))
        
    return agents

def create_food(surface_size, n_food):
    
    return (np.random.rand(n_food, 2) * surface_size).astype(np.int32)

def calculate_nearest_food_arg(surface_size, agent, food):

    min_distance = surface_size * math.sqrt(2)# max distance between two points
    arg = len(food)

    for i in range(len(food)):
        
        distance = math.sqrt(((agent.pos[0] - food[i][0])**2) + 
                             ((agent.pos[1] - food[i][1])**2))
        
        if distance < min_distance:
            
            min_distance = distance
            arg = i
            
    return arg

def calculate_agents_nearest_food(surface_size, agents, food):
    
    # food_competitors[food_arg][agent_arg] = 1 if: 
    # agent[agent_arg] nearest food is food[food_arg] 
    
    food_competitors = np.zeros((len(food), len(agents)))
    
    for agent_arg in range(len(agents)):
        
        food_arg = calculate_nearest_food_arg(surface_size, agents[agent_arg], food)
        food_competitors[food_arg][agent_arg] = 1
        
    return food_competitors

def calculate_distance(a, b):
    
    return math.sqrt(((a[0] - b[0])**2) + (a[1] - b[1])**2)

def calculate_energy(agents, competitors, food, f):
    
    best_competitors = np.array([], dtype = np.int32)
    
    for c in competitors:
        
        distance = calculate_distance(agents[c].pos, food[f])
        remain_energy = agents[c].energy - (agents[c].spending * distance)
        
        if remain_energy >= 0:
            
            best_competitors = np.append(best_competitors, c)
            
    return best_competitors
        
def compare_fastest(agents, competitors):
    
    best_competitors = np.array([], dtype = np.int32)
    
    competitors_speeds = np.array([])
    
    for c in competitors:
        
        competitors_speeds = np.append(competitors_speeds, agents[c].speed)
        
    max_speed = np.amax(competitors_speeds)
    
    for c in competitors:
        
        if agents[c].speed == max_speed:
            
            best_competitors = np.append(best_competitors, c)
            
    return best_competitors

def compare_strongest(agents, competitors):

    best_competitors = np.array([], dtype = np.int32)
    
    competitors_powers = np.array([])
    
    for c in competitors:
        
        competitors_powers = np.append(competitors_powers, agents[c].power)
        
    max_power = np.amax(competitors_powers)
    
    for c in competitors:
        
        if agents[c].power == max_power:
            
            best_competitors = np.append(best_competitors, c)
            
    return best_competitors
        
def competition(agents, food, food_competitors, food_value):
    
    for f in range(len(food_competitors)):
        
        competitors = np.array([], dtype = np.int32)
        
        for i in range(len(food_competitors[f])):
            
            if food_competitors[f][i] == 1:
                
                competitors = np.append(competitors, i)
                
        #print(competitors)
                
        for c in competitors:
            
            agents[c].energy -= agents[c].spending
          
        if len(competitors) > 0:    
            competitors = calculate_energy(agents, competitors, food, f)
            
        if len(competitors) > 0:
            competitors = compare_fastest(agents, competitors)
            
        if len(competitors) > 0:
            competitors = compare_strongest(agents, competitors)
        
        if len(competitors) > 0:
        
            distance = calculate_distance(agents[competitors[0]].pos, food[f])
            
            agents[competitors[0]].energy += food_value
            agents[competitors[0]].energy -= (agents[competitors[0]].spending * distance) 
            
            agents[competitors[0]].pos = food[f]
        
    return agents
        
def eliminate(agents):

    best_agents = np.array([])

    for agent in agents:
        
        if agent.energy > 0 and agent.speed > 0 and agent.power > 0:
            
            best_agents = np.append(best_agents, agent)
            
    return best_agents

def reproduce(surface_size, agents, mutation_rate, reproduce_energy):
    
    new_agents = np.array([])
    
    # create new agents
    for agent in agents:
        
        if agent.energy > reproduce_energy*2:
            
            while agent.energy > reproduce_energy*2:
            
                agent.energy -= reproduce_energy
                
                pos = np.random.rand(2)
                power = agent.power + (mutation_rate * (np.random.rand() - 0.5))
                speed = agent.speed + (mutation_rate * (np.random.rand() - 0.5))
                energy = reproduce_energy
                
                new_agents = np.append(new_agents, 
                                       Agent(pos, power, speed, energy))
                
            new_agents = np.append(new_agents, agent)
            
        else:
            
            new_agents = np.append(new_agents, agent)

    # set new positions
    new_positions = np.random.rand(len(new_agents), 2) * surface_size
    
    for i in range(len(new_agents)):
        
        new_agents[i].set_pos(new_positions[i])

    return new_agents

def get_bar_height(prop_list):
    
    resolution = np.arange(int(np.amax(prop_list) + 1), dtype = np.int32)
    
    bar_height = np.zeros(int(np.amax(prop_list) + 1))
    
    for i in resolution:
        
        bar_height[i] = np.count_nonzero(prop_list == i)

    return bar_height, resolution

def get_properties_list(agents):
    
    power_list = np.array([])
    speed_list = np.array([])
    energy_list = np.array([])
    
    for agent in agents:
        
        power_list = np.append(power_list, agent.power)
        speed_list = np.append(speed_list, agent.speed)
        energy_list = np.append(energy_list, agent.energy)
        
    return power_list, speed_list, energy_list
    
def graph_property_distribution(agents, n_agents_list):
    
    # collect data
    power_list, speed_list, energy_list = get_properties_list(agents)
        
    power_bar_height, power_resolution = get_bar_height(power_list)
    speed_bar_height, speed_resolution = get_bar_height(speed_list)
    energy_bar_height, energy_resolution = get_bar_height(energy_list)
    
    # Graphs
    plt.figure(2, figsize = (7,7))
    
    plt.subplot(221)
    plt.title('Power')
    plt.bar(power_resolution, power_bar_height)
    
    plt.subplot(222)
    plt.title('speed')
    plt.bar(speed_resolution, speed_bar_height)
    
    plt.subplot(223)
    plt.title('Agents by properties')
    plt.scatter(power_list, speed_list, alpha = 0.1)
    plt.xlabel('Power')
    plt.ylabel('Speed')
    
    plt.subplot(224)
    plt.title('Population')
    plt.plot(np.arange(len(n_agents_list)), n_agents_list)
    
    plt.show()

def make_animation(fig, frames):
    
    # create animation
    ani = animation.ArtistAnimation(fig, 
                                    frames, 
                                    interval = 50, 
                                    blit = True, 
                                    repeat_delay = 400)
    
    ani.save('agents_by_speed_power.mp4') # save animation

# variables
epochs = 1000
n_agents = 100

power_mean = 5
power_sd = 1

speed_mean = 5
speed_sd = 1

energy_mean = 500
energy_sd = 0

mutation_rate = 1

reproduce_energy = 500

n_food = 100
food_value = 100

surface_size = 25 # environment size


# animation
fig = plt.figure()
ims = []

# create population
agents = create_population(surface_size, n_agents, power_mean, power_sd, 
                                                   speed_mean, speed_sd,
                                                   energy_mean, energy_sd)  
n_agents_list = np.array([n_agents])

for i in tqdm(range(epochs)):
    
    # create food
    food = create_food(surface_size, n_food) # food position
    food_competitors = calculate_agents_nearest_food(surface_size, 
                                                     agents, food) # calculate nearest food 
    # food competition
    agents = competition(agents, food, food_competitors, food_value)
    # eliminate
    agents = eliminate(agents)
    # reproduce
    agents = reproduce(surface_size, agents, mutation_rate, reproduce_energy)                
    
    # metrics
    n_agents_list = np.append(n_agents_list, len(agents))
    
    #animation
    power_list, speed_list, _ = get_properties_list(agents)
    plt.title('Agents by properties')
    plt.xlabel('Power')
    plt.ylabel('Speed')
    im = plt.scatter(power_list, speed_list, alpha = 0.1)
    ims.append([im])
    
agents = eliminate(agents)

#metrics
n_agents_list = np.append(n_agents_list, len(agents))

if len(agents) > 0:  
    
    make_animation(fig, ims)
    graph_property_distribution(agents, n_agents_list)
    
else:
    
    plt.plot(np.arange(len(n_agents_list)), n_agents_list)
    
    plt.show()