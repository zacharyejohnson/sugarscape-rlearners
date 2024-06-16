
import copy
import random
from math import sqrt
import numpy as np
from scipy.stats.mstats import gmean
import torch
from Patch import *
import scipy.stats as stats
import numbers
import os
#import tensorflow as tf 
#import keras
from q_learning import DQN, ReplayMemory, Transition
from collections import deque, namedtuple
import copy 

#Agent.py

class Agent:
    # **inheritance are the inherited
    def __init__(self, model, row, col, ID, parent = None):

        self.period_consumption = 0
        self.period_income = 0 
        self.period_savings = 0 
        self.parent = parent 
        self.period_savings_by_good = {"sugar": 0, "water": 0}
        
        def selectParameters(mutate = False,
                             mutate_kwargs = {}):    
         
            # at first, you are the agent does not know any one else
            # give all agents these variables to avoid error when deleted from
            # inheritance dict
            def setReservationDemand():
                self.wealth_by_good = {}
                for good in self.model.goods: 
                    self.wealth_by_good[good] = (getattr(self,good) / self.model.consumption_rate[good])
                
                ### set rates of adjustment
                ## price_change defined in kwargs if mutate
                if self.parent == None:

                    # reservation ratio is the agents internal valuation of sugar; their internal valuation of water is its inverse 
                    min_res_ratio = self.model.init_res_ratio_ranges["min"]
                    max_res_ratio = self.model.init_res_ratio_ranges["max"]
                    
                    self.reservation_ratio = np.exp(np.log(min_res_ratio) + random.random() * (np.log(max_res_ratio) - np.log(min_res_ratio)))
                    self.reservation_ratio *= (self.model.consumption_rate["water"] / self.model.consumption_rate["sugar"])
                    min_price_change = .5
                    max_price_change = 2
                    self.price_change = np.e ** (np.log(min_price_change) + random.random() * (np.log(max_price_change) - np.log(min_price_change)))
           
                else:
                    if mutate_kwargs["reservation_ratio"]:
                        self.reservation_ratio = mutateAttrRate(getattr(parent, "reservation_ratio"))
                        # if random.getrandbits(1) == 1:
                        #             self.reservation_ratio = parent.reservation_ratio * (1 + self.mutate_rate)
                        # else:
                        #             self.reservation_ratio = parent.reservation_ratio / (1 + self.mutate_rate)

                    else: 
                        self.reservation_ratio = self.parent.reservation_ratio
                    if mutate_kwargs["price_change"]:
                        # if random.getrandbits(1) == 1:
                        #             self.price_change = parent.price_change * (1 + self.mutate_rate)
                        # else:
                        #             self.price_change = parent.price_change / (1 + self.mutate_rate)
                        self.price_change = mutateAttrRate(getattr(parent, "price_change"))
                    else: 
                        self.price_change = self.parent.price_change

                # if self.price_change > 1: 
                #     self.price_change = 1

            
            def setReproductionLevel():
                self.reproduction_criteria = {}
                self.reproduction_ratio = {}

                # first, reproduction ratio is set, which is the multiplier of ones own current endowment they must be able to provide their children
                min_reproduction_ratio = {}
                max_reproduction_ratio = {}
                if self.parent == None:                    
                        for good in self.model.goods: 
                            min_reproduction_ratio[good] = self.model.init_reproduction_ratio_ranges[good]["min"] 
                            max_reproduction_ratio[good] = self.model.init_reproduction_ratio_ranges[good]["max"]  
                            self.reproduction_ratio[good] = get_lognormal_rand(min_reproduction_ratio[good], max_reproduction_ratio[good])
                else: 
                        if mutate_kwargs["reproduction_ratio"]: 
                            for good in self.model.goods: 
                                 mutation = np.log(parent.reproduction_ratio[good] * self.mutate_rate)
                                 if random.getrandbits(1) == 1:
                                    self.reproduction_ratio[good] = np.exp(np.log(parent.reproduction_ratio[good]) - mutation)
                                 else:
                                    self.reproduction_ratio[good] = np.exp(np.log(parent.reproduction_ratio[good]) + mutation)

                                #  max_val = parent.reproduction_ratio[good] * (1 + self.mutate_rate)

                                #  self.reproduction_ratio[good] = min_val + random.random() * (max_val - min_val)
                                 if self.reproduction_ratio[good] > 10: 
                                       self.reproduction_ratio[good] = 10

                    
                        else: 
                            for good in self.model.goods: 
                                self.reproduction_ratio[good] = parent.reproduction_ratio[good]

                #set reproduction criteria for each good to be ones initial endowment times some reproduction ratio
                for good in self.model.goods: 
                            self.reproduction_criteria[good] = getattr(self, good) * self.reproduction_ratio[good]

            def selectBreed():
                if self.parent != None: 
                    # place herder first in list
                    shuffle_breeds = copy.copy(self.model.primary_breeds)
                    random.shuffle(shuffle_breeds)
                    for breed_ in ["herder"] + shuffle_breeds:
                        if random.random() < self.mutate_rate:
                            # if mutation occurs, switch breed boolean
                            select_breed = False if getattr(self.parent, breed_) else True
                            setattr(self, breed_, select_breed)
                            
                            if select_breed == True and breed_ in shuffle_breeds:
                                shuffle_breeds.remove(breed_)
                                for not_my_breed in shuffle_breeds:
                                    setattr(self, not_my_breed, False)
                                break
                    # set breed basic if all breeds are turned to False
                    if True not in (getattr(self, brd)
                                    for brd in self.model.primary_breeds):
                        
                            self.setBreedBasic(herder = self.herder)

                # select breed randomly if agent has no parent            
                else:                            
                    self.setBreedBasic(herder = False)

           
                    
                self.selectBreedParameters(mutate, self.parent, 
                                           herding = False)

            def setHerdingStrategy(): 
                if self.model.mutate: 
                    if self.parent == None: 
                        self.num_alive_children = 0
                        # first generation of agents will have half and half 
                        if random.getrandbits(1) == 1: 
                            self.herding_metric = "wealth"
                        else: 
                            self.herding_metric = "num_alive_children"

                        self.top_wealth = getattr(self, self.herding_metric)
                        self.wealthiest = self
                    else: 
                        self.num_alive_children = 0
                        if mutate_kwargs["herding_metric"]: 
                            self.herding_metric = "wealth" if self.parent.herding_metric == "num_alive_children" else "num_alive_children"
                        else: 
                            self.herding_metric = self.parent.herding_metric

                        



            def setMutateRate():
                 if self.model.mutate:
                    if self.parent == None:
                        #min_rate = 0.01
                        #max_rate = self.model.max_mutate_rate

                        #draw mutate rates from reciprocal probability dist 
                        
                        x = np.random.uniform(2,50)
                        self.mutate_rate = 1 / x
                        #self.mutate_rate = min_rate + random.random() * (max_rate - min_rate)
                    elif mutate_kwargs["mutate_rate"]: 
                            if random.getrandbits(1) == 1: 
                                self.mutate_rate = np.exp(np.log(self.parent.mutate_rate) - np.log(self.parent.mutate_rate * self.parent.mutate_rate))
                            else: 
                                self.mutate_rate = np.exp(np.log(self.parent.mutate_rate) + np.log(self.parent.mutate_rate * self.parent.mutate_rate))
                            #self.mutate_rate = min_rate + random.random() * (max_rate - min_rate)
                    else: 
                            self.mutate_rate = self.parent.mutate_rate
                        # keep a hard limit on the height of mutation rate
                    if self.mutate_rate >= self.model.max_mutate_rate:
                            self.mutate_rate = self.model.max_mutate_rate

            def init_nn():
                n_input = 6  # Number of inputs in the agent's state
                if self.parent == None:
                    discount_rate = min(0.999, random.normalvariate(mu=0.95, sigma=.01))
                    learning_rate = max(0.0001, random.normalvariate(mu=0.2, sigma=0.075))
                    self.min_res_ratio = 0.2
                    self.max_res_ratio = 5
                    self.num_bins = 100
                    self.replay_memory_length = random.randint(1, 100)
                    self.n = random.randint(1, 10)
                    self.n_layers = random.randint(1, 5)  # Genetically determine the number of layers
                    self.layer_size = random.randint(32, 128)  # Genetically determine the layer size
                    self.nn = DQN(n_input, self.num_bins, discount_rate=discount_rate, learning_rate=learning_rate, n_layers=self.n_layers, layer_size=self.layer_size)
                else:
                    discount_rate = mutateAttr(self.parent.nn.discount_rate)
                    learning_rate = mutateAttrRate(self.parent.nn.learning_rate)
                    self.min_res_ratio = mutateAttr(self.parent.min_res_ratio)
                    self.max_res_ratio = mutateAttr(self.parent.max_res_ratio)
                    self.num_bins = 100  # mutateAttr(self.parent.num_bins, integer=True)
                    self.replay_memory_length = mutateAttr(self.parent.replay_memory_length, integer=True)
                    self.n = mutateAttr(self.parent.n, True)
                    self.n_layers = mutateAttr(self.parent.n_layers, integer=True)
                    self.layer_size = mutateAttr(self.parent.layer_size, integer=True)
                    if self.replay_memory_length > 100:
                        self.replay_memory_length = 100  # cap this
                    self.nn = DQN(n_input, self.num_bins, discount_rate=discount_rate if discount_rate < .999 else .999, learning_rate=learning_rate, n_layers=self.n_layers, layer_size=self.layer_size)
                self.memory = ReplayMemory(10)

            # TODO change this to allow for mutation later
            # def select_rlearner_params():
            #      if self.parent == None: 
            #           self.gamma = 1-get_lognormal_rand(0.1, 1)
            #           self.epsilon = get_lognormal_rand(0.01, 0.1)
            #           self.learning_rate = get_lognormal_rand(0.01, 0.5)
            #           self.max_memory_size = 1000
                      

            #      else: 
            #           self.gamma = mutateAttrRate(getattr(parent, "gamma"))
            #           self.epsilon = mutateAttrRate(getattr(parent,"epsilon"))
            #           self.learning_rate = mutateAttrRate(getattr(parent,"learning_rate"))



            def get_lognormal_rand(min, max): 
                 return np.exp(np.log(min) + random.random() * (np.log(max) - np.log(min)))
                 

            ###################################################################            

            # define mutate rate first so that it effects mutation of all
            # other attributes
            setMutateRate() 
            setStocks()
            selectBreed()
            setHerdingStrategy()

            # set value of commodity holdings, if agent has parents,
            # these values will be replaced by the max values
            
            #if reservation_demand: 
            setReservationDemand()
            
            # select_rlearner_params()
            setReproductionLevel()  
            init_nn()
               
            setTargets()
            self.vision = random.randint(1, self.model.max_vision)
            
        #######################################################################

        def setStocks():

            self.wealth_by_good = {"sugar": 0, "water": 0}
    
            if self.parent == None:
                for good, vals in self.model.goods_params.items():
                    val = random.randint(vals["min"], vals["max"])
                    setattr(self, good, val)

            else:
                for good in self.model.goods:
                    setattr(self, good, self.parent.reproduction_criteria[good] / 2)
                    setattr(self.parent, good, 
                            getattr(self.parent, good) - self.parent.reproduction_criteria[good] / 2)
                self.parent.update_stocks()


            self.initial_goods = {}
            for good in self.model.goods: 
                self.initial_goods[good] = getattr(self, good)
                self.wealth_by_good[good] = getattr(self, good) / self.model.consumption_rate[good]
            
            self.wealth = np.sum(list(self.wealth_by_good.values()))
            
        def mutateAttr(attr, integer=False):
            min_val = attr / (1 + self.mutate_rate)
            max_val = attr * (1 + self.mutate_rate)
            if integer:
                attr_val = random.randint(int(min_val), int(max_val))
            else:
                attr_val = random.uniform(min_val, max_val)
            return attr_val
        
        def mutateAttrRate(attr): 
           
            # mutation must be symmetric around zero change such that the attribute that the parameter adjusts expectation has not changed
            if random.getrandbits(1) == 1: 
                return np.exp(np.log(attr) - (np.log(attr) * self.mutate_rate))
            else: 
                return np.exp(np.log(attr) + (np.log(attr) * self.mutate_rate))

        def setTargets():
            # set exchange target randomly at first
            goods = list(self.model.goods)
            random.shuffle(goods)
            self.target = goods.pop()
            self.not_target = goods[0]
            
        def mutate():
            # select which parameters will be mutated
            mutate_dict = {key: True if random.random() < self.mutate_rate else False for key, val in inheritance.items()} 
            # mutate select parameters
            selectParameters(mutate = True, mutate_kwargs = mutate_dict)
            
        # if parent != None:
        #     inheritance = parent.defineInheritance()
        self.model = model
        
        if self.parent != None:
            inheritance = self.parent.defineInheritance()
            ####### parameters already inherited if agent has parent ########
            for attr, val in inheritance.items():
                 setattr(self, attr, val)
            #setStocks()
            # randomly set target, will be redifined in according to breed
            # parameters in the following period
            #setTargets()
            # inherited values are mutated vals in dictionary if mutation is on
            if self.model.mutate:
                mutate()    
            else:
                self.selectBreedParameters(mutate = False,
                                           parent = self.parent,
                                           herding  = False)
        
        else:
            selectParameters()
        # allocate each .good to agent within quantity in range specified by 
        # randomly choose initial target good
        self.col = col
        self.row = row 
        self.dx = 0
        self.dy = 0
        self.id = ID
        self.num_alive_children = 0



###############################################################################     
    def setBreedBasic(self, herder):
        if "basic" in self.model.primary_breeds: 
            self.basic = True
            self.optimizer = False 
            self.rlearner = False
        elif "optimizer" in self.model.primary_breeds and "rlearner" not in self.model.primary_breeds: 
            self.optimizer = True 
            self.basic = False
            self.rlearner = False
        elif "rlearner" in self.model.primary_breeds: 
            self.optimizer = False 
            self.basic = False
            self.rlearner = True
        else: 
            self.optimizer = False 
            self.basic = False
            self.rlearner = True

        self.arbitrageur = False
        self.herder = herder
        
    def mutateAttr(self, attr, rate): 
           
            min_val = getattr(self, attr) / (1 + rate)
            max_val = getattr(self, attr) * (1 + rate)
            attr_val = min_val + random.random() * (max_val - min_val)

            return attr_val
   
        

    def selectBreedParameters(self, mutate, parent, herding, 
                              partner = None):
        #inheritance = parent.defineInheritance() if parent is not None else ""
        def generateBreedParameters():
            if breed  == "herder":      

                self.wealthiest = self.parent if inheritance is not None else self
                self.top_wealth = getattr(self.parent, self.herding_metric) if inheritance is not None else getattr(self, self.herding_metric)

        
            self.MRS = (self.wealth_by_good["water"] / self.model.consumption_rate["water"]) / (self.wealth_by_good["sugar"] / self.model.consumption_rate["sugar"])
            
            if breed == "arbitrageur": 
                 
                self.reservation_ratio = 1
        
        def copyPartnerParameters():
            # if copied breed and missing parameter value, draw from partner
            
                if breed  == "herder":  
                        self.top_wealth = getattr(partner, self.herding_metric)
                        self.wealthiest = partner

                    
                self.MRS = (self.wealth_by_good["water"] / self.model.consumption_rate["water"]) / (self.wealth_by_good["sugar"] / self.model.consumption_rate["sugar"])
                    
                    #self.reservation_ratio = partner.reservation_ratio

                if breed == "arbitrageur": 
                     self.reservation_ratio = 1 
                     if not hasattr(self, "present_price_weight"):                    
                        self.present_price_weight = partner.present_price_weight 
                     
                if breed == "basic": 
                    if getattr(partner, "optimizer"): 
                        # basics do not want to copy optimizer reservation ratios because they are not used and selected for by optimizers.
                        # the situation where a herding agent switches only res rati and not breed is where this is an issue, but is a relatively rare problem. 
                        # when this happens, revert basic parameters back to their parents. 
                        if self.reservation_ratio == partner.reservation_ratio: 
                            self.reservation_ratio = self.parent.reservation_ratio
                        if self.price_change == partner.price_change:
                            self.prce_change = self.parent.price_change


        for breed in self.model.breeds:
            if getattr(self, breed):
                inheritance = self.parent.defineInheritance() if self.parent != None else None
                # those who change breed due to herding need only need to fill missing
                # parameter values
                if herding:
                    copyPartnerParameters()
                else:
                    generateBreedParameters()        

    def defineInheritance(self):
        # use attributes to define inheritance
        copy_attributes = copy.copy(vars(self))
        # redefine "good" or else values are drawn from parent for children

        for key in self.model.drop_attr:
            try:
                del copy_attributes[key]
            except:
                continue 

        # copy_attributes = list(copy_attributes.items())

        # random.shuffle(copy_attributes)

        # copy_attributes = dict(copy_attributes)

        return copy_attributes
    
    def updateParams(self, trade = False):

        self.update_stocks()
        if self.herder and not trade:
            self.top_wealth = getattr(self.wealthiest, self.herding_metric)
            if getattr(self, self.herding_metric) > self.top_wealth:
                    self.wealthiest = self
                    self.top_wealth = getattr(self, self.herding_metric)

            # the herder only copies strategies from alive agents - if a firm fails, they will not be copied 
            if self.wealthiest not in self.model.agent_dict.values(): 
                 self.wealthiest = self
                 self.top_wealth = getattr(self, self.herding_metric)

        for good in self.model.goods: 
                self.reproduction_criteria[good] = self.reproduction_ratio[good] * self.initial_goods[good]


        def checkReservation():
                 
            reproduction_ratio = (self.reproduction_criteria["water"] / self.model.consumption_rate["water"]) / (self.reproduction_criteria["sugar"] / self.model.consumption_rate["sugar"])
            adjustment = np.log(reproduction_ratio / ((self.water / self.model.consumption_rate["water"]) / (self.sugar / self.model.consumption_rate["sugar"])))

            # max_rr = self.reservation_ratio * np.exp(adjustment)
            # change = self.price_change * (max_rr - self.reservation_ratio)

            # self.reservation_ratio += change

            self.reservation_ratio *= (self.price_change * np.exp(adjustment))

            if self.reservation_ratio > 1000: 
                        self.reservation_ratio = 1000
            if self.reservation_ratio < 0.001: 
                        self.reservation_ratio = 0.001

        
        
   

        def checkMRS(): 
            # optimizers choose internal valuations of each good based on their relative scarcity
            
            if self.wealth_by_good["water"] > 0 and self.wealth_by_good["sugar"] > 0: 
                self.MRS = (self.wealth_by_good["water"] / self.model.consumption_rate["water"]) / (self.wealth_by_good["sugar"] / self.model.consumption_rate["sugar"])
            else: 
                if trade: 
                    print(self.wealth_by_good["water"])
                    print("MRS: " + str(self.MRS))
                self.MRS = float("-inf")
        
        checkMRS()


        def checkQValuation():
            state = self.get_state()
            action = self.choose_action(state)
            reservation_ratio_bins = np.logspace(np.log10(self.min_res_ratio), np.log10(self.max_res_ratio), num=self.num_bins)
            if action >= len(reservation_ratio_bins):
                action = len(reservation_ratio_bins) - 1
            self.reservation_ratio = reservation_ratio_bins[action]
                        

        if self.water > 0 and self.sugar > 0:
            if not self.rlearner: 
                checkReservation()
            elif not trade: 
                checkQValuation()
                if self.reservation_ratio > 1000: 
                        self.reservation_ratio = 1000
                if self.reservation_ratio < 0.001: 
                        self.reservation_ratio = 0.001


            #print(self.reservation_ratio)

    ############### RL Learner Functions ###############################
    def remember(self, transition):
        self.memory.push(*transition)

    def choose_action(self, state):
        normalized_state = self.get_normalized_state(state)
        state_tensor = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.nn(state_tensor)
        action = torch.argmax(q_values).item()
        return action

    def learn(self, transition):
        state, action, reward, next_state = transition
        self.nn.update(state, action, reward, next_state, self.nn.discount_rate)

        # Ensure that state and next_state are in the correct shape for a batch of size 1
        self.nn.sarsa_update(state, reward, next_state)

    def replay(self, batch_size):
        if len(self.memory.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array([self.get_normalized_state(state) for state in batch.state]), dtype=torch.float32)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array([self.get_normalized_state(next_state) for next_state in batch.next_state]), dtype=torch.float32)

        for _ in range(self.n):
            self.nn.update(state_batch, action_batch, reward_batch, next_state_batch, self.nn.discount_rate, self.n)

    def get_state(self):
        norm_row = (self.row - (self.model.rows / 2)) / (self.model.rows)
        norm_col = (self.col - (self.model.cols / 2)) / (self.model.cols)
        movement_embedding = [self.dx, self.dy]  # Add movement embedding
        return np.array([norm_row,
                        norm_col,
                        self.wealth_by_good["water"] / self.reproduction_criteria["water"],
                        self.wealth_by_good["sugar"] / self.reproduction_criteria["sugar"]] + movement_embedding)
            
         
    def get_normalized_state(self, state):
        stored_states = self.memory.get_all_states()

        if len(stored_states) > 0:
            min_values = np.min(stored_states, axis=0)
            max_values = np.max(stored_states, axis=0)
            normalized_state = (state - min_values) / (max_values - min_values + 1e-8)
        else:
            normalized_state = state

        return normalized_state

    def weighted_rr(self, neurons):
        return np.dot(neurons, np.array([self.dx, self.dy, self.wealth_by_good["water"], self.wealth_by_good["sugar"]]))


    def save_nn_params(self, period):
        params = {
            "weights": self.nn.fc.weight.detach().numpy(),
            "bias": self.nn.fc.bias.detach().numpy(),
            "discount_rate": self.nn.discount_rate,
            "learning_rate": self.nn.learning_rate
        }
        if f"agent_{self.id}_nn_params" not in self.model.data_dict:
            self.model.data_dict[f"agent_{self.id}_nn_params"] = {}
        self.model.data_dict[f"agent_{self.id}_nn_params"][period] = params

    def consume(self):
        self.period_consumption = 0
        i = 0
        for good, rate in self.model.consumption_rate.items():
            curr_level = getattr(self, good)
            if curr_level < self.model.consumption_rate[good]: 
                setattr(self, good, curr_level - curr_level)
                if curr_level > 0:
                    self.period_consumption += curr_level
            elif curr_level > 0: 
                setattr(self, good, curr_level - rate)
                self.period_consumption += rate
        self.update_stocks()


    def update_stocks(self): 
        for good in self.model.goods: 
                self.wealth_by_good[good] = (getattr(self,good) / self.model.consumption_rate[good])
                self.reproduction_criteria[good] = self.reproduction_ratio[good] * self.initial_goods[good]

        self.wealth = sum(list(self.wealth_by_good.values()))
            
    
    def check_alive(self):
        
        for good in self.model.goods:
            if getattr(self, good) <= 0:
                agent_patch = self.model.patches_dict[self.row][self.col]
                # self.model.dead_agent_dict[self.id] = self
                self.model.empty_patches[self.row, self.col] = agent_patch
                if self.model.live_visual:
                    self.model.GUI.canvas.delete(self.image)

                # agent drops goods on patch when they die 
                if getattr(self, agent_patch.good) > 0: 
                    agent_patch.Q += getattr(self, agent_patch.good)
                     
                del self.model.agent_dict[self.id]

                if self.parent != None: 

                    self.parent.num_alive_children -= 1

                break

    def calculate_market_cap(self): 
         return sum([agent.wealth for agent in self.model.agent_dict.values() if agent.parent == self]) + self.wealth
                          
                 
    def reproduce(self):

        # if getattr(self, "sugar") > self.reproduction_criteria["sugar"] and\
        #     getattr(self, "water") > self.reproduction_criteria["water"]:
        if getattr(self, "sugar") > self.reproduction_criteria["sugar"] and\
            getattr(self, "water") > self.reproduction_criteria["water"]:


            neighbor_patches = [(self.row + i, self.col + j)
                        for i in self.model.nav_dict[1] if 0 <= self.row + i < 50
                        for j in self.model.nav_dict[1][i] if 0 <= self.col + j < 50 ]            
            empty_neighbor_patches = [patch for patch in neighbor_patches if patch in self.model.empty_patches]
  
            if len(empty_neighbor_patches) > 0: #situation where map is full

                reproduction_patch = random.choice(empty_neighbor_patches)

                self.model.total_agents_created += 1
                new_patch = self.model.empty_patches[reproduction_patch]
                row, col = new_patch.row, new_patch.col
                del self.model.empty_patches[reproduction_patch]
                ID = self.model.total_agents_created
                self.model.agent_dict[ID] =  Agent(self.model, row=row, col=col, 
                                                ID=ID, parent = self)
                
                self.update_stocks()

                self.num_alive_children += 1
                
                self.model.agent_dict[ID].top_wealth = getattr(self, self.model.agent_dict[ID].herding_metric)
                self.model.agent_dict[ID].wealthiest = self



                self.model.patches_dict[row][col].agent =  self.model.agent_dict[ID]
                if self.model.live_visual:
                    self.model.GUI.drawAgent(self.model.agent_dict[ID])

                
    def getPatchUtility(self, patch): 

        if self.basic or self.arbitrageur or self.rlearner: 
            if patch.good == "sugar": 
                new_utility = (patch.Q / self.model.consumption_rate["sugar"]) * (1/self.reservation_ratio)
            elif patch.good == "water": 
                new_utility = (patch.Q / self.model.consumption_rate["water"]) * self.reservation_ratio
            else: 
                new_utility = 0 
            
            
        elif self.optimizer: 
            if patch.good == "sugar": 
                new_utility = (self.wealth_by_good["sugar"] + (patch.Q / self.model.consumption_rate["sugar"])) ** self.model.consumption_rate["sugar"]\
                    * (self.wealth_by_good["water"]) ** self.model.consumption_rate["water"]
            elif patch.good == "water": 
                new_utility = (self.wealth_by_good["sugar"]) ** self.model.consumption_rate["sugar"]\
                    * (self.wealth_by_good["water"] + (patch.Q / self.model.consumption_rate["water"])) ** self.model.consumption_rate["water"]
            else: 
                #print(new_utility)
                new_utility = 0 
      

        return new_utility



######################## move method and functions ############################
    def move(self):  
        def findMaxEmptyPatch(curr_row, curr_col):
            # dict to save empty patch with max q for each good for basics, utility gain for optimizers 

            max_patch = {good:{"U":0,
                                "patch":None}
                            for good in self.model.goods}

            
            patch_moves = [(curr_row + dy, curr_col + dx)  
                           for dy in self.model.nav_dict[self.vision] if 0 <= curr_row + dy < 50
                           for dx in self.model.nav_dict[self.vision][dy] if 0 <= curr_col + dx < 50]
            
            # shuffle patches so not movement biased in one direction
            random.shuffle(patch_moves)
            near_empty_patch = False #{good: False for good in self.good}
            empty_patches = []
            for coords in patch_moves:
                if coords in self.model.empty_patches.keys:
                    row, col = coords[0], coords[1]
                    empty_patch = self.model.patches_dict[row][col]
                    empty_patches.append(empty_patch)
                    patch_q = self.getPatchUtility(empty_patch)
                    patch_good = empty_patch.good
                    #try: 
                    if patch_q > max_patch[patch_good]["U"]:
                            # only mark near empty patch if Q > 0
                            near_empty_patch = True
                            max_patch[patch_good]["patch"] = empty_patch
                            max_patch[patch_good]["U"] = patch_q
                    #except: 
                        
                     #   print("movement error:" +  getattr(self, "sugar"), getattr(self, "water"))
            return max_patch, near_empty_patch, empty_patches    

        def moveToMaxEmptyPatch(curr_row, curr_col, 
                                max_patch, near_empty_patch,
                                target, not_target, empty_patches):
            
            def basicMove(max_patch):
                max_q = max(max_patch[good]["U"] for good in max_patch )
                # include both max water and max sugar patch if moth have max_q
                max_patches = [good for good in max_patch if max_patch[good]["U"] == max_q]
                #randomly select max water or max sugar patch
                max_good = random.choice(max_patches) 
                target_patch = max_patch[max_good]["patch"]
                return target_patch
            
            def chooseTargetOrAlternate(max_patch, target, not_target, empty_patches):
                if type(max_patch[target]["patch"]) is Patch:
                    target_patch = max_patch[target]["patch"]
                    return target_patch
                # use elif with return within the if statement, that way
                # an error is thrown if target == not_target
                elif type(max_patch[not_target]["patch"]) is Patch:
                    # choose patch that moves agent closest to target 
                    # commodity
                    max_val = float("-inf")
                    min_val = float("inf")
                    for patch in empty_patches:
                        coord_sum = patch.col + patch.row 
                        if target == "sugar":
                            if coord_sum < min_val:
                                max_val = coord_sum
                                target_patch = patch
                        elif target == "water":
                            if coord_sum > max_val:
                                min_val = coord_sum
                                target_patch = patch
                                                
                    return target_patch
            
            ###################################################################  
            
    
            if near_empty_patch:
                if self.basic or self.optimizer or self.arbitrageur or self.rlearner:
                    target_patch = basicMove(max_patch)
                else:
                    target_patch = chooseTargetOrAlternate(max_patch, target, not_target, empty_patches)
                # track relative position to move image
                if target_patch is not None: 
                    self.dx, self.dy = target_patch.col - curr_col, target_patch.row - curr_row
                # set new coordinates
                    self.row, self.col =  target_patch.row, target_patch.col 
                # register agent to patch
                    self.model.patches_dict[self.row][self.col].agent = self
                # set agent at old patch to none
                    self.model.patches_dict[curr_row][curr_col].agent = None
                # register old patch to empty_patches
                    self.model.empty_patches[curr_row, curr_col] = self.model.patches_dict[curr_row][curr_col]
                # remove agent's current position from emtpy_patches
                    del self.model.empty_patches[self.row, self.col]
            else:
                self.dx = 0
                self.dy = 0
    ###############################################################################

        # save agent coords to track agent movement, changes in (not) empty patches
        curr_row, curr_col = self.row, self.col
        max_patch, near_empty_patch, empty_patches = findMaxEmptyPatch(curr_row, curr_col)
        random.shuffle(empty_patches)
        
        # if near_empty_patch:
        moveToMaxEmptyPatch(curr_row, curr_col, max_patch, 
             near_empty_patch, self.target, self.not_target, empty_patches)


    
    def harvest(self):    
        agent_patch = self.model.patches_dict[self.row][self.col]
        setattr(self, agent_patch.good, getattr(self, agent_patch.good) + agent_patch.Q)

        agent_patch.Q = 0 
        self.update_stocks()
    
    def prospective_MRS(self, price): 
            if self.target == "water": 
                if price >= 1:
                    water = price / self.model.consumption_rate["water"]
                    sugar = 1 / self.model.consumption_rate["sugar"]
                    if self.wealth_by_good["sugar"] - sugar> 0: 
                        return (self.wealth_by_good["water"] + water) / (self.wealth_by_good["sugar"] - sugar)
                    else: 
                        return float("inf")
                elif price < 1: 
                    water = 1 / self.model.consumption_rate["water"]
                    sugar = (1/price) / self.model.consumption_rate["sugar"]
                    if self.wealth_by_good["sugar"] - sugar > 0: 
                        return (self.wealth_by_good["water"] + water) / (self.wealth_by_good["sugar"] - sugar)
                    else: 
                        return float("inf")
                else: 
                    return float("inf")
            else: 
                if price >= 1: 
                    water = price / self.model.consumption_rate["water"]
                    sugar = 1 / self.model.consumption_rate["sugar"]
                    if self.wealth_by_good["water"] - water > 0: 
                        return (self.wealth_by_good["water"] - water) / (self.wealth_by_good["sugar"] + sugar)
                    else: 
                        return float("-inf")
                        
                elif price < 1: 
                    water = 1 / self.model.consumption_rate["water"]
                    sugar = (1/price) / self.model.consumption_rate["sugar"]
                    if self.wealth_by_good["water"] - water > 0: 
                        return (self.wealth_by_good["water"] - water) / (self.wealth_by_good["sugar"] + sugar)
                    else: 
                        return float("-inf")
                else: 
                    return float("-inf")
                
        # elif self.target == "water": 
        #     return float("inf")
        # else: 
        #     return float("-inf")
    
    def transfer(self, partner, price, num_trades):
            if self.target == "water": 
                # self gets water, partner gets sugar 
                if price >= 1: 
                    water = price * num_trades
                    sugar = num_trades
                    setattr(self, self.target, (getattr(self, self.target) + water) )
                    setattr(self, self.not_target, (getattr(self, self.not_target) - sugar) )
                    setattr(partner, self.target, (getattr(partner, self.target) - water))
                    setattr(partner, self.not_target, getattr(partner, self.not_target) + sugar)
                else: 
                    water = num_trades
                    sugar = (1/price) * num_trades
                    setattr(self, self.target, (getattr(self, self.target) + water)) 
                    setattr(self, self.not_target, (getattr(self, self.not_target) - sugar) )
                    setattr(partner, self.target, (getattr(partner, self.target) - water))
                    setattr(partner, self.not_target, getattr(partner, self.not_target) + sugar)
            else: 
                # self gets sugar partner gets water 
                if price >= 1: 
                    water = price * num_trades
                    sugar = num_trades
                    setattr(self, self.target, (getattr(self, self.target) + sugar) )
                    setattr(self, self.not_target, (getattr(self, self.not_target) - water) )
                    setattr(partner, self.target, (getattr(partner, self.target) - sugar ))
                    setattr(partner, self.not_target, getattr(partner, self.not_target) + water)
                else: 
                    water = num_trades
                    sugar = (1/price) * num_trades
                    setattr(self, self.target, (getattr(self, self.target) + sugar) )
                    setattr(self, self.not_target, (getattr(self, self.not_target) - water ))
                    setattr(partner, self.target, (getattr(partner, self.target) - sugar ))
                    setattr(partner, self.not_target, getattr(partner, self.not_target) + water)
        
    def trade(self):
        def still_can_trade(price): 
            #if price > 0.01 and price < 100: 
                # helper function for executeTrade; used in while loop to continue trading
                # basic agents need to 1. Not corssover their reservation ratio and 2. be able to afford the trade 
                if self.basic or self.arbitrageur or self.rlearner: 
                    if self.target == "water":
                        self_not_crossover = self.prospective_MRS(price) < self.reservation_ratio
                        partner_not_crossover = partner.prospective_MRS(price) > partner.reservation_ratio
                        if partner.optimizer: 
                            no_crossover = self_not_crossover and partner.prospective_MRS(price) > self.reservation_ratio
                        else: 
                            no_crossover = self_not_crossover and partner_not_crossover
                        if price >= 1: 
                            valid_trade = getattr(self, "sugar") > price and getattr(partner, "water") > 1 and no_crossover
                        else: 
                            valid_trade = getattr(self, "sugar") > 1 and getattr(partner, "water") > 1/price  and no_crossover
                    else: 
                        self_not_crossover = self.prospective_MRS(price) > self.reservation_ratio
                        partner_not_crossover = partner.prospective_MRS(price) < partner.reservation_ratio
                        if partner.optimizer: 
                            no_crossover = self_not_crossover and partner.prospective_MRS(price) < self.reservation_ratio
                        else: 
                            no_crossover = self_not_crossover and partner_not_crossover
                        if price >= 1: 
                            valid_trade = getattr(partner, "sugar") > price and getattr(self, "water") >  1 and no_crossover
                        else: 
                            valid_trade = getattr(partner, "sugar")  > 1 and getattr(self, "water") >  1/price and no_crossover
                elif self.optimizer: 
                    if self.target == "water":
                        self_not_crossover = self.prospective_MRS(price) < partner.prospective_MRS(price)
                        if partner.basic or partner.arbitrageur: 
                            partner_not_crossover = partner.prospective_MRS(price) > partner.reservation_ratio
                            self_not_crossover = self.prospective_MRS(price) < partner.reservation_ratio
                        else: 
                            partner_not_crossover = partner.prospective_MRS(price) > self.prospective_MRS(price)
                        no_crossover = partner_not_crossover and self_not_crossover
                        if price >= 1: 
                            valid_trade = getattr(self, "sugar") > price and getattr(partner, "water") > 1 and no_crossover
                        else: 
                            valid_trade = getattr(self, "sugar") > 1 and getattr(partner, "water") > 1/price and no_crossover
                    else: 
                        self_not_crossover = self.prospective_MRS(price) > partner.prospective_MRS(price)
                        if partner.basic or partner.arbitrageur: 
                            partner_not_crossover = partner.prospective_MRS(price) < partner.reservation_ratio
                            self_not_crossover = self.prospective_MRS(price) > partner.reservation_ratio
                        else: 
                            partner_not_crossover = partner.prospective_MRS(price) < self.prospective_MRS(price)
                        no_crossover = partner_not_crossover and self_not_crossover
                        if price >= 1: 
                            valid_trade = getattr(partner, "sugar") > price and getattr(self, "water") >  1 and no_crossover
                        else: 
                            valid_trade = getattr(partner, "sugar")  > 1 and getattr(self, "water") >  1/price  and no_crossover

                return valid_trade
            # else: 
            #         return False

        
        def askToTrade(patch):
            partner = patch.agent
            #check if partner is looking for good agent is selling
            right_good = True
                

            return partner, right_good

        def bargain(partner):
            can_trade = False
            price = None
            if not self.optimizer and not partner.optimizer:   
                positive_MRSs = self.MRS > 0 and partner.MRS > 0 
                if self.reservation_ratio < partner.reservation_ratio and positive_MRSs: 
                        self.target = "sugar"
                        self.not_target = "water"
                        partner.target = "water"
                        partner.not_target = "sugar"
                        price = sqrt(self.reservation_ratio * partner.reservation_ratio)
                        if still_can_trade(price):
                                can_trade = True
                       
                    # partner values sugar more than self 
                elif self.reservation_ratio > partner.reservation_ratio and positive_MRSs: 
                        self.target = "water"
                        self.not_target = "sugar"
                        partner.target = "sugar"
                        partner.not_target = "water"
                        price = sqrt(self.reservation_ratio * partner.reservation_ratio)
                        if still_can_trade(price):
                                can_trade = True
                elif self.reservation_ratio == partner.reservation_ratio: 
                        can_trade = False
                        price = None
                else: 
                        price, can_trade = None, False 
                        #print(self.MRS, partner.MRS)
            elif partner.optimizer and self.optimizer: 
                    # partner values water more than self 
                    positive_MRSs = self.MRS > 0 and partner.MRS > 0
                    if self.MRS > partner.MRS and positive_MRSs: 
                        self.target = "sugar"
                        self.not_target = "water"
                        partner.target = "water"
                        partner.not_target = "sugar"
                        price = sqrt(self.MRS * partner.MRS)
                        if still_can_trade(price):
                                can_trade = True
                    # partner values sugar more than self 
                    elif self.MRS < partner.MRS and positive_MRSs: 
                        self.target = "water"
                        self.not_target = "sugar"
                        partner.target = "sugar"
                        partner.not_target = "water"
                        price = sqrt(self.MRS * partner.MRS)
                        if still_can_trade(price):
                                can_trade = True
                    elif self.MRS == partner.MRS: 
                        can_trade = False
                        price = None
                    else: 
                        price, can_trade = None, False 
                        #print(self.MRS, partner.MRS)
            elif self.optimizer and not partner.optimizer: 
                    positive_MRSs = self.MRS > 0 and partner.MRS > 0
                    if self.MRS > partner.reservation_ratio and positive_MRSs: 
                        self.target = "sugar"
                        self.not_target = "water"
                        partner.target = "water"
                        partner.not_target = "sugar"
                        price = sqrt(self.MRS * partner.reservation_ratio)
                        can_trade = still_can_trade(price)
                    elif self.MRS < partner.reservation_ratio and positive_MRSs: 
                        self.target = "water"
                        self.not_target = "sugar"
                        partner.target = "sugar"
                        partner.not_target = "water"
                        price = sqrt(self.MRS * partner.reservation_ratio)
                        can_trade = still_can_trade(price)
                    else: 
                        can_trade = False
                        price = None
            elif not self.optimizer and partner.optimizer: 
                    positive_MRSs = self.MRS > 0 and partner.MRS > 0
                    if partner.MRS > self.reservation_ratio and positive_MRSs: 
                        partner.target = "sugar"
                        partner.not_target = "water"
                        self.target = "water"
                        self.not_target = "sugar"
                        price = sqrt(partner.MRS * self.reservation_ratio)
                        can_trade = still_can_trade(price)
                    elif partner.MRS < self.reservation_ratio and positive_MRSs: 
                        partner.target = "water"
                        partner.not_target = "sugar"
                        self.target = "sugar"
                        self.not_target = "water"
                        price = sqrt(partner.MRS* self.reservation_ratio)
                        can_trade = still_can_trade(price)
                    else: 
                        can_trade = False
                        price = None


            return price, can_trade
        

        def executeTrade(partner, price):
            

            if not self.optimizer and not partner.optimizer: 
               
                # while the agents remain above their reservation demand for the good they are losing and can afford the trade 
                if self.reservation_ratio > partner.reservation_ratio: 
                    while (still_can_trade(price)): 
                        self.transfer(partner, price, 1)
                        # save price of sugar or implied price of sugar for every exchange
                        transaction_price = price if self.target == "sugar" else 1 / price
                        self.model.transaction_prices[self.target].append(transaction_price)
                        self.model.all_prices.append(transaction_price)
                        self.model.total_exchanges += 1
                        self.updateParams(trade=True)
                        partner.updateParams(trade=True)
                        price = sqrt(self.reservation_ratio * partner.reservation_ratio)
                        if self.arbitrageur:
                            self.reservation_ratio = (self.reservation_ratio * (
                                self.present_price_weight) + transaction_price) / self.present_price_weight
                        
                elif self.reservation_ratio < partner.reservation_ratio: 
                    while (still_can_trade(price)): 
                    
                         
                            partner.transfer(self, price, 1)
                            
                            # save price of sugar or implied price of sugar for every exchange
                            transaction_price = price if self.target == "sugar" else 1 / price
                            self.model.transaction_prices[self.target].append(transaction_price)
                            self.model.all_prices.append(transaction_price)
                            self.model.total_exchanges += 1
                            self.updateParams(trade=True)
                            partner.updateParams(trade=True)
                            price = sqrt(self.reservation_ratio * partner.reservation_ratio)
                            if self.arbitrageur:
                                self.reservation_ratio = (self.reservation_ratio * (
                                    self.present_price_weight) + transaction_price) / self.present_price_weight
            
            
            elif self.optimizer and partner.optimizer: 

                if self.MRS > partner.MRS: # self is relatively richer in water than partner 
                    while (still_can_trade(price)):# > 0 and partner.prospective_MRS(price) < price < self.prospective_MRS(price)): 
                               
                            self.transfer(partner, price, 1)
                            
                            
                            transaction_price = price if self.target == "sugar" else 1 / price
                            self.model.transaction_prices[self.target].append(transaction_price)
                            self.model.all_prices.append(transaction_price)
                            self.model.total_exchanges += 1
                            self.updateParams(trade=True)
                            partner.updateParams(trade=True)

                            price = sqrt(self.MRS * partner.MRS) 
                            # if self.arbitrageur:
                            #     self.expected_price = (self.expected_price * (
                            #         self.present_price_weight) + transaction_price) / self.present_price_weight
                                
                elif self.MRS < partner.MRS: 
        
                    while (still_can_trade(price)):# > 0 and self.prospective_MRS(price) < price < partner.prospective_MRS(price)): 
                          
                            partner.transfer(self, price, 1)
                            
                            # save price of sugar or implied price of sugar for every exchange
                            transaction_price = price if self.target == "sugar" else 1 / price
                            self.model.transaction_prices[self.target].append(transaction_price)
                            self.model.all_prices.append(transaction_price)
                            self.model.total_exchanges += 1
                            self.updateParams(trade=True)
                            partner.updateParams(trade=True)

                            price = sqrt(self.MRS * partner.MRS)
                            # if partner.arbitrageur:
                            #     partner.expected_price = (partner.expected_price * (
                            #         partner.present_price_weight) + transaction_price) / partner.present_price_weight

                                
            elif self.optimizer and not partner.optimizer:
                if self.MRS > partner.reservation_ratio: 
                    while (still_can_trade(price)):
                         
                            self.transfer(partner, price, 1)
                            
                            # save price of sugar or implied price of sugar for every exchange
                            transaction_price = price if self.target == "sugar" else 1 / price
                            self.model.transaction_prices[self.target].append(transaction_price)
                            self.model.all_prices.append(transaction_price)
                            self.model.total_exchanges += 1
                            self.updateParams(trade=True)
                            partner.updateParams(trade=True)
                            price = sqrt(self.MRS * partner.reservation_ratio)
                            # if self.arbitrageur:
                            #     self.expected_price = (self.expected_price * (
                            #         self.present_price_weight) + transaction_price) / self.present_price_weight
                elif self.MRS < partner.reservation_ratio:  
                        
                        while (still_can_trade(price)): 
                    
                         
                            self.transfer(partner, price, 1)
                            
                            # save price of sugar or implied price of sugar for every exchange
                            transaction_price = price if self.target == "sugar" else 1 / price
                            self.model.transaction_prices[self.target].append(transaction_price)
                            self.model.all_prices.append(transaction_price)
                            self.model.total_exchanges += 1
                            self.updateParams(trade=True)
                            partner.updateParams(trade=True)
                            price = sqrt(self.MRS * partner.reservation_ratio)
                            # if self.arbitrageur:
                            #     self.expected_price = (self.expected_price * (
                            #         self.present_price_weight) + transaction_price) / self.present_price_weight
                                
               

            elif not self.optimizer and partner.optimizer:
                if partner.MRS > self.reservation_ratio: 
                    while (still_can_trade(price)):
                       
                            partner.transfer(self, price, 1)
                            
                            # save price of sugar or implied price of sugar for every exchange
                            transaction_price = price if self.target == "sugar" else 1 / price
                            self.model.transaction_prices[self.target].append(transaction_price)
                            self.model.all_prices.append(transaction_price)
                            self.model.total_exchanges += 1
                            partner.updateParams(trade=True)
                            self.updateParams(trade=True)
                            price = sqrt(partner.MRS * self.reservation_ratio)
                            # if self.arbitrageur:
                            #     self.expected_price = (self.expected_price * (
                            #         self.present_price_weight) + transaction_price) / self.present_price_weight
                elif partner.MRS < self.reservation_ratio:  
                        
                        while (still_can_trade(price)): 
                    
                            partner.transfer(self, price, 1)
                            
                            # save price of sugar or implied price of sugar for every exchange
                            transaction_price = price if self.target == "sugar" else 1 / price
                            self.model.transaction_prices[self.target].append(transaction_price)
                            self.model.all_prices.append(transaction_price)
                            self.model.total_exchanges += 1
                            partner.updateParams(trade=True)
                            self.updateParams(trade=True)
                            price = sqrt(partner.MRS * self.reservation_ratio)
                            # if self.arbitrageur:
                            #     self.expected_price = (self.expected_price * (
                            #         self.present_price_weight) + transaction_price) / self.present_price_weight

                                   
        def herdTraits(agent, partner):
            def turn_off_other_primary_breeds(agent, breed, have_attr):
                if attr in self.model.primary_breeds:
                    # if breed changed, set other values false
                    if have_attr == True:
                        for brd in self.model.primary_breeds:
                            if brd != breed:
                                setattr(agent, brd, False)

            copy_attributes = partner.defineInheritance()
            l = list(copy_attributes.items())
            random.shuffle(l)
            copy_attributes = dict(l)

            del copy_attributes["mutate_rate"]
            mut_dict = {"mutate_rate": partner.mutate_rate}
            ordered_traits = {**mut_dict, **copy_attributes}

            # Copy rlearner-specific attributes
            rlearner_attributes = ["min_res_ratio", "max_res_ratio", "num_bins", "nn", "memory"]
            for attr in rlearner_attributes:
                if hasattr(partner, attr):
                    if attr == "nn":
                        # Copy discount rate and learning rate from the partner's neural network
                        ordered_traits["discount_rate"] = partner.nn.discount_rate
                        ordered_traits["learning_rate"] = partner.nn.learning_rate
                    ordered_traits[attr] = copy.deepcopy(getattr(partner, attr))

            if agent.model.genetic:
                for attr, val in ordered_traits.items():
                    if random.random() <= agent.model.cross_over_rate:
                        if isinstance(attr, numbers.Number):
                            setattr(agent, attr, 0)
                            setattr(agent, attr, getattr(agent, attr) + attr)
                        else:
                            setattr(agent, attr, val)
                        # if attr is a primary breed, other breeds
                        # will be switched off
                        turn_off_other_primary_breeds(agent, attr, val)

                # set basic True if all primary breeds switched to false
                # due to genetic algorithm
                if True not in (getattr(agent, breed)
                                for breed in self.model.primary_breeds):
                    agent.setBreedBasic(herder=agent.herder)
                agent.selectBreedParameters(mutate=False, parent=None,
                                            herding=True, partner=partner)

            else:
                for attr, val in copy_attributes.items():
                    setattr(agent, attr, val)
    ###############################################################################            

        # find trading partner
        neighbor_patches = [(self.row + i, self.col + j)
                        for i in self.model.nav_dict[1] if 0 <= self.row + i < 50
                        for j in self.model.nav_dict[1][i] if 0 <= self.col + j < 50 ]
        random.shuffle(neighbor_patches)
        for coords in neighbor_patches:
            if coords not in self.model.empty_patches.keys:
                row, col = coords[0], coords[1]
                target_patch = self.model.patches_dict[row][col]
                # if partner found on patch, ask to trade
                if target_patch.agent != None: 
                    partner, right_good = askToTrade(target_patch)
                
                    if right_good: 
                        price, can_trade = bargain(partner)

                    
                    else:
                        price, can_trade = None, False 
                    # check if partner has appropriate goods and WTP, WTA
                    if can_trade:
                                            
                        # execute trades
                        executeTrade(partner, price)
                        if self.herder:
                            if self.top_wealth <  getattr(partner, self.herding_metric):
                                herdTraits(self, partner)
                        if partner.herder:
                            if partner.top_wealth < getattr(self, partner.herding_metric):    
                                herdTraits(partner, self)
                        
                        #  genetic?
                        # only trade with one partner per agent search
                        # agents can be selected by more than one partner
                        self.update_stocks()
                        
                        