import os
import time
import random
import multiprocess
from tkinter import Tk, Canvas
from DataAggregator import DataAggregator
from memory_profiler import memory_usage
import gc
from Model import Model

class GUI:
    """
    This class handles the graphical user interface for the Sugarscape simulation.
    It creates and manages the visualization of agents and patches, and integrates with the model for simulation runs.
    """

    def __init__(self, name, run, num_agents, live_visual, plots, model_primary_breeds, 
                every_t_frames_GUI=1, every_t_frames_plots=100, 
                mutate=True, genetic=True, agent_attributes=None, 
                model_attributes=None):
        """
        Initializes the GUI with the given parameters and creates the visualization components if live_visual is True.

        Args:
            name (str): Name identifier for the simulation run.
            run (int): Current run number.
            num_agents (int): Number of agents to simulate.
            live_visual (bool): Whether to show live visualization.
            plots (bool): Whether to generate plots.
            model_primary_breeds (list): List of primary breeds in the model.
            every_t_frames_GUI (int): Update frequency for GUI frames.
            every_t_frames_plots (int): Update frequency for plotting.
            mutate (bool): Flag to enable/disable mutation in agents.
            genetic (bool): Flag to enable/disable genetic algorithms.
            agent_attributes (list): List of agent attributes for data collection.
            model_attributes (list): List of model attributes for data collection.
        """
        if live_visual:
            self.parent = Tk()

        self.name = name
        self.run = run
        self.plots = plots
        self.model = Model(self, num_agents, mutate, genetic, live_visual, plots, agent_attributes, model_attributes, primary_breeds=model_primary_breeds)
        self.dimPatch = 16
        self.live_visual = live_visual
        self.every_t_frames_GUI = every_t_frames_GUI
        self.every_t_frames_plots = every_t_frames_plots

        if self.live_visual: 
            canvasWidth = self.model.cols * self.dimPatch
            canvasHeight = self.model.rows * self.dimPatch
            self.canvas = Canvas(self.parent, width=canvasWidth, height=canvasHeight, background="white")
            self.canvas.pack()
            self.drawPatches()
            self.drawAgents()
            self.canvas.update()
            
    def drawPatches(self):
        """
        Draws the patches on the canvas using the patch attributes from the model.
        """
        for i in self.model.patches_dict:
            for patch in self.model.patches_dict[i].values():
                patch.image = self.canvas.create_rectangle(
                    patch.col * self.dimPatch,  # Left x coordinate
                    patch.row * self.dimPatch,  # Top y coordinate
                    (patch.col + 1) * self.dimPatch,  # Right x coordinate
                    (patch.row + 1) * self.dimPatch,  # Bottom y coordinate
                    fill=self.color(patch.Q - 1, patch.good),  # Color based on quantity and type
                    width=0  # Border width set to 0 for a filled rectangle
                )

    def drawAgent(self, agent):
        """
        Draws an individual agent on the canvas.

        Args:
            agent: The agent object to be drawn.
        """
        agent.image = self.canvas.create_oval(
            agent.col * self.dimPatch + 2,
            agent.row * self.dimPatch + 2,
            (agent.col + 1) * self.dimPatch - 2,
            (agent.row + 1) * self.dimPatch - 2,
            fill='red',  # Initial color; this could be modified based on agent properties
            width=0  # Border width
        )

    def drawAgents(self):
        """
        Iterates through all agents in the model and draws them.
        """
        for agent in self.model.agent_dict.values():
            self.drawAgent(agent)

    def moveAgents(self):
        """
        Moves agents on the canvas based on their updated positions in the model.
        """
        for agent in self.model.agent_dict.values():
            self.canvas.move(agent.image, agent.dx * self.dimPatch, agent.dy * self.dimPatch)
            color, outline = self.agentColor(agent)
            self.canvas.itemconfig(agent.image, fill=color, outline=outline, width=2)

    def agentColor(self, agent):
        """
        Determines the color and outline of an agent based on its properties.

        Args:
            agent: The agent object.

        Returns:
            Tuple (color, outline): The color and outline for the agent.
        """
        color = "red"  # Default color
        if agent.arbitrageur:
            color = "green"
        if agent.optimizer:
            color = "magenta"
        outline = "black" if agent.herder else color
        return color, outline

    def updatePatches(self):
        """
        Updates the color of patches on the canvas based on their current state.
        """
        for i in self.model.patches_dict:
            for patch in self.model.patches_dict[i].values():
                self.canvas.itemconfig(patch.image, fill=self.color(patch.Q, patch.good))

    def color(self, q, good):
        """
        Determines the color for a patch based on its quantity and type.

        Args:
            q (int): Quantity level of the patch.
            good (str): The type of good (e.g., 'sugar').

        Returns:
            str: Hexadecimal color string.
        """
        if q > 5:
            q = 5
        q = int(q)
        if good == "sugar":
            rgb = (255 - 3 * q, 255 - 10 * q, 255 - 51 * q)
        else:
            rgb = (30 - 3 * q, 50 - 5 * q, 255 - 35 * q)

        color = '#'
        for v in rgb:
            hx = hex(v)[2:]
            hx = hx.zfill(2)  # Ensure hexadecimal is two characters
            color += hx
        return color


# List of attributes to be monitored for agents and the model
agent_attributes = []  # Attributes like "water", "sugar", "wealth", etc. (currently empty)
model_attributes = ["population", "total_exchanges", "total_agents_created", "total_avg_price",
                    "runtime", "agent_wealth", "price_variance", "preference_variance", "cw", "cs",
                    "real_income_per_capital", "wealth_per_capita", "savings", "income", "consumption", 
                    "num_optimizers", "num_herders", "num_basics", "num_rlearners", "basic_wealth_per_capita", 
                    "optimizer_wealth_per_capita", "rlearner_wealth_per_capita", "num_wealth_herders", "num_progenycount_herders",
                    "herder_wealth_per_capita", "wealth_herder_wealth_per_capita", 
                    "progenycount_herder_wealth_per_capita", "mutate_rate", "max_mutate_rate", 
                    "price_change", "reservation_ratio", "reproduction_criteria_water", 
                    "reproduction_criteria_sugar", "reproduction_ratio_water", "reproduction_ratio_sugar",
                    "avg_learning_rate", "avg_discount_rate"]

# Set of primary breeds to be used in the simulation
breed_sets = [["rlearner"]]

# Number of runs and periods for the simulation
#runs = 5
#periods = 10000
data_collecting = True  # Flag to indicate if data should be collected

# Main loop for running simulations
def run_simulation(params):
    primary_breed_set, mutate, genetic, run, data_collecting, agent_attributes,model_attributes, data_agg = params

    periods = 10000 if primary_breed_set in [["basic", "optimizer"], ["optimizer"]] else 10000

    # Iterating over different configurations
    # for mutate in [True]:
    #     for genetic in [True]:
    name = "sugarscape"
    print("mutate", "genetic", sep="\t")
    print(mutate, genetic, sep="\t")
    print("trial", "agents", "periods", "time", sep="\t")


    # Running multiple simulation runs
    #for run in range(runs):
    mem_usage = memory_usage(-1, interval=1)
    print(run, "mem:", str(int(mem_usage[0])) + " MB", sep="\t")

    num_agents = 200
    start = time.time()
    gui_instance = GUI(name + str(run), run, num_agents, live_visual=False, plots=True,
                    model_primary_breeds=primary_breed_set, mutate=True, genetic=True,
                    agent_attributes=agent_attributes, model_attributes=model_attributes)
    
    # Running the model for the specified number of periods
    gui_instance.model.runModel(periods)

    # Saving run data and cleaning up if data collection is enabled
    if data_collecting: 
        data_agg.saveRun(gui_instance.name, str(gui_instance.run), gui_instance.model.data_dict)
        del gui_instance.model.data_dict
        gc.collect()

    # Closing the GUI visualization
    if gui_instance.live_visual:
        gui_instance.parent.quit()
        gui_instance.parent.destroy()

    # Calculating and printing runtime
    end = time.time()
    elapse = end - start
    print("runtime:", int(elapse), "seconds", sep="\t")

    return

# Main loop for setting up multiprocessing
if __name__ == '__main__':
    print("running")
    runs = 4
    data_collecting = True  # Flag to indicate if data should be collected
    mutate = True
    genetic = True
    

    for primary_breed_set in breed_sets:
        # Parameters for multiprocessing
        pool = multiprocess.Pool(processes=4)#multiprocess.cpu_count() - 4)  # decide how many cores to use 
        # Initialize DataAggregator if data collection is enabled
        if data_collecting: 
             data_agg = DataAggregator(primary_breed_set, agent_attributes, model_attributes)
             data_agg.prepSetting()
        tasks = []
        for run in range(runs):
            tasks.append((primary_breed_set, mutate, genetic, run, data_collecting, agent_attributes, model_attributes, data_agg))

        

        # Run simulations in parallel and close pool
        with pool as p: 
            p.map(run_simulation, tasks)
            p.close()

        # Saving and cleaning up data if data collection is enabled
        if data_collecting: 
            data_agg.saveDistributionByPeriodWithParquet("sugarscape", runs)
            
    if data_collecting:         
            for primary_breed_set in breed_sets: 
                data_agg.set_folder(primary_breed_set)

                data_agg.remove_parquet()

