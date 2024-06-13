import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

class DataAggregator:
    """
    This class is responsible for aggregating and managing data collected from the Sugarscape simulations.
    It handles the creation of data directories, collection of agent and model attributes, and storage of this data in various formats.
    """

    def __init__(self, primary_breed_set, agent_attributes, model_attributes):
        """
        Initializes the DataAggregator with necessary attributes.

        Args:
            primary_breed_set (list): List of primary breeds involved in the simulation.
            agent_attributes (list): List of attributes to track for each agent.
            model_attributes (list): List of attributes to track for the model.

        Creates a directory for storing data and handles directory existence issues.
        """
        self.folder = os.path.join(os.getcwd(), "parquet", "-".join(primary_breed_set))
        self.agent_attributes = agent_attributes
        self.model_attributes = model_attributes
        self.attributes = agent_attributes + model_attributes
        self.primary_breeds = primary_breed_set

        try:
            os.mkdir(self.folder)
        except FileExistsError:
            # Remove all files in the existing folder to avoid errors
            shutil.rmtree(self.folder)
            os.mkdir(self.folder)

    def set_folder(self, breed_set): 
        self.folder = os.path.join(os.getcwd(), "parquet", "-".join(breed_set))

    def prepSetting(self):
        """
        Prepares the settings by ensuring the data folder exists.
        Creates the folder if it does not exist.
        """
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        for attr in self.attributes:
            attr_path =  os.path.join(self.folder, attr)
            if not os.path.exists(attr_path): 
                os.makedirs(attr_path)



    def collectData(self, model, name, run, period):
        """
        Collects data from agents and the model for a specific period.

        Args:
            model (Model): The model instance from which to collect data.
            name (str): The name of the current trial.
            run (int): The current run number.
            period (int): The current period number.
        """
        def collectAgentAttributes():
            for attribute in self.agent_attributes:
                self.trial_data[name][run][attribute][period] = []
                for ID, agent in model.agent_dict.items():
                    self.trial_data[name][run][attribute][period].append(getattr(agent, attribute))
                self.trial_data[name][run][attribute][period] = np.mean(self.trial_data[name][run][attribute][period])

        def collectModelAttributes():
            for attribute in self.model_attributes:
                self.trial_data[name][run][attribute][period] = getattr(model, attribute)

        collectAgentAttributes()
        collectModelAttributes()

    def saveRun(self, name, run, run_data):
        """
        Saves the data of a particular run into a parquet file.

        Args:
            name (str): The name of the current trial.
            run (int): The current run number.
            run_data (dict): The data to be saved for this run.
        """
        df = pd.DataFrame.from_dict(run_data)
        table = pa.Table.from_pandas(df)
        for attr in df.keys():
            file_name = os.path.join(self.folder, attr, f"{run}.parquet")
            try: 
                pq.write_table(table.select([attr]), file_name)
            except:            
                print(f"error saving run data: {attr}") 

               # print(table)
                
                continue

    def saveData(self, name, trial):
        """
        Placeholder for a method to save trial data.
        Currently not implemented.

        Args:
            name (str): The name of the current trial.
            trial (int): The trial number.
        """
        pass  # Implementation pending

    def saveDistributionByPeriod(self, name):
        """
        Saves the distribution of attributes by period.

        Args:
            name (str): The name of the current trial.
        """
        self.distribution_dict = {name: {attr: {trial: {} for trial in self.trial_data[name]}
                                         for attr in self.attributes}}
        for attr in self.attributes:
            for trial in self.trial_data[name]:
                self.distribution_dict[name][attr][trial] = self.trial_data[name][trial][attr]

    def saveDistributionByPeriodWithParquet(self, name, runs):
        """
        Saves the distribution of attributes by period into a parquet file.

        Args:
            name (str): The name of the current trial.
            runs (int): The number of runs to process.
        """
        for attr in self.attributes:
            attr_df = self.create_attr_df_from_parquet(attr, runs)
            attr_df.index = attr_df.index.astype(int)
            path = os.path.join(self.folder, attr, f"{attr}_df")
            pq_table = pa.Table.from_pandas(attr_df)
            pq.write_table(pq_table, path)

    def create_attr_df_from_parquet(self, attr, runs):
        """
        Creates a DataFrame for a specific attribute from parquet files.

        Args:
            attr (str): The attribute for which to create the DataFrame.
            runs (int): The number of runs to include.

        Returns:
            pandas.DataFrame: The DataFrame containing the specified attribute's data.
        """
        attr_df = pd.DataFrame()
        for run in range(runs):
            filepath = os.path.join(self.folder, attr, f"{run}.parquet")
            run_df = pd.read_parquet(filepath)
            attr_df[run] = run_df[attr]

        attr_df = attr_df.astype(np.float32)
        return attr_df

    def remove_shelves(self):
        """
        Removes temporary shelf files from the data directory.
        """
        def process_files(path):
            files = os.listdir(path)
            for file in files:
                if file.endswith((".dat", ".dir", ".bak")):
                    os.remove(os.path.join(path, file))

        process_files(self.folder)
        process_files(".")

    def remove_parquet(self):
        """
        Removes parquet files from the data directory and subdirectories.
        """
        def process_files(path):
            for folder in os.listdir(path):
                folder_path = os.path.join(path, folder)
                for file in os.listdir(folder_path):
                    if file.endswith(".parquet"):
                        os.remove(os.path.join(folder_path, file))

        process_files(self.folder)
        print("Data collection completed.")
