import os
import psycopg2
import numpy as np
import pandas as pd

class DataAggregatorSQL:
    def __init__(self, primary_breed_set, agent_attributes, model_attributes, db_config):
        self.primary_breeds = primary_breed_set
        self.agent_attributes = agent_attributes
        self.model_attributes = model_attributes
        self.attributes = agent_attributes + model_attributes
        self.db_config = db_config
        self.connection = self.create_db_connection()

    def create_db_connection(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print("Error connecting to PostgreSQL database:", e)
            return None

    def create_table(self):
        create_table_query = f"""CREATE TABLE IF NOT EXISTS simulation_data (
                                 id SERIAL PRIMARY KEY,
                                 breed_set VARCHAR(255),
                                 run INT,
                                 period INT,
                                 {', '.join([f"{attr} FLOAT" for attr in self.attributes])}
                             );"""
        with self.connection.cursor() as cursor:
            cursor.execute(create_table_query)
            self.connection.commit()

    def collectData(self, model, name, run, period):
        data_row = {
            "trial_name": name,
            "run": run,
            "period": period
        }

        def collectAgentAttributes():
            for attribute in self.agent_attributes:
                values = [getattr(agent, attribute) for agent in model.agent_dict.values()]
                data_row[attribute] = np.mean(values)

        def collectModelAttributes():
            for attribute in self.model_attributes:
                data_row[attribute] = getattr(model, attribute)

        collectAgentAttributes()
        collectModelAttributes()
        self.insert_data(data_row)

    def insert_data(self, data):
        columns = ', '.join(data.keys())
        values = ', '.join(['%s'] * len(data))
        insert_query = f"INSERT INTO simulation_data ({columns}) VALUES ({values})"
        with self.connection.cursor() as cursor:
            cursor.execute(insert_query, list(data.values()))
            self.connection.commit()

    def close_connection(self):
        if self.connection:
            self.connection.close()


