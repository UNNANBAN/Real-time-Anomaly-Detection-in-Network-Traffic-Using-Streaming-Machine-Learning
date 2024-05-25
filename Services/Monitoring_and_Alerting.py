import logging
import threading
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from Services import Logging_and_Auditing, Feature_Engineering, Model_Retraining

class Monitor:
    def __init__(self, model, kafka_bootstrap_servers, kafka_topic):
        """
        Initialize the monitoring system with a trained model.

        Args:
        - model: Trained machine learning model.
        - kafka_bootstrap_servers: Kafka bootstrap servers (list of strings).
        - kafka_topic: Kafka topic to consume messages from.
        """
        self.model = model
        self.kafka_consumer = KafkaConsumer(kafka_topic,
                                            bootstrap_servers=kafka_bootstrap_servers,
                                            group_id=None)

    def start_monitoring(self):
        df = pd.read_csv('Dataset/data.csv')
        c,retrain_req=0,0

        for message in self.kafka_consumer:
            # Assuming the message value is the data batch
            new_row = message.value.decode('utf-8').split(',')
            df.loc[len(df)] = new_row
            c+=1
            if c==1000:
                c,retrain_req=0,retrain_req+1
                data_batch = Feature_Engineering.perform_feature_engineering(df)
                anomalies = self.monitor_traffic(data_batch.iloc[-1000:])
                alerts = self.generate_alerts(anomalies)

                for alert in alerts:
                    if "No anomalies detected" in alert:
                        Logging_and_Auditing.log_activity(alert, level=Logging_and_Auditing.NO_ANOMALIES)
                    else:
                        Logging_and_Auditing.log_activity(alert, level=logging.WARNING)
                
                if retrain_req==10:
                    # Start a new thread for model retraining
                    retrain_thread = threading.Thread(target=self.retrain_model_in_background, args=(df,))
                    retrain_thread.start()
                    retrain_req=0

    def retrain_model_in_background(self, data):
        # Retrain the model in the background
        data.iloc[-1000:].to_csv('Dataset/retrain.csv', mode='a', header=False, index=False)
        self.model=Model_Retraining.retrain_model()
        Logging_and_Auditing.log_activity("Model retraining completed and saved")

    def monitor_traffic(self, data_batch):
        # Assuming 'attack' column is the target variable
        X = data_batch.drop(columns=["attack"])
        predictions = self.model.predict_proba(X)[:, 1]

        # Define the threshold for anomaly detection (you need to define this threshold)
        threshold = 0.5

        # Identify anomalies based on predictions and threshold
        anomalies = (predictions > threshold).astype(int)
        return anomalies

    def generate_alerts(self, anomalies):
        alerts = []
        num_anomalies = np.sum(anomalies)

        if num_anomalies > 0:
            alerts.append(f"{num_anomalies} anomalies detected in the incoming traffic.")
            alerts.append("Alert: Potential network intrusion detected!")
        else:
            alerts.append("No anomalies detected in the incoming traffic.")

        return alerts
