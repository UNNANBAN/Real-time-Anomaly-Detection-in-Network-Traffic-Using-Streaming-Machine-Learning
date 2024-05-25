import os
import joblib
import threading
import pandas as pd
from Services import Feature_Engineering, Online_Learning, Monitoring_and_Alerting, Logging_and_Auditing, Model_Retraining, Streaming_Data_Integration

if __name__ == "__main__":
    # Set up logging
    Logging_and_Auditing.setup_logging()

    # Load raw data from train.csv
    existing_data = pd.read_csv("Dataset/train.csv")
    
    # Log activity
    Logging_and_Auditing.log_activity("Raw data loaded from train.csv")

    # Perform feature engineering
    feature_engineered_data = Feature_Engineering.perform_feature_engineering(existing_data)

    # Log activity
    Logging_and_Auditing.log_activity("Feature engineering completed")

    # Path to store the trained model
    model_save_path = "Services/trained_model.joblib"

    # Check if the trained model already exists
    if os.path.exists(model_save_path):
        # Load the existing model
        model = joblib.load(model_save_path)
        Logging_and_Auditing.log_activity("Existing model loaded from storage")
    else:
        # Train the model and save it
        model = Online_Learning.train_incremental_random_forest(feature_engineered_data, model_save_path)
        Logging_and_Auditing.log_activity("Online learning completed and model saved")

    # Start the producer in a separate thread
    producer_thread = threading.Thread(target=Streaming_Data_Integration.produce)
    producer_thread.start()

    # Log activity
    Logging_and_Auditing.log_activity("Kafka Producer Initiated")

    # Initialize the monitor with the trained model
    kafka_bootstrap_servers = ['localhost:9092']
    kafka_topic = 'network_traffic_stream'
    monitor = Monitoring_and_Alerting.Monitor(model, kafka_bootstrap_servers, kafka_topic)

    # Start monitoring the Kafka topic
    monitor.start_monitoring()