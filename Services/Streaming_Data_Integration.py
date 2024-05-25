import time
import pandas as pd
from confluent_kafka import Producer

def generate_data():
    df = pd.read_csv('Dataset/streaming_data.csv')
    for _, row in df.iterrows():
        # Convert each row to a string representation
        data_str = ','.join(str(val) for val in row)
        yield data_str

def publish_data(producer, topic, data):
    producer.produce(topic, data)
    producer.flush()  # Flush messages to ensure they are sent immediately
    time.sleep(0.05)  # Introduce a delay of 0.05 seconds

def produce():
    # Kafka broker address
    bootstrap_servers = 'localhost:9092'
    # Kafka topic
    kafka_topic = 'network_traffic_stream'

    # Kafka producer configuration
    kafka_conf = {
        'bootstrap.servers': bootstrap_servers,
    }

    # Create Kafka producer instance
    producer = Producer(kafka_conf)

    # Generate data from the dataset
    data_generator = generate_data()

    # Publish data to Kafka topic
    for data in data_generator:
        publish_data(producer, kafka_topic, data)

    # Flush messages to ensure they are sent immediately
    producer.flush()

    # Set the producer object to None to release resources
    producer = None