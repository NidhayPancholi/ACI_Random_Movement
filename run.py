# Real-Time Traffic Analysis Platform
# Architecture and Implementation using Python and GCP

### 1. Data Ingestion Layer

from google.cloud.pubsub import PublisherClient
from google.cloud import storage
import json

class TrafficDataIngestion:
    def __init__(self, project_id, topic_name):
        self.publisher = PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_name)
        
    def ingest_traffic_data(self, traffic_data):
        """
        Ingest traffic data into Pub/Sub
        traffic_data format:
        {
            "timestamp": "2024-11-23T10:00:00Z",
            "location": {"lat": 37.7749, "lng": -122.4194},
            "speed": 45.5,
            "vehicle_count": 120,
            "road_segment": "US-101-N-Mile-400"
        }
        """
        try:
            data = json.dumps(traffic_data).encode("utf-8")
            future = self.publisher.publish(self.topic_path, data)
            return future.result()
        except Exception as e:
            print(f"Error publishing message: {e}")
            raise

### 2. Data Processing Layer

from apache_beam import Pipeline, Map, WindowInto
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window

class TrafficDataProcessor:
    def __init__(self, project_id, subscription_name):
        self.project_id = project_id
        self.subscription_path = f"projects/{project_id}/subscriptions/{subscription_name}"
        
    def process_traffic_data(self, element):
        """Process individual traffic data points"""
        traffic_data = json.loads(element.decode('utf-8'))
        
        # Calculate derived metrics
        processed_data = {
            "timestamp": traffic_data["timestamp"],
            "location": traffic_data["location"],
            "speed": traffic_data["speed"],
            "vehicle_count": traffic_data["vehicle_count"],
            "road_segment": traffic_data["road_segment"],
            "congestion_level": self.calculate_congestion_level(
                traffic_data["speed"], 
                traffic_data["vehicle_count"]
            ),
            "segment_status": self.determine_segment_status(traffic_data["speed"])
        }
        return processed_data
    
    @staticmethod
    def calculate_congestion_level(speed, vehicle_count):
        """Calculate congestion level based on speed and vehicle count"""
        if speed < 20:
            return "High"
        elif speed < 40:
            return "Medium"
        return "Low"
    
    @staticmethod
    def determine_segment_status(speed):
        """Determine road segment status"""
        if speed < 10:
            return "Blocked"
        elif speed < 25:
            return "Heavy Traffic"
        elif speed < 45:
            return "Moderate Traffic"
        return "Clear"

### 3. Prediction Model

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

class TrafficPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        
    def prepare_features(self, historical_data):
        """
        Prepare features for prediction
        Expected columns: timestamp, speed, vehicle_count, road_segment
        """
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Create lag features
        df['speed_lag1'] = df.groupby('road_segment')['speed'].shift(1)
        df['vehicle_count_lag1'] = df.groupby('road_segment')['vehicle_count'].shift(1)
        
        return df.dropna()
    
    def train(self, historical_data):
        """Train the prediction model"""
        df = self.prepare_features(historical_data)
        
        features = ['hour', 'day_of_week', 'speed_lag1', 'vehicle_count_lag1']
        X = df[features]
        y = df['speed']
        
        self.model.fit(X, y)
    
    def predict(self, current_data):
        """Make predictions for future traffic conditions"""
        df = self.prepare_features([current_data])
        features = ['hour', 'day_of_week', 'speed_lag1', 'vehicle_count_lag1']
        
        prediction = self.model.predict(df[features])
        return float(prediction[0])

### 4. Data Storage Layer

from google.cloud import bigquery

class TrafficDataStorage:
    def __init__(self, project_id):
        self.client = bigquery.Client(project=project_id)
        
    def store_processed_data(self, processed_data):
        """Store processed traffic data in BigQuery"""
        table_id = f"{self.project_id}.traffic_dataset.processed_traffic_data"
        
        rows_to_insert = [{
            "timestamp": processed_data["timestamp"],
            "location": processed_data["location"],
            "speed": processed_data["speed"],
            "vehicle_count": processed_data["vehicle_count"],
            "road_segment": processed_data["road_segment"],
            "congestion_level": processed_data["congestion_level"],
            "segment_status": processed_data["segment_status"]
        }]
        
        errors = self.client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            raise Exception(f"Error inserting rows: {errors}")

### 5. Main Application

def main():
    # Configuration
    project_id = "your-project-id"
    topic_name = "traffic-data"
    subscription_name = "traffic-data-sub"
    
    # Initialize components
    ingestion = TrafficDataIngestion(project_id, topic_name)
    processor = TrafficDataProcessor(project_id, subscription_name)
    predictor = TrafficPredictor()
    storage = TrafficDataStorage(project_id)
    
    # Set up Dataflow pipeline options
    pipeline_options = PipelineOptions(
        streaming=True,
        project=project_id,
        runner='DataflowRunner',
        region='us-central1',
        temp_location='gs://your-bucket/temp'
    )
    
    # Create and run the pipeline
    with Pipeline(options=pipeline_options) as pipeline:
        (pipeline
         | "Read from Pub/Sub" >> beam.io.ReadFromPubSub(subscription=subscription_path)
         | "Window" >> WindowInto(window.FixedWindows(60))  # 1-minute windows
         | "Process Data" >> Map(processor.process_traffic_data)
         | "Write to BigQuery" >> beam.io.WriteToBigQuery(
             table_id,
             schema=schema,
             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
         ))

if __name__ == "__main__":
    main()