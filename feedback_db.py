"""
Feedback database module for simple RLHF system
Handles CSV import/export and data management for feedback
"""

import os
import csv
import json
import sqlite3
import pandas as pd
from datetime import datetime
import logging
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeedbackDatabase:
    """
    Simple database for storing prompts, model outputs, and human feedback ratings.
    Supports CSV import/export for feedback collection.
    """
    
    def __init__(self, db_path="data/feedback.db"):
        """
        Initialize the feedback database.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.cursor = None
        self.lock = threading.RLock()  # Add thread lock for synchronization
        
        logger.info(f"Initializing feedback database at {db_path}")
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish connection to the database."""
        try:
            # Add check_same_thread=False to allow cross-thread usage
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            logger.info("Connected to database")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            with self.lock:  # Use lock for thread safety
                # Table for storing prompts and outputs together for simplicity
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompt_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    output TEXT NOT NULL,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Table for storing human feedback
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_output_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL,
                    comments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prompt_output_id) REFERENCES prompt_outputs (id)
                )
                ''')
                
                self.conn.commit()
                logger.info("Database tables created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
    
    def add_prompt_output(self, prompt, output, model_version="mistral-7b"):
        """
        Add a prompt-output pair to the database.
        
        Args:
            prompt (str): The input prompt
            output (str): The model's output
            model_version (str): Version or identifier of the model
            
        Returns:
            int: ID of the inserted prompt-output pair
        """
        try:
            with self.lock:  # Use lock for thread safety
                self.cursor.execute(
                    "INSERT INTO prompt_outputs (prompt, output, model_version) VALUES (?, ?, ?)",
                    (prompt, output, model_version)
                )
                self.conn.commit()
                prompt_output_id = self.cursor.lastrowid
                logger.info(f"Added prompt-output pair with ID {prompt_output_id}")
                return prompt_output_id
        except sqlite3.Error as e:
            logger.error(f"Error adding prompt-output pair: {str(e)}")
            with self.lock:
                self.conn.rollback()
            return None
    
    def add_feedback(self, prompt_output_id, rating, comments=None):
        """
        Add human feedback for a prompt-output pair.
        
        Args:
            prompt_output_id (int): ID of the associated prompt-output pair
            rating (int): Numerical rating (e.g., 1-5)
            comments (str, optional): Additional feedback comments
            
        Returns:
            int: ID of the inserted feedback
        """
        try:
            with self.lock:  # Use lock for thread safety
                self.cursor.execute(
                    "INSERT INTO feedback (prompt_output_id, rating, comments) VALUES (?, ?, ?)",
                    (prompt_output_id, rating, comments)
                )
                self.conn.commit()
                feedback_id = self.cursor.lastrowid
                logger.info(f"Added feedback with ID {feedback_id} for prompt-output {prompt_output_id}")
                return feedback_id
        except sqlite3.Error as e:
            logger.error(f"Error adding feedback: {str(e)}")
            with self.lock:
                self.conn.rollback()
            return None
    
    def get_all_prompt_outputs(self, with_feedback_only=False):
        """
        Retrieve all prompt-output pairs, optionally filtered to those with feedback.
        
        Args:
            with_feedback_only (bool): If True, only return pairs that have feedback
            
        Returns:
            list: List of dictionaries containing prompt-output pairs
        """
        try:
            with self.lock:  # Use lock for thread safety
                query = """
                SELECT po.id, po.prompt, po.output, po.model_version, 
                       f.rating, f.comments, po.created_at
                FROM prompt_outputs po
                """
                
                if with_feedback_only:
                    query += " JOIN feedback f ON po.id = f.prompt_output_id "
                else:
                    query += " LEFT JOIN feedback f ON po.id = f.prompt_output_id "
                    
                query += " ORDER BY po.created_at DESC "
                
                self.cursor.execute(query)
                rows = self.cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'id': row[0],
                        'prompt': row[1],
                        'output': row[2],
                        'model_version': row[3],
                        'rating': row[4],
                        'comments': row[5],
                        'created_at': row[6]
                    })
                
                logger.info(f"Retrieved {len(results)} prompt-output pairs")
                return results
        except sqlite3.Error as e:
            logger.error(f"Error retrieving prompt-output pairs: {str(e)}")
            return []
    
    def get_training_data(self, min_rating=None):
        """
        Retrieve training data for RLHF, optionally filtered by minimum rating.
        
        Args:
            min_rating (int, optional): Minimum rating threshold
            
        Returns:
            list: List of dictionaries containing training data
        """
        try:
            with self.lock:  # Use lock for thread safety
                query = """
                SELECT po.prompt, po.output, f.rating
                FROM prompt_outputs po
                JOIN feedback f ON po.id = f.prompt_output_id
                """
                
                params = []
                if min_rating is not None:
                    query += " WHERE f.rating >= ? "
                    params.append(min_rating)
                    
                query += " ORDER BY f.rating DESC "
                
                self.cursor.execute(query, params)
                rows = self.cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'prompt': row[0],
                        'completion': row[1],
                        'rating': row[2]
                    })
                
                logger.info(f"Retrieved {len(results)} training examples")
                return results
        except sqlite3.Error as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            return []
    
    def import_from_csv(self, csv_path):
        """
        Import feedback data from a CSV file.
        Expected CSV format: prompt,output,rating,comments
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            int: Number of records imported
        """
        try:
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return 0
            
            df = pd.read_csv(csv_path)
            required_columns = ['prompt', 'output', 'rating']
            
            # Validate CSV format
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"CSV file missing required column: {col}")
                    return 0
            
            # Add comments column if missing
            if 'comments' not in df.columns:
                df['comments'] = None
            
            # Import data
            count = 0
            for _, row in df.iterrows():
                prompt_output_id = self.add_prompt_output(
                    row['prompt'], 
                    row['output']
                )
                
                if prompt_output_id:
                    self.add_feedback(
                        prompt_output_id,
                        int(row['rating']),
                        row['comments']
                    )
                    count += 1
            
            logger.info(f"Imported {count} records from CSV")
            return count
        
        except Exception as e:
            logger.error(f"Error importing from CSV: {str(e)}")
            return 0
    
    def export_to_csv(self, csv_path, with_feedback_only=True):
        """
        Export data to a CSV file.
        
        Args:
            csv_path (str): Path to save the CSV file
            with_feedback_only (bool): If True, only export pairs that have feedback
            
        Returns:
            int: Number of records exported
        """
        try:
            data = self.get_all_prompt_outputs(with_feedback_only)
            
            if not data:
                logger.warning("No data to export")
                return 0
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Select and rename columns for export
            export_df = df[['prompt', 'output', 'rating', 'comments']]
            
            # Export to CSV
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            export_df.to_csv(csv_path, index=False)
            
            logger.info(f"Exported {len(data)} records to {csv_path}")
            return len(data)
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return 0
    
    def export_training_data_json(self, json_path, min_rating=None):
        """
        Export training data to a JSON file for RLHF.
        
        Args:
            json_path (str): Path to save the JSON file
            min_rating (int, optional): Minimum rating threshold
            
        Returns:
            int: Number of records exported
        """
        try:
            training_data = self.get_training_data(min_rating)
            
            if not training_data:
                logger.warning("No training data to export")
                return 0
            
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            logger.info(f"Exported {len(training_data)} training examples to {json_path}")
            return len(training_data)
        
        except Exception as e:
            logger.error(f"Error exporting training data: {str(e)}")
            return 0
    
    def generate_example_csv(self, csv_path, num_examples=5):
        """
        Generate an example CSV file with the correct format.
        
        Args:
            csv_path (str): Path to save the example CSV file
            num_examples (int): Number of example rows to generate
            
        Returns:
            bool: Success status
        """
        try:
            examples = []
            
            # Example 1: Troubleshooting runbook
            examples.append({
                'prompt': 'Write a troubleshooting runbook for when a web server returns 503 errors',
                'output': '# Troubleshooting 503 Service Unavailable Errors\n\n## Overview\n503 Service Unavailable errors indicate that the server is temporarily unable to handle the request due to maintenance or overloading...',
                'rating': 4,
                'comments': 'Good structure but could include more specific Apache/Nginx configurations'
            })
            
            # Example 2: Network configuration
            examples.append({
                'prompt': 'Create a network configuration for a secure DMZ setup',
                'output': '# DMZ Network Configuration\n\n## Network Topology\n1. External Firewall (connects to internet)\n2. DMZ Segment\n3. Internal Firewall\n4. Internal Network...',
                'rating': 5,
                'comments': 'Excellent, very comprehensive'
            })
            
            # Example 3: Poor response
            examples.append({
                'prompt': 'Explain how to troubleshoot high CPU usage on a Linux server',
                'output': 'To troubleshoot high CPU usage, you should check what processes are using CPU. You can use top command.',
                'rating': 2,
                'comments': 'Too brief, missing specific commands and interpretation guidance'
            })
            
            # Example 4: Medium quality
            examples.append({
                'prompt': 'Write a script to monitor disk space and send alerts',
                'output': '#!/bin/bash\n\nTHRESHOLD=90\nEMAIL="admin@example.com"\n\ndf -h | grep -vE "Filesystem|tmpfs" | awk \'{ print $5 " " $1 }\' | while read output;\ndo\n  usage=$(echo $output | awk \'{ print $1}\' | cut -d\'%\' -f1 )\n  partition=$(echo $output | awk \'{ print $2 }\')\n  if [ $usage -ge $THRESHOLD ]; then\n    echo "ALERT: Partition $partition is ${usage}% full" | mail -s "Disk Space Alert" $EMAIL\n  fi\ndone',
                'rating': 4,
                'comments': 'Good script but could use more comments'
            })
            
            # Example 5: Another good response
            examples.append({
                'prompt': 'Create a checklist for securing a new Ubuntu server',
                'output': '# Ubuntu Server Security Checklist\n\n## Initial Setup\n1. Update system packages\n2. Configure firewall (UFW)\n3. Set up SSH key authentication\n4. Disable root login\n\n## User Management\n1. Create non-root admin user\n2. Configure sudo with minimal privileges\n\n## System Hardening\n1. Install and configure fail2ban\n2. Set up automatic security updates\n3. Configure system logging\n\n## Application Security\n1. Install only necessary services\n2. Configure each service securely\n3. Use application firewalls where appropriate',
                'rating': 5,
                'comments': 'Very thorough and well-organized'
            })
            
            # Limit to requested number
            examples = examples[:num_examples]
            
            # Create DataFrame and export
            df = pd.DataFrame(examples)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Generated example CSV with {num_examples} rows at {csv_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error generating example CSV: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            with self.lock:  # Use lock for thread safety
                self.conn.close()
                logger.info("Database connection closed")


# Example usage
if __name__ == "__main__":
    db = FeedbackDatabase()
    
    # Generate example CSV
    db.generate_example_csv("data/example_feedback.csv")
    
    # Import from CSV
    db.import_from_csv("data/example_feedback.csv")
    
    # Export training data
    db.export_training_data_json("data/training_data.json", min_rating=3)
    
    # Close connection
    db.close()
