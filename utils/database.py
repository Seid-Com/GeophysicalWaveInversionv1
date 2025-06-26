"""
Database utilities for FWI application
Handles storage of experiments, models, and training metrics
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

class FWIDatabase:
    """Database interface for FWI experiments and results"""
    
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('PGHOST'),
            'port': os.getenv('PGPORT'),
            'database': os.getenv('PGDATABASE'),
            'user': os.getenv('PGUSER'),
            'password': os.getenv('PGPASSWORD')
        }
        self._init_tables()
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.connection_params)
    
    def _init_tables(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Experiments table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        config JSONB NOT NULL,
                        status VARCHAR(50) DEFAULT 'created',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Training metrics table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS training_metrics (
                        id SERIAL PRIMARY KEY,
                        experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
                        epoch INTEGER NOT NULL,
                        train_loss FLOAT,
                        val_loss FLOAT,
                        train_mae FLOAT,
                        val_mae FLOAT,
                        learning_rate FLOAT,
                        epoch_time FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Model checkpoints table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_checkpoints (
                        id SERIAL PRIMARY KEY,
                        experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
                        epoch INTEGER NOT NULL,
                        file_path VARCHAR(500),
                        is_best BOOLEAN DEFAULT FALSE,
                        val_loss FLOAT,
                        model_size_mb FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Predictions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
                        file_id VARCHAR(255) NOT NULL,
                        prediction_data BYTEA,
                        prediction_shape VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Data statistics table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS data_statistics (
                        id SERIAL PRIMARY KEY,
                        dataset_name VARCHAR(255) NOT NULL,
                        family_name VARCHAR(100),
                        file_count INTEGER,
                        sample_count INTEGER,
                        stats JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
    
    def create_experiment(self, name: str, config: Dict, description: str = None) -> int:
        """Create a new experiment"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO experiments (name, description, config)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (name, description, json.dumps(config)))
                
                experiment_id = cur.fetchone()[0]
                conn.commit()
                return experiment_id
    
    def update_experiment_status(self, experiment_id: int, status: str):
        """Update experiment status"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE experiments 
                    SET status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, experiment_id))
                conn.commit()
    
    def log_training_metrics(self, experiment_id: int, metrics: Dict):
        """Log training metrics for an epoch"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO training_metrics 
                    (experiment_id, epoch, train_loss, val_loss, train_mae, val_mae, learning_rate, epoch_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    experiment_id,
                    metrics.get('epoch'),
                    metrics.get('train_loss'),
                    metrics.get('val_loss'),
                    metrics.get('train_mae'),
                    metrics.get('val_mae'),
                    metrics.get('lr'),
                    metrics.get('epoch_time')
                ))
                conn.commit()
    
    def save_checkpoint_info(self, experiment_id: int, epoch: int, file_path: str, 
                           is_best: bool = False, val_loss: float = None, model_size_mb: float = None):
        """Save model checkpoint information"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_checkpoints 
                    (experiment_id, epoch, file_path, is_best, val_loss, model_size_mb)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (experiment_id, epoch, file_path, is_best, val_loss, model_size_mb))
                conn.commit()
    
    def save_predictions(self, experiment_id: int, file_id: str, prediction: np.ndarray):
        """Save prediction results"""
        prediction_bytes = prediction.tobytes()
        prediction_shape = str(prediction.shape)
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO predictions (experiment_id, file_id, prediction_data, prediction_shape)
                    VALUES (%s, %s, %s, %s)
                """, (experiment_id, file_id, prediction_bytes, prediction_shape))
                conn.commit()
    
    def get_experiments(self, limit: int = 50) -> pd.DataFrame:
        """Get list of experiments"""
        with self.get_connection() as conn:
            query = """
                SELECT id, name, description, status, created_at, updated_at,
                       config->>'batch_size' as batch_size,
                       config->>'learning_rate' as learning_rate,
                       config->>'num_epochs' as num_epochs
                FROM experiments 
                ORDER BY created_at DESC 
                LIMIT %s
            """
            return pd.read_sql(query, conn, params=(limit,))
    
    def get_training_history(self, experiment_id: int) -> pd.DataFrame:
        """Get training history for an experiment"""
        with self.get_connection() as conn:
            query = """
                SELECT epoch, train_loss, val_loss, train_mae, val_mae, learning_rate, epoch_time
                FROM training_metrics 
                WHERE experiment_id = %s 
                ORDER BY epoch
            """
            return pd.read_sql(query, conn, params=(experiment_id,))
    
    def get_best_experiments(self, metric: str = 'val_loss', limit: int = 10) -> pd.DataFrame:
        """Get best performing experiments"""
        with self.get_connection() as conn:
            if metric == 'val_loss':
                order = 'ASC'
            else:
                order = 'DESC'
            
            query = f"""
                SELECT e.id, e.name, e.status, e.created_at,
                       MIN(tm.{metric}) as best_{metric},
                       COUNT(tm.epoch) as total_epochs
                FROM experiments e
                LEFT JOIN training_metrics tm ON e.id = tm.experiment_id
                WHERE e.status = 'completed'
                GROUP BY e.id, e.name, e.status, e.created_at
                HAVING MIN(tm.{metric}) IS NOT NULL
                ORDER BY best_{metric} {order}
                LIMIT %s
            """
            return pd.read_sql(query, conn, params=(limit,))
    
    def get_experiment_summary(self, experiment_id: int) -> Dict:
        """Get detailed summary of an experiment"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get experiment details
                cur.execute("""
                    SELECT * FROM experiments WHERE id = %s
                """, (experiment_id,))
                experiment = dict(cur.fetchone() or {})
                
                # Get training metrics summary
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_epochs,
                        MIN(val_loss) as best_val_loss,
                        MAX(epoch) as final_epoch,
                        AVG(epoch_time) as avg_epoch_time
                    FROM training_metrics 
                    WHERE experiment_id = %s
                """, (experiment_id,))
                metrics_summary = dict(cur.fetchone() or {})
                
                # Get checkpoint info
                cur.execute("""
                    SELECT COUNT(*) as checkpoint_count,
                           COUNT(*) FILTER (WHERE is_best) as best_checkpoints
                    FROM model_checkpoints 
                    WHERE experiment_id = %s
                """, (experiment_id,))
                checkpoint_summary = dict(cur.fetchone() or {})
                
                return {
                    'experiment': experiment,
                    'metrics': metrics_summary,
                    'checkpoints': checkpoint_summary
                }
    
    def save_data_statistics(self, dataset_name: str, family_name: str, 
                           file_count: int, sample_count: int, stats: Dict):
        """Save dataset statistics"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO data_statistics 
                    (dataset_name, family_name, file_count, sample_count, stats)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (dataset_name, family_name) 
                    DO UPDATE SET 
                        file_count = EXCLUDED.file_count,
                        sample_count = EXCLUDED.sample_count,
                        stats = EXCLUDED.stats,
                        created_at = CURRENT_TIMESTAMP
                """, (dataset_name, family_name, file_count, sample_count, json.dumps(stats)))
                conn.commit()
    
    def get_data_overview(self) -> pd.DataFrame:
        """Get overview of all datasets"""
        with self.get_connection() as conn:
            query = """
                SELECT dataset_name, family_name, file_count, sample_count, created_at
                FROM data_statistics 
                ORDER BY created_at DESC
            """
            return pd.read_sql(query, conn)
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old experiment data"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM experiments 
                    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    AND status IN ('failed', 'cancelled')
                """, (days,))
                
                deleted_count = cur.rowcount
                conn.commit()
                return deleted_count
    
    def export_experiment_data(self, experiment_id: int) -> Dict:
        """Export all data for an experiment"""
        with self.get_connection() as conn:
            # Get experiment details
            experiment_df = pd.read_sql("""
                SELECT * FROM experiments WHERE id = %s
            """, conn, params=(experiment_id,))
            
            # Get training metrics
            metrics_df = pd.read_sql("""
                SELECT * FROM training_metrics WHERE experiment_id = %s ORDER BY epoch
            """, conn, params=(experiment_id,))
            
            # Get checkpoints
            checkpoints_df = pd.read_sql("""
                SELECT * FROM model_checkpoints WHERE experiment_id = %s ORDER BY epoch
            """, conn, params=(experiment_id,))
            
            return {
                'experiment': experiment_df.to_dict('records')[0] if not experiment_df.empty else {},
                'training_metrics': metrics_df.to_dict('records'),
                'checkpoints': checkpoints_df.to_dict('records')
            }