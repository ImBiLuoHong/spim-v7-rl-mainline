import numpy as np
import torch
from sklearn.metrics import roc_auc_score

class Phase4Evaluator:
    """
    Strict Phase 4.5 Evaluator.
    Calculates metrics based on physical verification of the source.
    """
    def __init__(self):
        self.episodes = []

    def process_episode(self, episode_data):
        """
        Ingest a single episode's result.
        
        Args:
            episode_data (dict):
                - 'success': bool (Did we physically sample the true source?)
                - 'steps': int (Number of interaction steps taken)
                - 'budget': float (Total budget/samples used)
                - 'trajectory': list (Step-by-step logs, optional but good for debugging)
                - 'start_node': int (ID of starting node, for Geodesic calc)
                - 'source_node': int (ID of true source node)
                - 'geodesic_dist': float (Distance from start to source)
        """
        self.episodes.append(episode_data)

    def summarize(self):
        """
        Return the summary metrics for the experiment.
        """
        if not self.episodes:
            return {
                "Success_Rate": 0.0,
                "Predict_Hit@1": 0.0,
                "Exhaustion_Rate": 0.0,
                "Avg_Physical_Time": 0.0,
                "Avg_Samples": 0.0,
                "Avg_Rounds": 0.0,
                "Hard_Success_Rate": 0.0
            }

        n = len(self.episodes)
        
        # 1. Success Rate (Hit@1 by Physical Verification)
        success_count = sum(1 for e in self.episodes if e['success'])
        success_rate = success_count / n
        
        # Audit: Predict Hit@1
        predict_hit_count = sum(1 for e in self.episodes if e.get('predict_hit', False))
        predict_hit_rate = predict_hit_count / n
        
        # Audit: Exhaustion Rate
        exhaustion_rate = sum(1 for e in self.episodes if not e['success']) / n
        
        # 2. Avg Physical Time
        total_steps = sum(e['steps'] for e in self.episodes)
        avg_time = total_steps / n
        
        # Audit: Avg Rounds
        total_rounds = sum(e.get('rounds', 0) for e in self.episodes)
        avg_rounds = total_rounds / n
        
        # 3. Avg Samples (Total nodes revealed)
        total_budget = sum(e['budget'] for e in self.episodes)
        avg_samples = total_budget / n
        
        # 4. Hard Success Rate (Distance > 6)
        hard_episodes = [e for e in self.episodes if e.get('geodesic_dist', 0) > 6]
        if hard_episodes:
            hard_success_count = sum(1 for e in hard_episodes if e['success'])
            hard_success_rate = hard_success_count / len(hard_episodes)
        else:
            hard_success_rate = 0.0
            
        # 5. Hit Surrogate AUROC
        # Episode Level AUROC
        y_true_ep = [1 if e['success'] else 0 for e in self.episodes]
        y_score_ep = [e.get('max_hit_prob', 0.0) for e in self.episodes]
        
        hit_auroc_episode = 0.5
        if len(set(y_true_ep)) > 1:
            try:
                hit_auroc_episode = roc_auc_score(y_true_ep, y_score_ep)
            except:
                pass
        
        # Step Level AUROC
        y_true_step = []
        y_score_step = []
        for e in self.episodes:
            traj = e.get('trajectory_probs', [])
            hits = e.get('trajectory_hits', [])
            if traj and hits:
                y_true_step.extend(hits)
                y_score_step.extend(traj)
        
        hit_auroc_step = 0.5
        if len(y_true_step) > 0 and len(set(y_true_step)) > 1:
            try:
                hit_auroc_step = roc_auc_score(y_true_step, y_score_step)
            except:
                pass
            
        return {
            "Success_Rate": success_rate,
            "Predict_Hit@1": predict_hit_rate,
            "Exhaustion_Rate": exhaustion_rate,
            "Avg_Physical_Time": avg_time,
            "Avg_Samples": avg_samples,
            "Avg_Rounds": avg_rounds,
            "Hard_Success_Rate": hard_success_rate,
            "Num_Hard_Samples": len(hard_episodes),
            "hit_auroc_episode": hit_auroc_episode,
            "hit_auroc_step": hit_auroc_step
        }

    def reset(self):
        self.episodes = []
