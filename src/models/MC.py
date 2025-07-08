import os
import pandas as pd
import numpy as np
import pickle
import json
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import your existing modules
from utils.experiment_utils import get_participant_ids, load_image_data
from utils.experiment_data import filter_data
import utils.style as stl

import networkx as nx
from collections import defaultdict, Counter
import json
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import os

class MarkovChainGenerator:
    """
    Class to generate and analyze Markov Chains from gaze-driven segmentation data
    """
    
    def __init__(self, img_id, img_type, results_dir, data_dir, filename_labeled_masks):
        """
        Initialize the MarkovChainGenerator with image and experiment details.
        """
        
        self.img_id = img_id
        self.img_type = img_type
        self.participant_chains = {}
        self.transition_matrices = {}
        self.label_to_indx = {}
        self.indx_to_label = {}
        self.RESULTS_DIR = results_dir
        self.DATA_DIR = data_dir
        self.all_labels = []
        self.labeled_data = self.load_labeled_masks(filename_labeled_masks)

    def load_labeled_masks(self, filename):
        """Load the labeled mask data for all participants"""
        labeling_dir = os.path.join(self.RESULTS_DIR, "masks_gaze_driven", "best_mask_labeling")
        labeled_file = os.path.join(labeling_dir, f"{filename}")
        
        if os.path.exists(labeled_file):
            self.labeled_data = pd.read_csv(labeled_file)
        else:
            # If no labeled data exists, create a placeholder
            print(f"Warning: No labeled mask data found at {labeled_file}")
            self.labeled_data = pd.DataFrame()
            
        self.all_labels = self.labeled_data['best_label'].unique().tolist()
        
        for i in range(len(self.all_labels)):
            self.indx_to_label[i] = self.all_labels[i]
            self.label_to_indx[self.all_labels[i]] = i

        return self.labeled_data

    def extract_fixation_sequence(self, participant_id):
        """
        Extract the sequence of fixated objects/labels for a participant
        """
        # Filter data for this participant and image
        participant_data = self.labeled_data[self.labeled_data['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            print(f"No data found for participant {participant_id}")
            return []
            
        # Get the corresponding labels for each fixation
        fixation_labels = participant_data['best_label'].tolist()
            
        return fixation_labels
    
    def generate_markov_chain(self, participant_id):
        """
        Generate a Markov chain from the fixation sequence of a participant
        """
        sequence = self.extract_fixation_sequence(participant_id)
        
        transition_matrix = pd.DataFrame(np.zeros((len(self.all_labels), len(self.all_labels))),
                                    index=self.all_labels, columns=self.all_labels)
        
        if len(sequence) < 2:
            return pd.DataFrame(), pd.DataFrame()

        # Count transitions
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            transition_matrix.loc[current_state, next_state] += 1
            
        transition_matrix_prob = transition_matrix.copy()

        # Convert to probabilities
        transition_matrix_prob = transition_matrix_prob.div(transition_matrix_prob.sum(axis=1), axis=0)
        
        # fill NaN values with 0
        transition_matrix_prob.fillna(0, inplace=True)
            
        # Store the chain
        self.participant_chains[int(participant_id)] = {
            'sequence': sequence,
            'transitions': transition_matrix,
            'probabilities': transition_matrix_prob
        }

        return transition_matrix, transition_matrix_prob

    def generate_all_chains(self, participant_ids):
        """Generate Markov chains for all participants"""
        print(f"Generating Markov chains for {len(participant_ids)} participants...")
        
        for participant_id in participant_ids:
            try:
                transitions, probs = self.generate_markov_chain(participant_id)
                print(f"Generated chain for participant {participant_id}: {len(transitions)} states")
            except Exception as e:
                print(f"Error processing participant {participant_id}: {e}")
                
        return self.participant_chains
    
    def create_aggregate_chain(self):
        """
        Create an aggregate Markov chain from all participants
        """
        all_transitions = defaultdict(lambda: defaultdict(int))
        
        # Aggregate all transitions
        for participant_id, chain_data in self.participant_chains.items():
            transitions = chain_data['transitions']
            for current_state, next_states in transitions.items():
                for next_state, count in next_states.items():
                    all_transitions[current_state][next_state] += count
                    
        # Convert to probabilities
        aggregate_probs = {}
        for current_state, next_states in all_transitions.items():
            total_transitions = sum(next_states.values())
            aggregate_probs[current_state] = {
                next_state: count / total_transitions 
                for next_state, count in next_states.items()
            }
            
        self.aggregate_chain = {
            'transitions': dict(all_transitions),
            'probabilities': aggregate_probs
        }
        
        return self.aggregate_chain
    
    def visualize_markov_chain(self, chain_data=None, participant_id=None, save_path=None):
        """
        Visualize the Markov chain as a directed graph
        """
        if chain_data is None:
            if participant_id:
                chain_data = self.participant_chains[participant_id]
            else:
                chain_data = self.aggregate_chain
                
        if 'probabilities' not in chain_data:
            print("No probability data found for visualization")
            return
            
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for current_state, next_states in chain_data['probabilities'].items():
            for next_state, prob in next_states.items():
                if prob > 0.1:  # Only show significant transitions
                    G.add_edge(current_state, next_state, weight=prob)
                    
        if len(G.nodes()) == 0:
            print("No significant transitions to visualize")
            return
            
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, alpha=0.7)
        
        # Draw edges with varying thickness based on probability
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                              alpha=0.6, edge_color='gray', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Add edge labels (probabilities)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        title = f"Markov Chain - "
        if participant_id:
            title += f"Participant {participant_id}"
        else:
            title += "Aggregate (All Participants)"
            
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def calculate_chain_metrics(self):
        """
        Calculate various metrics for the Markov chains
        """
        metrics = {}
        
        for participant_id, chain_data in self.participant_chains.items():
            sequence = chain_data['sequence']
            transitions = chain_data['transitions']
            
            # Number of unique states
            unique_states = len(set(sequence))
            
            # Number of transitions
            total_transitions = len(sequence) - 1
            
            # Number of unique transitions
            unique_transitions = sum(len(next_states) for next_states in transitions.values())
            
            # Entropy of the chain
            entropy = self.calculate_entropy(chain_data['probabilities'])
            
            metrics[participant_id] = {
                'unique_states': unique_states,
                'total_transitions': total_transitions,
                'unique_transitions': unique_transitions,
                'entropy': entropy,
                'sequence_length': len(sequence)
            }
            
        self.chain_metrics = metrics
        return metrics
    
    def calculate_entropy(self, probabilities):
        """Calculate the entropy of transition probabilities"""
        entropy = 0
        for current_state, next_states in probabilities.items():
            for prob in next_states.values():
                if prob > 0:
                    entropy -= prob * np.log2(prob)
        return entropy
    
    def save_chains(self, output_dir=None):
        """Save all generated chains to files"""
        if output_dir is None:
            output_dir = os.path.join(self.RESULTS_DIR, "masks_gaze_driven", "markov_chain")

        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual chains
        chains_file = os.path.join(output_dir, f"{self.img_type}_{self.img_id}_markov_chains.pkl")
        with open(chains_file, 'wb') as f:
            pickle.dump(self.participant_chains, f)
            
        # Save aggregate chain
        if hasattr(self, 'aggregate_chain'):
            aggregate_file = os.path.join(output_dir, f"{self.img_type}_{self.img_id}_aggregate_chain.pkl")
            with open(aggregate_file, 'wb') as f:
                pickle.dump(self.aggregate_chain, f)
                
        # Save metrics
        if hasattr(self, 'chain_metrics'):
            metrics_file = os.path.join(output_dir, f"{self.img_type}_{self.img_id}_chain_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.chain_metrics, f, indent=2)
                
        print(f"Saved chains and metrics to {output_dir}")
        
    def analyze_semantic_alignment(self, knowledge_graph_data=None):
        """
        Analyze alignment between Markov chain transitions and semantic relationships
        """
        if knowledge_graph_data is None:
            # Try to load knowledge graph data
            kg_dir = os.path.join(self.RESULTS_DIR, "masks_gaze_driven", "knowledge_graphs")
            kg_file = os.path.join(kg_dir, f"{self.img_type}_{self.img_id}_knowledge_graph.pkl")
            
            if os.path.exists(kg_file):
                with open(kg_file, 'rb') as f:
                    knowledge_graph_data = pickle.load(f)
            else:
                print("No knowledge graph data found for semantic alignment analysis")
                return {}
                
        alignment_scores = {}
        
        for participant_id, chain_data in self.participant_chains.items():
            # Count transitions that align with semantic relationships
            aligned_transitions = 0
            total_transitions = 0
            
            for current_state, next_states in chain_data['transitions'].items():
                for next_state, count in next_states.items():
                    total_transitions += count
                    
                    # Check if this transition exists in the knowledge graph
                    if self.has_semantic_relationship(current_state, next_state, knowledge_graph_data):
                        aligned_transitions += count
                        
            alignment_score = aligned_transitions / total_transitions if total_transitions > 0 else 0
            alignment_scores[participant_id] = {
                'aligned_transitions': aligned_transitions,
                'total_transitions': total_transitions,
                'alignment_score': alignment_score
            }
            
        return alignment_scores
    
    def has_semantic_relationship(self, state1, state2, kg_data):
        """
        Check if two states have a semantic relationship in the knowledge graph
        """
        # This is a placeholder - implement based on your KG structure
        if not kg_data:
            return False
            
        # Add your logic to check semantic relationships here
        # This depends on how your knowledge graph is structured
        return False

