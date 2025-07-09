import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import re

from src.core.ImageData import ImageData
from src.core.Mask import Mask
from src.models.LLM import LLM

class KnowledgeGraph:
    """
    Enhanced Knowledge Graph implementation for scene understanding and spatial reasoning
    """
    
    def __init__(self, llm: LLM, image_data: ImageData, output_dir: str = "outputs"):
        self.llm = llm
        self.image_data = image_data
        self.output_dir = output_dir
        self.graph = nx.DiGraph()
        self.triples = []
        self.bbox_info = {}
        self.scene_description = imageData.description
        self.relations_count = defaultdict(int)
        
        # Ensure output directories exist
        self.results_dir = os.path.join(output_dir, "results")
        self.kg_dir = os.path.join(output_dir, "knowledge_graphs")
        self.matrix_dir = os.path.join(output_dir, "knowledge_matrices")
        
        for directory in [self.results_dir, self.kg_dir, self.matrix_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def create_mask_dict(self, debug: bool = True) -> Dict:
        """
        Create a dictionary of mask bounding boxes and labels
        """




        
        bbox_dict = {}
        
        for i, mask in enumerate(masks):
            if hasattr(mask, 'x_min') and hasattr(mask, 'labels') and mask.labels:
                bbox_dict[f"mask_{i}"] = {
                    'label': mask.labels[0] if isinstance(mask.labels, list) else mask.labels,
                    'bbox': {
                        'x_min': mask.x_min,
                        'x_max': mask.x_max,
                        'y_min': mask.y_min,
                        'y_max': mask.y_max
                    },
                    'area': mask.area if hasattr(mask, 'area') else 0,
                    'score': mask.score if hasattr(mask, 'score') else 0
                }
        
        if debug:
            print(f"Created bbox dictionary with {len(bbox_dict)} masks")
        
        return bbox_dict
    
    def generate_spatial_analysis(self, masks: List[Mask], debug: bool = True) -> str:
        """
        Generate spatial relationship analysis using LLM
        """
        bbox_dict = self.create_mask_dict(image_data, masks, debug)
        self.bbox_info = bbox_dict
        
        spatial_prompt = f"""Consider the following dictionary that contains a set of objects in the scene. 
        Analyze the spatial relationships between these objects based on their positions. 
        Analyze the properties of the objects (ex: shape, color, complexity, functionality, etc.).
        Analyze the functionality of the objects.
        These are the bounding boxes: {bbox_dict}
        
        Scene context: {scene_description}"""
        
        try:
            # Use the LLM's describe_scene method with custom prompt
            analysis = self.llm.describe_scene(image_data, custom_prompt=spatial_prompt)
            
            if debug:
                print("Spatial analysis completed")
            
            return analysis
            
        except Exception as e:
            print(f"Error in spatial analysis: {e}")
            return f"Error: {str(e)}"
    
    def generate_triples(self, image_data: ImageData, masks: List[Mask], 
                        scene_description: str, debug: bool = True) -> str:
        """
        Generate knowledge graph triples from scene analysis
        """
        # First get spatial analysis
        bbox_info = self.generate_spatial_analysis(image_data, masks, scene_description, debug)
        
        triples_prompt = f"""Please generate triples to build a scene graph for the following scene. 
        Use your visual understanding and prior knowledge to enrich the scene graph. 
        The triples should cover the following types of relationships:
        1. Spatial Relationships: Describing the position and arrangement of objects in the scene.
        2. Functional Relationships: Describing the purpose or use of the objects.
        3. Semantic Relationships: Describing meaningful connections between objects.
        4. Property Relationships: Describing the attributes of objects (e.g., shape, color, texture, material).
        5. Contextual Relationships: Describing the relationships between objects and the context of the scene.
        6. Distance Relationships: Describing the distance between objects.
        7. Size Relationships: Describing the size of objects.
        
        Scene description: {scene_description}
        Spatial analysis: {bbox_info}
        
        Please return only the triples in the format (subject, RELATION, object). Avoid including numbers in the triples.
        
        Example:
            (Slippers, are_on, Rug)
            (Table, is_under, Pendant_light)
            (Sofa, is_near, Window)
            (Teapot, has_color, White)
            (Sofa, has_shape, Rectangular)
            (Rug, is_used_for, Adding_warmth)
            (Cup, is_used_for, Drinking)
            (Cabinets, have_color, White)
            
        Please generate similar triples for the given scene. Return only the triples in the format (subject, RELATION, object)."""
        
        try:
            # Use the LLM's describe_scene method with custom prompt
            triples_text = self.llm.describe_scene(image_data, custom_prompt=triples_prompt)
            
            # Clean up the triples text
            triples_text = self._clean_triples_text(triples_text)
            
            if debug:
                print("Triples generation completed")
                print(f"Generated triples: {triples_text[:200]}...")
            
            return triples_text
            
        except Exception as e:
            print(f"Error in triples generation: {e}")
            return f"Error: {str(e)}"
    
    def clean_triples_text(self, triples_text: str) -> str:
        """
        Clean and normalize triples text
        """
        # Remove extra whitespace and newlines
        triples_text = triples_text.replace("\n", "")
        triples_text = re.sub(r'\s+', ' ', triples_text)
        
        # Normalize parentheses spacing
        triples_text = triples_text.replace(")  ", ")")
        triples_text = triples_text.replace(")   (", ")  (")
        triples_text = triples_text.replace(")(", ")  (")
        
        return triples_text.strip()
    
    def parse_triples(self, triples_text: str) -> List[Tuple[str, str, str]]:
        """
        Parse triples text into structured format
        """
        # Split triples
        triples_list = triples_text.split(')  (')
        
        if len(triples_list) == 1:
            triples_list = triples_list[0].split(')(')
        
        parsed_triples = []
        
        for triple in triples_list:
            # Clean triple
            triple = triple.replace("(", "").replace(")", "")
            tokens = triple.split(', ')
            
            if len(tokens) >= 3:
                subject = tokens[0].strip()
                predicate = tokens[1].strip().upper()
                obj = tokens[2].strip()
                
                parsed_triples.append((subject, predicate, obj))
                self.relations_count[predicate] += 1
        
        self.triples = parsed_triples
        return parsed_triples
    
    def build_graph(self, triples: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """
        Build NetworkX graph from triples
        """
        self.graph = nx.DiGraph()
        
        for subject, predicate, obj in triples:
            self.graph.add_edge(subject, obj, label=predicate)
        
        return self.graph
    
    def plot_knowledge_graph(self, image_id: str, figsize: Tuple[int, int] = (12, 8),
                           save: bool = True, show: bool = True) -> None:
        """
        Plot and save knowledge graph visualization
        """
        if not self.graph.nodes():
            print("No graph to plot. Build graph first.")
            return
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=1.2, iterations=100)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Draw nodes and edges
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", 
               node_size=3000, font_size=10, font_weight="bold", edge_color="gray")
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, 
                                    font_color="red", font_size=8)
        
        plt.title(f"Knowledge Graph for Image {image_id}", fontsize=14)
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.kg_dir, f"{image_id}_knowledge_graph.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            
            # Save graph as pickle
            pickle_path = os.path.join(self.kg_dir, f"{image_id}_knowledge_graph.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.graph, f)
            
            print(f"Knowledge graph saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_relationship_matrix(self, image_id: str, cmap_color: str = "viridis",
                               figsize: Tuple[int, int] = (12, 10), save: bool = True, 
                               show: bool = True) -> None:
        """
        Plot relationship matrix heatmap
        """
        if not self.triples:
            print("No triples to plot. Generate triples first.")
            return
        
        # Get unique entities and relations
        unique_entities = defaultdict(int)
        relations = set()
        
        for subject, relation, obj in self.triples:
            unique_entities[subject] += 1
            unique_entities[obj] += 1
            relations.add(relation)
        
        # Create color mapping
        cmap = sns.color_palette(cmap_color, len(relations))
        entities = sorted(unique_entities.keys())
        relation_colors = {rel: cmap[i] for i, rel in enumerate(sorted(relations))}
        
        # Create directional heatmap
        directional_heatmap_data = pd.DataFrame(0, index=entities, columns=entities)
        
        for subject, relation, obj in self.triples:
            if subject in entities and obj in entities:
                directional_heatmap_data.at[subject, obj] = list(relation_colors.keys()).index(relation) + 1
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        mask = directional_heatmap_data == 0
        
        ax = sns.heatmap(directional_heatmap_data, annot=False, cmap=cmap_color, 
                        cbar=False, mask=mask, linecolor='white', linewidth=0.5)
        
        # Add grid lines
        ax.hlines(range(len(directional_heatmap_data) + 1), *ax.get_xlim(), 
                 color="#e5e7e9", linewidth=1)
        ax.vlines(range(len(directional_heatmap_data) + 1), *ax.get_ylim(), 
                 color="#e5e7e9", linewidth=1)
        
        plt.title(f'Relationship Matrix for Image {image_id}', fontsize=14)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.xlabel("Object (Target)", fontsize=12)
        plt.ylabel("Object (Source)", fontsize=12)
        
        # Create legend
        for label, color in relation_colors.items():
            label_with_count = f"{label} ({self.relations_count[label]})"
            plt.plot([], [], label=label_with_count, color=color, linewidth=15)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), 
                  title="Relationships", fontsize=10)
        
        if save:
            output_path = os.path.join(self.matrix_dir, f"{image_id}_knowledge_matrix.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Relationship matrix saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_adjacency_matrix(self, image_id: str, save: bool = True) -> pd.DataFrame:
        """
        Generate and save adjacency matrix
        """
        if not self.graph.nodes():
            print("No graph available. Build graph first.")
            return pd.DataFrame()
        
        # Extract edges with labels
        edges = []
        for source, target, data in self.graph.edges(data=True):
            weight = data.get('label', 1)
            edges.append({'Source': source, 'Target': target, 'Weight': weight})
        
        edges_df = pd.DataFrame(edges)
        
        if edges_df.empty:
            return pd.DataFrame()
        
        # Create adjacency matrix
        nodes = list(self.graph.nodes())
        matrix_df = pd.DataFrame(0, columns=nodes, index=nodes)
        
        for _, row in edges_df.iterrows():
            matrix_df.loc[row['Source'], row['Target']] = 1  # Binary matrix
        
        if save:
            output_path = os.path.join(self.matrix_dir, f"{image_id}_adjacency_matrix.csv")
            matrix_df.to_csv(output_path)
            print(f"Adjacency matrix saved to {output_path}")
        
        return matrix_df
    
    def triple_to_sentence(self, triple: Tuple[str, str, str]) -> str:
        """
        Convert a triple to natural language sentence
        """
        subject, predicate, obj = triple
        
        # Handle different predicate formats
        if predicate.startswith("IS_"):
            predicate = predicate.replace("IS_", "is ")
        elif predicate.startswith("ARE_"):
            predicate = predicate.replace("ARE_", "are ")
        elif predicate.startswith("HAS_"):
            predicate = predicate.replace("HAS_", "has ")
        elif predicate.startswith("HAVE_"):
            predicate = predicate.replace("HAVE_", "have ")
        
        predicate = predicate.replace("_", " ").lower()
        
        return f"The {subject.lower()} {predicate} the {obj.lower()}."
    
    def ask_reasoning_question(self, query: str, temperature: float = 0.0) -> str:
        """
        Answer reasoning questions based on the knowledge graph
        """
        if not self.triples:
            return "No knowledge graph available. Generate triples first."
        
        # Convert triples to natural language
        sentences = [self.triple_to_sentence(triple) for triple in self.triples]
        knowledge_base = " ".join(sentences)
        
        reasoning_prompt = f"""Consider the following knowledge base: {knowledge_base}
        
        Question: {query}
        
        Please answer based on the spatial and semantic relationships in the knowledge base.
        Do not use colors, properties, or attributes in the locations."""
        
        try:
            temp_image = ImageData("dummy_path.jpg")  # This won't be used for vision
            answer = self.llm.describe_scene(temp_image, custom_prompt=reasoning_prompt)
            
            # Clean answer
            answer = answer.replace("\n", " ").replace("*", "").strip()
            
            return answer
            
        except Exception as e:
            print(f"Error in reasoning: {e}")
            return f"Error: {str(e)}"
    
    def process_image_complete(self, image_data: ImageData, masks: List[Mask], 
                             scene_description: str = None, debug: bool = True) -> Dict:
        """
        Complete knowledge graph processing pipeline
        """
        print(f"Processing knowledge graph for image {image_data.ID}")
        
        # Get scene description if not provided
        if scene_description is None:
            scene_description = self.llm.describe_scene(image_data)
        
        self.scene_description = scene_description
        
        # Generate triples
        triples_text = self.generate_triples(image_data, masks, scene_description, debug)
        
        # Parse triples
        parsed_triples = self.parse_triples(triples_text)
        
        # Build graph
        graph = self.build_graph(parsed_triples)
        
        # Create visualizations
        image_id = str(image_data.ID)
        self.plot_knowledge_graph(image_id, save=True, show=debug)
        self.plot_relationship_matrix(image_id, save=True, show=debug)
        
        # Get adjacency matrix
        adj_matrix = self.get_adjacency_matrix(image_id, save=True)
        
        # Save complete results
        results = {
            'image_id': image_data.ID,
            'scene_description': scene_description,
            'triples_text': triples_text,
            'parsed_triples': parsed_triples,
            'bbox_info': self.bbox_info,
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges()),
            'relations_count': dict(self.relations_count)
        }
        
        results_path = os.path.join(self.results_dir, f"kg_results_{image_id}.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        if debug:
            print(f"Knowledge graph processing complete for image {image_data.ID}")
            print(f"Generated {len(parsed_triples)} triples")
            print(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        
        return results
    
    def save_state(self, filepath: str) -> None:
        """
        Save the complete knowledge graph state
        """
        state = {
            'graph': self.graph,
            'triples': self.triples,
            'bbox_info': self.bbox_info,
            'scene_description': self.scene_description,
            'relations_count': dict(self.relations_count)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str) -> None:
        """
        Load knowledge graph state from file
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.graph = state.get('graph', nx.DiGraph())
        self.triples = state.get('triples', [])
        self.bbox_info = state.get('bbox_info', {})
        self.scene_description = state.get('scene_description', "")
        self.relations_count = defaultdict(int, state.get('relations_count', {}))

