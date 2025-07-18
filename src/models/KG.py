from nt import system
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



from src.utils.data_utils import *

class KnowledgeGraph:
    """
    Enhanced Knowledge Graph implementation for scene understanding and spatial reasoning
    """
    
    def __init__(self, llm: LLM, image_data: ImageData, output_dir: str = "outputs"):
        self.llm = llm
        self.image_data = image_data

        print("Disambiguating scene masks...")
        labels = self.image_data.get_scene_masks()

        self.output_dir = output_dir
        self.graph = nx.DiGraph()
        self.triples = []
        self.triples_raw = None             # for debug
        self.triples_cleaned = None         # for debug
        self.bbox_info = {}
        self.scene_description = image_data.description
        self.relations_count = defaultdict(int)
        


        # Ensure output directories exist
        self.rankings_dir = os.path.join(output_dir, "rankings")
        self.img_rankings = os.path.join(self.rankings_dir, f"IMG_{self.image_data.ID}_{self.image_data.img_type}")
        self.kg_dir = os.path.join(output_dir, "KnowledgeGraph")
        self.matrix_dir = os.path.join(self.kg_dir, "knowledge_matrices")
        self.triples_raw_dir = os.path.join(self.kg_dir, "triples_raw")
        self.triples_cleaned_dir = os.path.join(self.kg_dir, "triples_cleaned")
        self.triples_final_dir = os.path.join(self.kg_dir, "triples_final")
        self.scene_descr_dir = os.path.join(self.output_dir, "SceneDescriptions", llm.provider.lower() + "_results")

        for directory in [self.img_rankings, self.kg_dir, self.matrix_dir, self.triples_raw_dir, self.triples_cleaned_dir, self.triples_final_dir, self.scene_descr_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_txt_file( self, filepath ):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading text file: {str(e)}")
            return None

    def save_txt_file(self, filepath: str, content: str) -> None:
        try:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(content)
        except Exception as e:
            print(f"Error writing text file: {str(e)}")


    def load_triples(self, filepath: str) -> List[Tuple[str, str, str]]:
        """
        Load and parse triples from a text file.
        Handles markdown table format and returns list of (subject, relation, object) tuples.
        """

        parsed_triples = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return []
            
        for line in lines:
            line = line.strip()

            tokens = line.split(", ")
                
            if len(tokens) == 3:
                subject = tokens[0].strip()
                relation = tokens[1].strip().upper()
                obj = tokens[2].strip()
                
                if subject and relation and obj:
                    parsed_triples.append((subject, relation, obj))

            else:
                print("[ERROR]The triple is not well formated")
                print(line)
                return []
        
        # Store the triples in the instance
        self.triples = parsed_triples
        
        print(f"Successfully loaded {len(parsed_triples)} triples from {filepath}")
        return parsed_triples
        
    def create_mask_dict(self, masks, debug: bool = True) -> Dict:
        """
        Create a dictionary of mask bounding boxes and labels
        """

        #for i, mask in enumerate(masks):
        #    if hasattr(mask, 'x_min') and hasattr(mask, 'labels') and mask.labels:
        #        self.bbox_info[f"mask_{i}"] = {
        #            'label': mask.most_freq_label,
        #            'label_prob': mask.most_freq_prob,
        #            'bbox': {
        #                'x_min': int(mask.x_min),
        #                'x_max': int(mask.x_max),
        #                'y_min': int(mask.y_min),
        #                'y_max': int(mask.y_max)
        #            }
        #        }
        

        # only consider unique masks
        self.bbox_info = self.image_data.mask_labels

        if debug:
            print(f"Created bbox dictionary with {len(self.bbox_info)} masks")

    
    def generate_spatial_analysis(self, masks: List[Mask], debug: bool = True) -> str:
        """
        Generate spatial relationship analysis using LLM
        """
        # update the self.bbox_info dictionary with mask info
        self.create_mask_dict(masks, debug)
        
        spatial_prompt = f"""Consider the following dictionary that contains a set of objects in the scene. 
        Analyze the spatial relationships between these objects based on their positions. 
        Analyze the properties of the objects (ex: shape, color, complexity, functionality, etc.).
        Analyze the functionality of the objects.
        These are the bounding boxes: {self.bbox_info}
        
        Scene context: {self.scene_description}"""

        if debug:
            print("Debug mode enabled-----------------------------------------")
            print("Prompt for Spatial Analysis:")
            print(spatial_prompt)
            print("------------------------------------------------------------\n\n")
        
        try:
            # Use the LLM's describe_scene method with custom prompt
            analysis = self.llm.describe_scene(self.image_data, custom_prompt=spatial_prompt)
            
            return analysis
            
        except Exception as e:
            print(f"Error in spatial analysis: {e}")
            return f"Error: {str(e)}"
    
    def generate_triples(self, debug: bool = True) -> str:
        """
        Generate knowledge graph triples from scene analysis
        """

        masks = self.image_data.masks

        # First get spatial analysis
        bbox_info = self.generate_spatial_analysis(masks, debug)
        
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
        
        Scene description: {self.scene_description}
        Spatial analysis: {bbox_info}
        
        Please return only the triples in the format (subject, RELATION, object). Avoid including numbers in the triples. 
        The subject and the object MUST be a SINGLE word. The relation MUST contain a VERB.
        
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

        if debug:
            print("Debug mode enabled-----------------------------------------")
            print("Prompt for Tripple Extraction:")
            print(triples_prompt)
            print("------------------------------------------------------------\n\n")

        try:
            # Use the LLM's describe_scene method with custom prompt
            system_prompt = "You are an AI Assistant with expertise in extracting knowledge graph triples from textual scene descriptions."
            self.triples_raw = self.llm.text_query( query=triples_prompt, system_prompt = system_prompt )

            # save the triples into a text file
            path = os.path.join(self.triples_raw_dir, f"IMG_{self.image_data.ID}_{self.image_data.img_type}_triples_raw.txt")
            with open(path, "w") as f:
                f.write(self.triples_raw)
            
            # Clean up the triples text
            self.triples_cleaned = self.clean_triples_text(self.triples_raw)

            path = os.path.join(self.triples_cleaned_dir, f"IMG_{self.image_data.ID}_{self.image_data.img_type}_triples_cleaned.txt")
            with open(path, "w") as f:
                f.write(self.triples_cleaned)
            
            if debug:
                print("Triples generation completed")
                print(f"Generated triples: {self.triples_cleaned[:200]}...")
            
            return self.triples_cleaned

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
        triples_text = triples_text.replace(")", ")\n")
        
        return triples_text.strip()
    
    def parse_triples(self, triples_text: str) -> List[Tuple[str, str, str]]:
        """
        Parse triples text into structured format.
        Handles numbered list format and various LLM output variations.
        """
        parsed_triples = []
        
        # Split into lines and process each line
        lines = triples_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering (e.g., "1. ", "42. ", etc.)
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Extract content within parentheses using regex
            # This handles cases where there might be extra spaces or formatting
            match = re.search(r'\(([^)]+)\)', line)
            if not match:
                continue
                
            triple_content = match.group(1)
            
            # Split by comma and clean up
            tokens = [token.strip() for token in triple_content.split(',')]
            
            if len(tokens) >= 3:
                subject = tokens[0].strip()
                predicate = tokens[1].strip().upper()
                # Handle cases where object might contain commas
                obj = ', '.join(tokens[2:]).strip()
                
                # Clean up any remaining quotes or extra whitespace
                subject = subject.strip('"\'')
                predicate = predicate.strip('"\'')
                obj = obj.strip('"\'')
                
                if subject and predicate and obj:
                    parsed_triples.append((subject, predicate, obj))
                    if hasattr(self, 'relations_count'):
                        self.relations_count[predicate] += 1
        
        if hasattr(self, 'triples'):
            self.triples = parsed_triples

        path = os.path.join(self.triples_final_dir,  f"IMG_{self.image_data.ID}_{self.image_data.img_type}_triples.txt")
        with open(path, "w") as f:
            f.write("\n".join(f"{s}, {p}, {o}" for s, p, o in parsed_triples))


        return parsed_triples

    def rank_locations_by_likelihood(self, target: str, num_trials: int = 50, num_loc_sample: int = 10, save_dir: Optional[str] = None, debug: bool = True):
        """
        For a given target and list of locations, run multiple trials where the LLM ranks the most and least likely locations.
        Returns two dictionaries: most_likely and least_likely.
        """
        most_likely = {}
        least_likely = {}

        mask_labels = list(self.image_data.mask_labels.keys())

        for trial in range(1, num_trials + 1):
            sampled_locations = random.sample(mask_labels, num_loc_sample)
            query = (
                f"Given these locations {sampled_locations} which are the MOST and LEAST likely places to find {target}? "
                "1. Provide a ranked list of THREE locations for each. "
                "2. Ensure you ALWAYS provide a ranked list of THREE locations for MOST LIKELY. "
                "3. Ensure you ALWAYS provide a ranked list of THREE locations for LEAST LIKELY. "
                "4. Do not use conjunction e.g. AND, OR, NOT, WITH, etc. "
                "5. Do not use quotation marks or string quotes. "
                "6. Do not use attributes with the locations. "
                "7. Do not use Solution or Storage in the locations. "
                "8. Do not use Properties in the locations. "
                "9. Do not use COLORS in the locations. "
                "10. The same location cannot be in both lists. "
                "11. The output should have the format LIKEST: [location1, location2, location3]; UNLIKELIEST: [location1, location2, location3]"
            )

            answer = self.llm.query_knowledge_graph(
                query=query,
                triples=triples,
                temperature=self.llm.temperature
            )

            # Parse the answer
            most, least = self._parse_likelihood_answer(answer)
            most_likely[trial] = most
            least_likely[trial] = least

            if debug:
                print(f"Trial {trial}:")
                print(f"\tMost likely: {most}")
                print(f"\tLeast likely: {least}\n")

        if self.img_rankings:
            with open(os.path.join(self.img_rankings, f"IMG_{self.image_data.ID}_{self.image_data.img_type}_most_likely.pkl"), "wb") as f:
                pickle.dump(most_likely, f)
            with open(os.path.join(save_dir, f"IMG_{self.image_data.ID}_{self.image_data.img_type}_least_likely.pkl"), "wb") as f:
                pickle.dump(least_likely, f)

        return most_likely, least_likely
    
    def _parse_likelihood_answer(self, answer: str):
        """
        Parse the LLM's answer to extract the most and least likely locations.
        """
        # Example expected format:
        # LIKEST: [location1, location2, location3]; UNLIKELIEST: [location1, location2, location3]
        most, least = [], []
        try:
            most_match = re.search(r"LIKEST:\s*\[([^\]]+)\]", answer)
            least_match = re.search(r"UNLIKELIEST:\s*\[([^\]]+)\]", answer)
            if most_match:
                most = [loc.strip() for loc in most_match.group(1).split(",")]
            if least_match:
                least = [loc.strip() for loc in least_match.group(1).split(",")]
        except Exception as e:
            print(f"Error parsing answer: {e}")
        return most, least
    
    def build_graph(self) -> nx.DiGraph:
        """
        Build NetworkX graph from triples
        """
        self.graph = nx.DiGraph()
        
        for subject, predicate, obj in self.triples:
            self.graph.add_edge(subject, obj, label=predicate.upper())
        
        return self.graph
    
    def plot_knowledge_graph(self, figsize: Tuple[int, int] = (16, 12),
                        save: bool = True, show: bool = True, 
                        layout_algorithm: str = "spring", 
                        iterations: int = 2000, k_param = 3.0, scale = 2.0, min_distance = 0.3) -> None:
        """
        Plot and save professional knowledge graph visualization suitable for research publications
        
        Args:
            figsize: Figure size tuple
            save: Whether to save the plot
            show: Whether to display the plot
            layout_algorithm: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'shell')
            iterations: Number of iterations for spring layout
        """
        
        image_id = self.image_data.ID

        if not self.graph.nodes():
            print("No graph to plot. Build graph first.")
            return
        
        # Create figure with high DPI for publication quality
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        # Choose layout algorithm based on graph size and structure
        num_nodes = len(self.graph.nodes())
        
        if layout_algorithm == "spring":
            # Use spring layout with optimized parameters for better node separation
            pos = nx.spring_layout(
                self.graph, 
                k=k_param/np.sqrt(num_nodes),  # Optimal distance between nodes
                iterations=iterations,
                weight=None,
                scale=scale,
                center=(0, 0),
                dim=2,
                seed=42  # For reproducible layouts
            )
        elif layout_algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph, scale=scale)
        elif layout_algorithm == "circular":
            pos = nx.circular_layout(self.graph, scale=scale)
        elif layout_algorithm == "shell":
            pos = nx.shell_layout(self.graph, scale=scale)
        else:
            pos = nx.spring_layout(self.graph, k=k_param/np.sqrt(num_nodes), iterations=iterations)
        
        # Fine-tune positions to avoid overlaps
        pos = self._adjust_node_positions(pos, min_distance=min_distance)
        
        # Calculate node sizes based on degree centrality
        node_degrees = dict(self.graph.degree())
        max_degree = max(node_degrees.values()) if node_degrees else 1
        node_sizes = [3000 + (node_degrees.get(node, 0) / max_degree) * 2000 
                    for node in self.graph.nodes()]
        
        # Create color scheme for different node types (if applicable)
        node_colors = self._get_node_colors()
        
        # Draw edges with improved styling
        edge_weights = []
        edge_colors = []
        edge_styles = []
        
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('label', 'UNKNOWN')
            edge_weights.append(self._get_edge_weight(relation))
            edge_colors.append(self._get_edge_color(relation))
            edge_styles.append(self._get_edge_style(relation))
        
        # Draw edges with varying thickness and style
        for i, (u, v, data) in enumerate(self.graph.edges(data=True)):
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=[(u, v)],
                width=edge_weights[i],
                edge_color=edge_colors[i],
                style=edge_styles[i],
                alpha=0.7,
                arrows=True,
                arrowsize=25,
                arrowstyle='->',
                connectionstyle="arc3,rad=0.1",  # Curved edges to avoid overlap
                ax=ax
            )
        
        # Draw nodes with professional styling
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            linewidths=2.0,
            edgecolors='black',
            ax=ax
        )
        
        # Draw node labels with better formatting
        labels = {node: self._format_node_label(node) for node in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=labels,
            font_size=10,
            font_weight='bold',
            font_family='Arial',
            font_color='black',
            ax=ax
        )
        
        # Draw edge labels with improved positioning
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('label', 'UNKNOWN')
            edge_labels[(u, v)] = self._format_edge_label(relation)
        
        # Position edge labels to avoid overlap
        edge_label_pos = self._calculate_edge_label_positions(pos, edge_labels)
        
        for edge, label in edge_labels.items():
            if edge in edge_label_pos:
                x, y = edge_label_pos[edge]
                ax.text(x, y, label,
                    fontsize=8,
                    fontweight='normal',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8),
                    rotation=0)
        
        # Set professional styling
        ax.set_title(f'Knowledge Graph: Scene Understanding\nImage ID: {image_id}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Remove axes and clean up the plot
        ax.set_axis_off()
        
        # Add subtle grid for better readability (optional)
        # ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        
        # Add legend for edge types
        self._add_edge_legend(ax)
        
        # Add statistics box
        self._add_statistics_box(ax, num_nodes, len(self.graph.edges()))
        
        if save:
            output_path = os.path.join(self.kg_dir, f"{image_id}_knowledge_graph_professional.png")
            plt.savefig(output_path, 
                    bbox_inches='tight', 
                    dpi=300, 
                    facecolor='white',
                    edgecolor='none',
                    format='png')
            
            # Also save as PDF for publications
            pdf_path = os.path.join(self.kg_dir, f"{image_id}_knowledge_graph_professional.pdf")
            plt.savefig(pdf_path, 
                    bbox_inches='tight', 
                    format='pdf',
                    facecolor='white',
                    edgecolor='none')
            
            # Save graph as pickle
            pickle_path = os.path.join(self.kg_dir, f"IMG_{image_id}_{self.image_data.img_type}_knowledge_graph.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.graph, f)
            
            print(f"Professional knowledge graph saved to {output_path}")
            print(f"PDF version saved to {pdf_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def _adjust_node_positions(self, pos: Dict, min_distance: float = 0.3) -> Dict:
        """
        Adjust node positions to prevent overlapping using force-directed approach
        """
        adjusted_pos = pos.copy()
        nodes = list(pos.keys())
        
        for _ in range(50):  # Iterative adjustment
            forces = {node: np.array([0.0, 0.0]) for node in nodes}
            
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    pos1 = np.array(adjusted_pos[node1])
                    pos2 = np.array(adjusted_pos[node2])
                    
                    distance = np.linalg.norm(pos2 - pos1)
                    
                    if distance < min_distance and distance > 0:
                        # Calculate repulsive force
                        direction = (pos1 - pos2) / distance
                        force_magnitude = (min_distance - distance) * 0.1
                        
                        forces[node1] += direction * force_magnitude
                        forces[node2] -= direction * force_magnitude
            
            # Apply forces
            for node in nodes:
                adjusted_pos[node] = (
                    adjusted_pos[node][0] + forces[node][0],
                    adjusted_pos[node][1] + forces[node][1]
                )
        
        return adjusted_pos

    def _get_node_colors(self) -> List[str]:
        """
        Get color scheme for nodes based on their properties or types
        """
        # Default color scheme - can be customized based on node properties
        colors = []
        color_map = {
            'object': '#4CAF50',      # Green for objects
            'location': '#2196F3',    # Blue for locations  
            'attribute': '#FF9800',   # Orange for attributes
            'action': '#9C27B0',      # Purple for actions
            'default': '#607D8B'      # Blue-gray for others
        }
        
        for node in self.graph.nodes():
            # Classify node type based on common patterns
            node_type = self._classify_node_type(node)
            colors.append(color_map.get(node_type, color_map['default']))
        
        return colors

    def _classify_node_type(self, node: str) -> str:
        """
        Classify node type based on its name and relationships
        """
        node_lower = node.lower()
        
        # Location indicators
        if any(word in node_lower for word in ['room', 'kitchen', 'bedroom', 'floor', 'wall', 'corner']):
            return 'location'
        
        # Attribute indicators  
        if any(word in node_lower for word in ['color', 'shape', 'size', 'material', 'texture']):
            return 'attribute'
            
        # Action indicators
        if any(word in node_lower for word in ['moving', 'sitting', 'standing', 'lying']):
            return 'action'
            
        # Default to object
        return 'object'

    def _get_edge_weight(self, relation: str) -> float:
        """
        Get edge thickness based on relation type
        """
        weight_map = {
            'IS_ON': 3.0,
            'IS_NEAR': 2.5,
            'IS_UNDER': 3.0,
            'HAS_COLOR': 1.5,
            'HAS_SHAPE': 1.5,
            'IS_USED_FOR': 2.0,
            'IS_PART_OF': 2.5,
            'CONTAINS': 2.0
        }
        return weight_map.get(relation, 2.0)

    def _get_edge_color(self, relation: str) -> str:
        """
        Get edge color based on relation type
        """
        color_map = {
            'IS_ON': '#E53E3E',       # Red for spatial (on)
            'IS_UNDER': '#E53E3E',    # Red for spatial (under)
            'IS_NEAR': '#38A169',     # Green for proximity
            'HAS_COLOR': '#3182CE',   # Blue for properties
            'HAS_SHAPE': '#3182CE',   # Blue for properties
            'IS_USED_FOR': '#805AD5', # Purple for functional
            'IS_PART_OF': '#D69E2E',  # Orange for composition
            'CONTAINS': '#D69E2E'     # Orange for composition
        }
        return color_map.get(relation, '#718096')

    def _get_edge_style(self, relation: str) -> str:
        """
        Get edge style based on relation type
        """
        style_map = {
            'IS_ON': '-',         # Solid for strong spatial
            'IS_UNDER': '-',      # Solid for strong spatial
            'IS_NEAR': '--',      # Dashed for proximity
            'HAS_COLOR': ':',     # Dotted for properties
            'HAS_SHAPE': ':',     # Dotted for properties
            'IS_USED_FOR': '-.',  # Dash-dot for functional
            'IS_PART_OF': '-',    # Solid for composition
            'CONTAINS': '-'       # Solid for composition
        }
        return style_map.get(relation, '-')

    def _format_node_label(self, node: str) -> str:
        """
        Format node labels for better readability
        """
        # Replace underscores with spaces and capitalize
        formatted = node.replace('_', ' ').title()
        
        # Limit length for readability
        if len(formatted) > 15:
            formatted = formatted[:12] + '...'
        
        return formatted

    def _format_edge_label(self, relation: str) -> str:
        """
        Format edge labels for better readability
        """
        # Convert to readable format
        formatted = relation.replace('_', ' ').lower()
        formatted = formatted.replace('is ', '').replace('has ', '').replace('are ', '')
        
        # Shorten common relations
        short_map = {
            'on': 'on',
            'under': 'under', 
            'near': 'near',
            'color': 'color',
            'shape': 'shape',
            'used for': 'used for',
            'part of': 'part of',
            'contains': 'contains'
        }
        
        return short_map.get(formatted, formatted)

    def _calculate_edge_label_positions(self, pos: Dict, edge_labels: Dict) -> Dict:
        """
        Calculate optimal positions for edge labels to minimize overlap
        """
        label_positions = {}
        
        for (u, v), label in edge_labels.items():
            if u in pos and v in pos:
                # Calculate midpoint of edge
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                
                # Offset the label slightly to avoid overlap with edge
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # Add very small perpendicular offset to stay close to edge
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Much smaller perpendicular vector for closer positioning
                    perp_x = -dy / length * 0.02  # Reduced from 0.1 to 0.02
                    perp_y = dx / length * 0.02   # Reduced from 0.1 to 0.02
                    
                    label_positions[(u, v)] = (mid_x + perp_x, mid_y + perp_y)
                else:
                    label_positions[(u, v)] = (mid_x, mid_y)
        
        return label_positions

    def _add_edge_legend(self, ax) -> None:
        """
        Add legend for edge types and colors
        """
        legend_elements = []
        
        # Create legend entries for main relation types
        relation_info = [
            ('Spatial Relations', '#E53E3E', '-'),
            ('Proximity Relations', '#38A169', '--'),
            ('Property Relations', '#3182CE', ':'),
            ('Functional Relations', '#805AD5', '-.'),
            ('Composition Relations', '#D69E2E', '-')
        ]
        
        for label, color, style in relation_info:
            legend_elements.append(plt.Line2D([0], [0], color=color, linestyle=style,
                                            linewidth=2, label=label))
        
        ax.legend(handles=legend_elements, 
                loc='upper right', 
                bbox_to_anchor=(1.0, 1.0),
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=10)

    def _add_statistics_box(self, ax, num_nodes: int, num_edges: int) -> None:
        """
        Add statistics box to the plot
        """
        stats_text = f"Nodes: {num_nodes}\nEdges: {num_edges}\nDensity: {num_edges/max(1, num_nodes*(num_nodes-1)):.3f}"
        
        ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='lightgray',
                        alpha=0.8))
    
    def plot_relationship_matrix(self, cmap_color: str = "viridis",
                               figsize: Tuple[int, int] = (12, 10), save: bool = True, 
                               show: bool = True) -> None:
        """
        Plot relationship matrix heatmap
        """

        image_id = self.image_data.ID

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
            output_path = os.path.join(self.matrix_dir, f"IMG_{image_id}_{self.image_data.img_type}_knowledge_matrix.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Relationship matrix saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_adjacency_matrix(self, save: bool = True) -> pd.DataFrame:
        """
        Generate and save adjacency matrix
        """

        image_id = self.image_data.ID
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
            output_path = os.path.join(self.matrix_dir, f"IMG_{image_id}_{self.image_data.img_type}_adjacency_matrix.csv")
            matrix_df.to_csv(output_path)
            print(f"Adjacency matrix saved to {output_path}")
        
        return matrix_df
    
    
    
    def process_image_complete(self, num_iter,  debug: bool = True) -> Dict:
        """
        Complete knowledge graph processing pipeline
        """
        print(f"Processing knowledge graph for image {self.image_data.ID}")
        
        image_data = self.image_data
        
        # Generate triples
        # running this 3x so we can get a more enriched KG
        all_triples = []
        for i in range(num_iter):

            print(f"\n\nGenerating triples for iteration {i + 1}...")
            triples_raw = self.generate_triples(debug=debug)  
            triples_cleaned = self.clean_triples_text(triples_raw)
            lines = triples_cleaned.strip().split('\n')
            print("\nTotal triples generated:", str(len(lines)))

            for line in lines:
                # Remove parentheses and spaces, then split by comma
                triple = tuple(item.strip().replace('(', '').replace(')', '') for item in line.split(','))
                all_triples.append(triple)

        print("\nGenerated triples:", str(len(all_triples)))
        
        # Remove duplicates
        triples_final = list(set(all_triples))
        self.triples = triples_final
        print("Generated triples after removing duplicates: ", str(len(triples_final)))

        path = os.path.join(self.triples_final_dir,  f"IMG_{self.image_data.ID}_{self.image_data.img_type}_triples.txt")
        with open(path, "w") as f:
            f.write("\n".join(f"{s}, {p}, {o}" for s, p, o in triples_final))
        
        # Parse triples
        #parsed_triples = self.parse_triples(triples_text)
        
        # Build graph
        graph = self.build_graph()
        
        # Create visualizations
        image_id = str(self.image_data.ID)
        self.plot_knowledge_graph(save=True, show=True)
        self.plot_relationship_matrix(save=True, show=True)
        
        # Get adjacency matrix
        adj_matrix = self.get_adjacency_matrix(save=True)
        
        # Save complete results
        results = {
            'image_id': self.image_data.ID,
            'scene_description': self.image_data.description,
            'parsed_triples': triples_final,
            'adj_matrix' : adj_matrix,
            'bbox_info': self.bbox_info,
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges()),
            'graph' : graph
        }
        
        results_path =  os.path.join(self.kg_dir, f"IMG_{image_id}_{self.image_data.img_type}_knowledge_graph.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        if debug:
            print(f"Knowledge graph processing complete for image {image_data.ID}")
            print(f"Generated {len(triples_final)} triples")
            print(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        
        return results
    
    def save(self, state, filepath: str) -> None:
        """
        Save the complete knowledge graph state
        """
        
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
        self.num_edges = state.get('num_edges', 0)
        self.num_nodes = state.get('num_nodes', 0)


