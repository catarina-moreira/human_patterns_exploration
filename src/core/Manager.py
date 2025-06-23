
import os
import pickle
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.core.FixationTask import FixationTask
from src.core.Participant import Participant
from src.core.ImageData import ImageData
from src.core.Mask import Mask
from src.models.MaskGenerator import SAM2
from src.models.LLM import LLM
from src.models.KG import KnowledgeGraph
from src.models.Framework import SceneUnderstandingFramework


class ImageExperimentManager:
    """
    Manages the complete pipeline for processing eye-tracking experiments across 
    multiple participants and images. Handles segmentation, labeling, and KG generation.
    """
    
    def __init__(self, data_path: str, image_dir: str, results_dir: str, 
                sam_config: str, sam_model: str, llm: LLM):
        """
        Initialize the experiment manager.
        
        Args:
            data_path: Path to the eye-tracking CSV data
            image_dir: Directory containing experimental images
            results_dir: Directory to save results
            sam_config: Path to SAM2 config file
            sam_model: Path to SAM2 model weights
            llm: LLM instance for labeling and KG generation
        """
        self.data_path = data_path
        self.image_dir = image_dir
        self.results_dir = results_dir
        self.sam_config = sam_config
        self.sam_model = sam_model
        self.llm = llm
        
        # Load and prepare data
        self.data = pd.read_csv(data_path)
        self.image_participant_mapping = self.create_image_participant_mapping()
        
        # Storage for results
        self.fixation_tasks: Dict[int, Dict[int, FixationTask]] = defaultdict(dict)
        self.masks: Dict[int, Dict[int, List[Mask]]] = defaultdict(dict)
        self.labels: Dict[int, Dict[int, Dict[str, str]]] = defaultdict(dict)
        self.individual_kgs: Dict[int, Dict[int, KnowledgeGraph]] = defaultdict(dict)
        self.collective_kgs: Dict[int, KnowledgeGraph] = {}
        
        # Ensure results directory structure
        self.setup_directories()
    
    def create_image_participant_mapping(self) -> Dict[int, List[int]]:
        """Create mapping from image IDs to participant IDs who viewed them."""
        mapping = defaultdict(list)
        
        for _, row in self.data.iterrows():
            img_id = row['ItemNum']
            participant_id = row['ParticipantID']
            
            if participant_id not in mapping[img_id]:
                mapping[img_id].append(participant_id)
        
        return dict(mapping)
    
    def setup_directories(self):
        """Setup directory structure for results."""
        dirs_to_create = [
            self.results_dir,
            os.path.join(self.results_dir, "masks"),
            os.path.join(self.results_dir, "masks", "point"),
            os.path.join(self.results_dir, "masks", "triangle"),
            os.path.join(self.results_dir, "masks", "box"),
            os.path.join(self.results_dir, "masks", "cross"),
            os.path.join(self.results_dir, "masks", "best"),
            
            os.path.join(self.results_dir, "labels"),
            
            os.path.join(self.results_dir, "knowledge_graphs", "individual"),
            os.path.join(self.results_dir, "knowledge_graphs", "collective"),
            os.path.join(self.results_dir, "fixation_tasks")
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
    
    def get_participants_for_image(self, image_id: int) -> List[int]:
        """Get list of participant IDs who performed search task for given image."""
        return self.image_participant_mapping.get(image_id, [])
    
    def create_fixation_task(self, image_id: int, participant_id: int, 
                        condition: Optional[int] = None, 
                        group: Optional[int] = None) -> FixationTask:
        """Create a FixationTask for a specific image-participant combination."""
        
        # Load image
        img_path = os.path.join(self.image_dir, f"{image_id}exp.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image_data = ImageData(img_path)
        participant = Participant(participant_id)
        
        # Create fixation task
        fixation_task = FixationTask(
            participant=participant,
            imageData=image_data,
            data=self.data,
            condition=condition,
            group=group
        )
        
        # Store the task
        self.fixation_tasks[image_id][participant_id] = fixation_task
        
        return fixation_task
    
    def generate_masks_for_participant(self, image_id: int, participant_id: int, 
                                    save_masks: bool = True) -> List[Mask]:
        """Generate SAM2 masks for a specific participant's fixations on an image."""
        
        # Get or create fixation task
        if participant_id not in self.fixation_tasks[image_id]:
            fixation_task = self.create_fixation_task(image_id, participant_id)
        else:
            fixation_task = self.fixation_tasks[image_id][participant_id]
        
        # Initialize SAM2
        sam2 = SAM2(self.sam_model, self.sam_config, fixation_task)
        
        # Generate mask using fixation points as prompts
        mask = sam2.compute_masks_with_prompt(
            ID=f"img_{image_id}_part_{participant_id}",
            prompt_type="fixation_points",
            save_mask=save_masks,
            output_image_path=os.path.join(
                self.results_dir, "masks", 
                f"img_{image_id}_part_{participant_id}"
            ) if save_masks else None
        )
        
        # Store masks
        self.masks[image_id][participant_id] = [mask]  # SAM2 returns single mask
        
        return [mask]
    
    def label_masks_for_participant(self, image_id: int, participant_id: int,
                                use_context: bool = True) -> Dict[str, str]:
        """Label masks for a specific participant using LLM."""
        
        if participant_id not in self.masks[image_id]:
            raise ValueError(f"No masks found for participant {participant_id} on image {image_id}")
        
        masks = self.masks[image_id][participant_id]
        image_data = self.fixation_tasks[image_id][participant_id].imageData
        
        # Use SceneUnderstandingFramework for labeling
        framework = SceneUnderstandingFramework(self.llm)
        
        # Get scene description first
        scene_description = framework.analyze_scene(image_data)
        
        # Label masks
        labels = framework.label_masks_batch(masks, image_data, use_context)
        
        # Store labels
        self.labels[image_id][participant_id] = labels
        
        # Save labels to file
        labels_path = os.path.join(
            self.results_dir, "labels",
            f"img_{image_id}_part_{participant_id}_labels.pkl"
        )
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f)
        
        return labels
    
    def generate_individual_kg(self, image_id: int, participant_id: int) -> KnowledgeGraph:
        """Generate knowledge graph for a single participant's view of an image."""
        
        if participant_id not in self.masks[image_id]:
            raise ValueError(f"No masks found for participant {participant_id} on image {image_id}")
        
        masks = self.masks[image_id][participant_id]
        image_data = self.fixation_tasks[image_id][participant_id].imageData
        
        # Create KG
        kg = KnowledgeGraph(self.llm, image_data, 
                        output_dir=os.path.join(self.results_dir, "knowledge_graphs", "individual"))
        
        # Process complete KG pipeline
        results = kg.process_image_complete(image_data, masks, debug=True)
        
        # Store KG
        self.individual_kgs[image_id][participant_id] = kg
        
        return kg
    
    def generate_collective_kg(self, image_id: int) -> KnowledgeGraph:
        """Generate collective knowledge graph for all participants who viewed an image."""
        
        if image_id not in self.masks or not self.masks[image_id]:
            raise ValueError(f"No masks found for image {image_id}")
        
        # Collect all masks from all participants
        all_masks = []
        image_data = None
        
        for participant_id, masks in self.masks[image_id].items():
            all_masks.extend(masks)
            if image_data is None:
                image_data = self.fixation_tasks[image_id][participant_id].imageData
        
        # Create collective KG
        kg = KnowledgeGraph(self.llm, image_data,
                        output_dir=os.path.join(self.results_dir, "knowledge_graphs", "collective"))
        
        # Process with all masks
        results = kg.process_image_complete(image_data, all_masks, debug=True)
        
        # Store collective KG
        self.collective_kgs[image_id] = kg
        
        return kg
    
    def process_single_image_complete(self, image_id: int, 
                                    generate_individual_kgs: bool = True,
                                    generate_collective_kg: bool = True) -> Dict:
        """
        Complete processing pipeline for a single image:
        1. Get all participants for the image
        2. Generate masks for each participant
        3. Label masks for each participant
        4. Generate individual KGs (optional)
        5. Generate collective KG (optional)
        """
        
        print(f"\n=== Processing Image {image_id} ===")
        
        participants = self.get_participants_for_image(image_id)
        if not participants:
            print(f"No participants found for image {image_id}")
            return {}
        
        print(f"Found {len(participants)} participants: {participants}")
        
        results = {
            'image_id': image_id,
            'participants': participants,
            'masks': {},
            'labels': {},
            'individual_kgs': {},
            'collective_kg': None
        }
        
        # Process each participant
        for participant_id in participants:
            print(f"\nProcessing participant {participant_id}...")
            
            try:
                # Create fixation task
                self.create_fixation_task(image_id, participant_id)
                
                # Generate masks
                masks = self.generate_masks_for_participant(image_id, participant_id)
                results['masks'][participant_id] = masks
                print(f"✓ Generated {len(masks)} masks")
                
                # Label masks
                labels = self.label_masks_for_participant(image_id, participant_id)
                results['labels'][participant_id] = labels
                print(f"✓ Generated {len(labels)} labels")
                
                # Generate individual KG
                if generate_individual_kgs:
                    kg = self.generate_individual_kg(image_id, participant_id)
                    results['individual_kgs'][participant_id] = kg
                    print(f"✓ Generated individual KG with {len(kg.triples)} triples")
                
            except Exception as e:
                print(f"✗ Error processing participant {participant_id}: {e}")
                continue
        
        # Generate collective KG
        if generate_collective_kg and self.masks[image_id]:
            try:
                collective_kg = self.generate_collective_kg(image_id)
                results['collective_kg'] = collective_kg
                print(f"✓ Generated collective KG with {len(collective_kg.triples)} triples")
            except Exception as e:
                print(f"✗ Error generating collective KG: {e}")
        
        print(f"✓ Completed processing for image {image_id}")
        return results
    
    def process_all_images(self, image_ids: Optional[List[int]] = None,
                        generate_individual_kgs: bool = True,
                        generate_collective_kg: bool = True) -> Dict:
        """Process all images in the experiment (or specified subset)."""
        
        if image_ids is None:
            image_ids = list(self.image_participant_mapping.keys())
        
        print(f"=== Processing {len(image_ids)} images ===")
        
        all_results = {}
        
        for image_id in image_ids:
            try:
                results = self.process_single_image_complete(
                    image_id, generate_individual_kgs, generate_collective_kg
                )
                all_results[image_id] = results
            except Exception as e:
                print(f"✗ Error processing image {image_id}: {e}")
                continue
        
        # Save summary results
        summary_path = os.path.join(self.results_dir, "experiment_summary.pkl")
        with open(summary_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n✓ Completed processing all images. Results saved to {summary_path}")
        return all_results
    
    def get_experiment_statistics(self) -> Dict:
        """Get comprehensive statistics about the experiment."""
        
        stats = {
            'total_images': len(self.image_participant_mapping),
            'total_participants': len(self.data['ParticipantID'].unique()),
            'total_fixations': len(self.data),
            'images_processed': len(self.fixation_tasks),
            'participants_per_image': {},
            'masks_generated': 0,
            'labels_generated': 0,
            'individual_kgs_generated': 0,
            'collective_kgs_generated': len(self.collective_kgs)
        }
        
        for image_id, participants in self.image_participant_mapping.items():
            stats['participants_per_image'][image_id] = len(participants)
        
        # Count masks and labels
        for image_id in self.masks:
            for participant_id in self.masks[image_id]:
                stats['masks_generated'] += len(self.masks[image_id][participant_id])
        
        for image_id in self.labels:
            for participant_id in self.labels[image_id]:
                stats['labels_generated'] += len(self.labels[image_id][participant_id])
        
        for image_id in self.individual_kgs:
            stats['individual_kgs_generated'] += len(self.individual_kgs[image_id])
        
        return stats
    
    def save_complete_state(self, filepath: str):
        """Save the complete state of the experiment manager."""
        
        state = {
            'image_participant_mapping': self.image_participant_mapping,
            'fixation_tasks': self.fixation_tasks,
            'masks': self.masks,
            'labels': self.labels,
            'individual_kgs': self.individual_kgs,
            'collective_kgs': self.collective_kgs,
            'statistics': self.get_experiment_statistics()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Complete experiment state saved to {filepath}")
    
    def load_complete_state(self, filepath: str):
        """Load the complete state of the experiment manager."""
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.image_participant_mapping = state.get('image_participant_mapping', {})
        self.fixation_tasks = state.get('fixation_tasks', defaultdict(dict))
        self.masks = state.get('masks', defaultdict(dict))
        self.labels = state.get('labels', defaultdict(dict))
        self.individual_kgs = state.get('individual_kgs', defaultdict(dict))
        self.collective_kgs = state.get('collective_kgs', {})
        
        print(f"Complete experiment state loaded from {filepath}")

