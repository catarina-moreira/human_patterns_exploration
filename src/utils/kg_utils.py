
import os
import pandas as pd
from src.core.FixationTask import FixationTask
from src.core.Participant import Participant
from src.core.ImageData import ImageData
from src.models.KG import KnowledgeGraph  # Import the new KG class
from src.models.LLM import LLM, OpenAI, Ollama
from src.models.Framework import SceneUnderstandingFramework
from src.utils.llm_utils import create_llm_instance
from typing import List, Dict, Optional, Tuple

import os
import pandas as pd
from src.core.FixationTask import FixationTask
from src.core.Participant import Participant
from src.core.ImageData import ImageData
from src.models.KG import KnowledgeGraph  # Import the new KG class
from src.models.LLM import LLM, OpenAI, Ollama
from src.models.Framework import SceneUnderstandingFramework
from src.utils.llm_utils import create_llm_instance

def process_multiple_images(image_paths: List[str], llm: LLM, masks_dict: Dict = None,
                          output_dir: str = "outputs", debug: bool = True) -> Dict:
    """
    Process multiple images for knowledge graph generation
    """
    kg = KnowledgeGraph(llm, output_dir)
    results = {}
    
    for i, image_path in enumerate(image_paths):
        try:
            print(f"\n=== Processing Image {i+1}/{len(image_paths)}: {image_path} ===")
            
            # Load image
            image_data = ImageData(image_path)
            
            # Get masks for this image (if available)
            masks = masks_dict.get(image_data.ID, []) if masks_dict else []
            
            # Process complete pipeline
            result = kg.process_image_complete(image_data, masks, debug=debug)
            results[image_data.ID] = result
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results[image_path] = {'error': str(e)}
    
    return results


def process_single_image_kg(imageData : ImageData, llm : LLM,  output_dir = "."):
    """Process a single image with knowledge graph generation"""
    
    print("=== SINGLE IMAGE KNOWLEDGE GRAPH PROCESSING ===")
    
    # Create Knowledge Graph processor
    kg = KnowledgeGraph(llm, output_dir=output_dir)
    
    # For now, we'll use empty masks list since MaskGenerator integration would require SAM2 setup
    # In practice, you would generate masks using MaskGenerator
    masks = []  # This would come from MaskGenerator.compute_masks_with_prompt()
    
    # Process complete knowledge graph pipeline
    results = kg.process_image_complete(
        image_data=imageData,
        masks=masks,
        scene_description=None,  # Will be generated automatically
        debug=True
    )
    
    print("\n=== RESULTS ===")
    print(f"Generated {len(results['parsed_triples'])} triples")
    print(f"Knowledge graph has {results['num_nodes']} nodes and {results['num_edges']} edges")
    
    # Example reasoning question
    if results['parsed_triples']:
        answer = kg.ask_reasoning_question("What objects are on the table?")
        print(f"\nReasoning Question: What objects are on the table?")
        print(f"Answer: {answer}")
    
    return kg, results

# Example: Batch Processing with Knowledge Graphs
def process_batch_images_kg():
    """Process multiple images for knowledge graph generation"""
    
    print("=== BATCH KNOWLEDGE GRAPH PROCESSING ===")
    
    # Get image paths
    image_paths = []
    for i in range(1, 4):  # Process images 1-3
        imgpath = os.path.join(IMAGE_DIR, f"{i}exp.jpg")
        if os.path.exists(imgpath):
            image_paths.append(imgpath)
    
    if not image_paths:
        print("No images found for batch processing")
        return None
    
    # Create LLM instance
    llm = create_llm_instance("openai", "gpt-4o")
    
    # Process all images
    from src.models.KG import process_multiple_images
    results = process_multiple_images(
        image_paths=image_paths,
        llm=llm,
        masks_dict=None,  # No masks for this example
        output_dir=RESULTS_DIR,
        debug=True
    )
    
    print(f"\n=== BATCH RESULTS ===")
    for img_id, result in results.items():
        if 'error' not in result:
            print(f"Image {img_id}: {len(result['parsed_triples'])} triples, "
                  f"{result['num_nodes']} nodes, {result['num_edges']} edges")
        else:
            print(f"Image {img_id}: Error - {result['error']}")
    
    return results

# Integration with existing eye-tracking analysis
def integrate_kg_with_fixations():
    """Integrate knowledge graph with eye-tracking fixation analysis"""
    
    print("=== INTEGRATING KG WITH EYE-TRACKING ===")
    
    # Load eye-tracking data
    data_path = os.path.join(DATA_DIR, "XSQ_Expt1_Data_2.csv")
    if not os.path.exists(data_path):
        print(f"Eye-tracking data not found at {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    
    # Load image
    IMG_ID = 1
    imgpath = os.path.join(IMAGE_DIR, f"{IMG_ID}exp.jpg")
    imageData = ImageData(imgpath)
    
    # Create participant and fixation task
    participant = Participant("ALL")  # Analyze all participants
    fixation_task = FixationTask(
        participant=participant,
        imageData=imageData,
        data=df,
        condition=1,
        group=1
    )
    
    # Generate knowledge graph
    llm = create_llm_instance("openai", "gpt-4o")
    kg = KnowledgeGraph(llm, output_dir=RESULTS_DIR)
    
    # Get scene description
    scene_description = llm.describe_scene(imageData)
    
    # For demonstration, we'll create some mock masks based on fixation regions
    # In practice, you would use actual masks from MaskGenerator
    masks = []  # This would be populated with real Mask objects
    
    # Process knowledge graph
    kg_results = kg.process_image_complete(imageData, masks, scene_description, debug=True)
    
    # Draw fixations
    print("\nDrawing fixation visualization...")
    fixation_task.draw_fixations(
        alpha=0.7,
        figsize=(12, 8),
        fix_color="skyblue",
        fix_edge_color="dark_red",
        title=f"Fixations for Image {IMG_ID} with {len(kg_results['parsed_triples'])} KG Triples"
    )
    
    # Draw heatmap
    print("Drawing fixation heatmap...")
    fixation_task.draw_heatmap(
        alpha=0.6,
        title=f"Fixation Heatmap - Image {IMG_ID}",
        cmap="viridis"
    )
    
    return fixation_task, kg, kg_results

# Advanced: Reasoning with spatial queries
def spatial_reasoning_demo():
    """Demonstrate spatial reasoning capabilities"""
    
    print("=== SPATIAL REASONING DEMONSTRATION ===")
    
    # Load image and create knowledge graph
    IMG_ID = 1
    imgpath = os.path.join(IMAGE_DIR, f"{IMG_ID}exp.jpg")
    imageData = ImageData(imgpath)
    
    llm = create_llm_instance("openai", "gpt-4o")
    kg = KnowledgeGraph(llm, output_dir=RESULTS_DIR)
    
    # Process with some example spatial reasoning
    results = kg.process_image_complete(imageData, [], debug=True)
    
    # Ask various spatial reasoning questions
    questions = [
        "What objects are near the window?",
        "Which objects are on top of other objects?",
        "What is the largest object in the scene?",
        "Which objects are used for sitting?",
        "What objects are in the center of the room?"
    ]
    
    print("\n=== SPATIAL REASONING QUESTIONS ===")
    for question in questions:
        answer = kg.ask_reasoning_question(question)
        print(f"Q: {question}")
        print(f"A: {answer}\n")
    
    return kg

def create_kg_summary_report(kg: KnowledgeGraph, image_id: str, output_path: str = None):
    """Create a summary report of the knowledge graph"""
    
    if not kg.triples:
        print("No knowledge graph data available")
        return
    
    report = f"""
# Knowledge Graph Summary Report - Image {image_id}

## Scene Description
{kg.scene_description[:500]}...

## Graph Statistics
- **Nodes**: {len(kg.graph.nodes())}
- **Edges**: {len(kg.graph.edges())}
- **Triples**: {len(kg.triples)}

## Top Relations
"""
    
    # Add top relations
    sorted_relations = sorted(kg.relations_count.items(), key=lambda x: x[1], reverse=True)
    for relation, count in sorted_relations[:10]:
        report += f"- **{relation}**: {count} occurrences\n"
    
    report += "\n## Sample Triples\n"
    for i, (s, p, o) in enumerate(kg.triples[:10]):
        report += f"{i+1}. ({s}, {p}, {o})\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")
    else:
        print(report)
    
    return report

def compare_knowledge_graphs(kg1: KnowledgeGraph, kg2: KnowledgeGraph, 
                           img1_id: str, img2_id: str):
    """Compare two knowledge graphs"""
    
    print(f"=== COMPARING KNOWLEDGE GRAPHS ===")
    print(f"Image {img1_id} vs Image {img2_id}")
    
    # Basic statistics comparison
    stats_comparison = {
        'Image': [img1_id, img2_id],
        'Nodes': [len(kg1.graph.nodes()), len(kg2.graph.nodes())],
        'Edges': [len(kg1.graph.edges()), len(kg2.graph.edges())],
        'Triples': [len(kg1.triples), len(kg2.triples)],
        'Unique_Relations': [len(kg1.relations_count), len(kg2.relations_count)]
    }
    
    comparison_df = pd.DataFrame(stats_comparison)
    print("\nStatistics Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Common relations
    common_relations = set(kg1.relations_count.keys()) & set(kg2.relations_count.keys())
    print(f"\nCommon Relations ({len(common_relations)}): {', '.join(sorted(common_relations))}")
    
    # Unique relations
    unique_kg1 = set(kg1.relations_count.keys()) - set(kg2.relations_count.keys())
    unique_kg2 = set(kg2.relations_count.keys()) - set(kg1.relations_count.keys())
    
    print(f"Unique to {img1_id} ({len(unique_kg1)}): {', '.join(sorted(unique_kg1))}")
    print(f"Unique to {img2_id} ({len(unique_kg2)}): {', '.join(sorted(unique_kg2))}")
    
    return comparison_df

