


# Helper function to create and run the experiment manager
def run_complete_experiment(data_path: str, image_dir: str, results_dir: str,
                        sam_config: str, sam_model: str, llm: LLM,
                        image_ids: Optional[List[int]] = None) -> ImageExperimentManager:
    """
    Convenience function to run the complete experiment pipeline.
    
    Args:
        data_path: Path to eye-tracking CSV data
        image_dir: Directory containing experimental images  
        results_dir: Directory to save results
        sam_config: Path to SAM2 config file
        sam_model: Path to SAM2 model weights
        llm: LLM instance for labeling and KG generation
        image_ids: Optional list of specific image IDs to process
    
    Returns:
        ImageExperimentManager instance with all results
    """
    
    # Create experiment manager
    manager = ImageExperimentManager(
        data_path=data_path,
        image_dir=image_dir, 
        results_dir=results_dir,
        sam_config=sam_config,
        sam_model=sam_model,
        llm=llm
    )
    
    # Process all images
    results = manager.process_all_images(image_ids)
    
    # Print statistics
    stats = manager.get_experiment_statistics()
    print("\n=== EXPERIMENT STATISTICS ===")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    # Save complete state
    state_path = os.path.join(results_dir, "complete_experiment_state.pkl")
    manager.save_complete_state(state_path)
    
    return manager