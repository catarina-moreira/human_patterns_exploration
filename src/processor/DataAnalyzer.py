import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """
    Comprehensive eye-tracking data analyzer for exploration vs exploitation experiment.
    
    This class provides methods to analyze fixation patterns, region preferences,
    and group differences in visual search behavior.
    """
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with the dataset.
        
        Args:
            data_path (str): Path to the CSV file containing eye-tracking data
        """
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.participants = None
        self.setup_data()
        
    def setup_data(self):
        """Prepare and clean the data for analysis."""
        # Basic data cleaning
        self.df = self.df.dropna(subset=['ParticipantID', 'Group', 'Condition'])
        
        # Create readable labels
        self.df['GroupLabel'] = self.df['Group'].map({
            1: 'Exploitation', 
            2: 'Exploration'
        })
        
        self.df['ConditionLabel'] = self.df['Condition'].map({
            1: 'Exploit_Present',
            2: 'Exploit_Absent', 
            3: 'Explore_Present',
            4: 'Explore_Absent'  # Corrected: condition 4 is Explore_Absent
        })
        
        # Calculate derived measures
        self._calculate_derived_measures()
        
        print(f"Dataset loaded: {len(self.df)} fixations from {self.df['ParticipantID'].nunique()} participants")
        
    def _calculate_derived_measures(self):
        """Calculate additional measures for analysis."""
        # Region fixation indicators (already in data but make sure they're binary)
        self.df['RelSceneFix'] = self.df['Rel_Scene_Region_RelSceneFix']
        self.df['IrrelSceneFix'] = self.df['Irrev_Scene_Region_IrrelSceneFix'] 
        self.df['CntlSceneFix'] = self.df['Cntl_Scene_Region_CntlSceneFix']
        
        # Calculate center coordinates for regions
        self.df['Rel_Center_X'] = (self.df['Rel_Scene_Region_Rel_X1'] + self.df['Rel_Scene_Region_Rel_X2']) / 2
        self.df['Rel_Center_Y'] = (self.df['Rel_Scene_Region_Rel_Y1'] + self.df['Rel_Scene_Region_Rel_Y2']) / 2
        
        # Distance to region centers
        self.df['Dist_to_Rel_Center'] = np.sqrt(
            (self.df['X'] - self.df['Rel_Center_X'])**2 + 
            (self.df['Y'] - self.df['Rel_Center_Y'])**2
        )
        
    def basic_descriptives(self):
        """Generate basic descriptive statistics."""
        print("=== BASIC DESCRIPTIVE STATISTICS ===\n")
        
        # Dataset overview
        print(f"Total fixations: {len(self.df):,}")
        print(f"Unique participants: {self.df['ParticipantID'].nunique()}")
        print(f"Unique items: {self.df['ItemNum'].nunique()}")
        print(f"Groups: {self.df['GroupLabel'].value_counts().to_dict()}")
        print(f"Conditions: {self.df['ConditionLabel'].value_counts().to_dict()}\n")
        
        # Fixation duration statistics
        print("Fixation Duration Statistics:")
        print(self.df['FixationDuration'].describe())
        print(f"Median: {self.df['FixationDuration'].median():.2f} ms\n")
        
        # Spatial distribution
        print("Spatial Distribution:")
        print(f"X coordinates: {self.df['X'].min():.1f} - {self.df['X'].max():.1f}")
        print(f"Y coordinates: {self.df['Y'].min():.1f} - {self.df['Y'].max():.1f}\n")
        
        return self.df.describe()
    
    def participant_level_analysis(self):
        """Analyze data at the participant level."""
        print("=== PARTICIPANT-LEVEL ANALYSIS ===\n")
        
        # Aggregate by participant
        participant_stats = self.df.groupby(['ParticipantID', 'GroupLabel']).agg({
            'FixationDuration': ['count', 'mean', 'sum'],
            'RelSceneFix': 'sum',
            'IrrelSceneFix': 'sum', 
            'CntlSceneFix': 'sum',
            'ItemNum': 'nunique'
        }).round(2)
        
        participant_stats.columns = ['Total_Fixations', 'Mean_Duration', 'Total_Duration',
                                   'Rel_Fixations', 'Irrel_Fixations', 'Cntl_Fixations', 'Items_Viewed']
        
        # Calculate proportions
        participant_stats['Prop_Rel'] = participant_stats['Rel_Fixations'] / participant_stats['Total_Fixations']
        participant_stats['Prop_Irrel'] = participant_stats['Irrel_Fixations'] / participant_stats['Total_Fixations']
        participant_stats['Prop_Cntl'] = participant_stats['Cntl_Fixations'] / participant_stats['Total_Fixations']
        
        self.participants = participant_stats.reset_index()
        
        print("Participant-level summary by group:")
        group_summary = self.participants.groupby('GroupLabel')[
            ['Total_Fixations', 'Mean_Duration', 'Prop_Rel', 'Prop_Irrel', 'Prop_Cntl']
        ].agg(['mean', 'std']).round(3)
        print(group_summary)
        
        return self.participants
    
    def region_analysis(self):
        """Analyze fixation patterns across different regions."""
        print("\n=== REGION FIXATION ANALYSIS ===\n")
        
        # Overall region preferences
        total_fixations = len(self.df)
        rel_fixations = self.df['RelSceneFix'].sum()
        irrel_fixations = self.df['IrrelSceneFix'].sum()
        cntl_fixations = self.df['CntlSceneFix'].sum()
        
        print("Overall Region Fixation Proportions:")
        print(f"Relevant region: {rel_fixations/total_fixations:.3f} ({rel_fixations:,} fixations)")
        print(f"Irrelevant region: {irrel_fixations/total_fixations:.3f} ({irrel_fixations:,} fixations)")
        print(f"Control region: {cntl_fixations/total_fixations:.3f} ({cntl_fixations:,} fixations)")
        
        # By group analysis
        print("\nRegion preferences by group:")
        region_by_group = self.df.groupby('GroupLabel')[['RelSceneFix', 'IrrelSceneFix', 'CntlSceneFix']].agg(['sum', 'mean'])
        print(region_by_group)
        
        # Statistical test for group differences in region preferences
        exploitation_rel = self.df[self.df['GroupLabel'] == 'Exploitation']['RelSceneFix']
        exploration_rel = self.df[self.df['GroupLabel'] == 'Exploration']['RelSceneFix']
        
        stat, p_val = mannwhitneyu(exploitation_rel, exploration_rel, alternative='two-sided')
        print(f"\nMann-Whitney U test for relevant region fixations:")
        print(f"U-statistic: {stat:.2f}, p-value: {p_val:.6f}")
        
        return region_by_group
    
    def fixation_duration_analysis(self):
        """Analyze fixation duration patterns."""
        print("\n=== FIXATION DURATION ANALYSIS ===\n")
        
        # Duration by group
        duration_by_group = self.df.groupby('GroupLabel')['FixationDuration'].agg(['count', 'mean', 'median', 'std'])
        print("Fixation duration by group:")
        print(duration_by_group)
        
        # Duration by region type
        print("\nFixation duration by region type:")
        
        # Relevant region durations
        rel_durations = self.df[self.df['RelSceneFix'] == 1]['FixationDuration']
        irrel_durations = self.df[self.df['IrrelSceneFix'] == 1]['FixationDuration']
        cntl_durations = self.df[self.df['CntlSceneFix'] == 1]['FixationDuration']
        
        print(f"Relevant region: Mean={rel_durations.mean():.2f}ms, Median={rel_durations.median():.2f}ms")
        print(f"Irrelevant region: Mean={irrel_durations.mean():.2f}ms, Median={irrel_durations.median():.2f}ms")
        print(f"Control region: Mean={cntl_durations.mean():.2f}ms, Median={cntl_durations.median():.2f}ms")
        
        # Statistical tests
        exploitation_dur = self.df[self.df['GroupLabel'] == 'Exploitation']['FixationDuration']
        exploration_dur = self.df[self.df['GroupLabel'] == 'Exploration']['FixationDuration']
        
        stat, p_val = mannwhitneyu(exploitation_dur, exploration_dur, alternative='two-sided')
        print(f"\nGroup difference in fixation duration (Mann-Whitney U):")
        print(f"U-statistic: {stat:.2f}, p-value: {p_val:.6f}")
        
        return duration_by_group
    
    def group_condition_structure(self):
        """Analyze the experimental design structure."""
        print("=== EXPERIMENTAL DESIGN STRUCTURE ===\n")
        
        # Cross-tabulation of Group and Condition
        crosstab = pd.crosstab(self.df['GroupLabel'], self.df['ConditionLabel'], margins=True)
        print("Group × Condition Cross-tabulation:")
        print(crosstab)
        print()
        
        # Verify the design structure
        print("Design Structure Verification:")
        design_check = self.df.groupby(['GroupLabel', 'ConditionLabel']).size().unstack(fill_value=0)
        print(design_check)
        print()
        
        # Check if conditions are nested within groups
        group_conditions = self.df.groupby('GroupLabel')['ConditionLabel'].unique()
        print("Conditions within each group:")
        for group, conditions in group_conditions.items():
            print(f"{group}: {list(conditions)}")
        
        return crosstab
    
    def condition_analysis(self):
        """Analyze differences across experimental conditions."""
        print("\n=== CONDITION ANALYSIS ===\n")
        
        condition_stats = self.df.groupby(['GroupLabel', 'ConditionLabel']).agg({
            'FixationDuration': ['count', 'mean'],
            'RelSceneFix': 'mean',
            'IrrelSceneFix': 'mean',
            'CntlSceneFix': 'mean'
        }).round(3)
        
        print("Statistics by Group and Condition:")
        print(condition_stats)
        
        # Chi-square test for region preferences by condition
        contingency_table = pd.crosstab([self.df['GroupLabel'], self.df['ConditionLabel']], 
                                      [self.df['RelSceneFix'], self.df['IrrelSceneFix'], self.df['CntlSceneFix']])
        
        try:
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
            print(f"\nChi-square test for region preferences by condition:")
            print(f"Chi2: {chi2:.2f}, p-value: {p_val:.6f}, df: {dof}")
        except ValueError as e:
            print(f"Chi-square test not applicable: {e}")
        
        return condition_stats
    
    def spatial_analysis(self):
        """Analyze spatial distribution of fixations."""
        print("\n=== SPATIAL ANALYSIS ===\n")
        
        # Basic spatial statistics
        print("Spatial distribution statistics:")
        spatial_stats = self.df.groupby('GroupLabel')[['X', 'Y']].agg(['mean', 'std'])
        print(spatial_stats)
        
        # Distance to relevant region center
        print("\nDistance to relevant region center:")
        dist_stats = self.df.groupby('GroupLabel')['Dist_to_Rel_Center'].agg(['mean', 'median', 'std'])
        print(dist_stats)
        
        return spatial_stats
    
    def statistical_tests(self):
        """Perform key statistical tests."""
        print("\n=== STATISTICAL TESTS SUMMARY ===\n")
        
        if self.participants is None:
            self.participant_level_analysis()
        
        results = {}
        
        # Test 1: Group differences in relevant region fixation proportion
        exploit_rel_prop = self.participants[self.participants['GroupLabel'] == 'Exploitation']['Prop_Rel']
        explore_rel_prop = self.participants[self.participants['GroupLabel'] == 'Exploration']['Prop_Rel']
        
        stat, p_val = mannwhitneyu(exploit_rel_prop, explore_rel_prop, alternative='two-sided')
        results['relevant_region_preference'] = {'statistic': stat, 'p_value': p_val}
        print(f"1. Relevant region preference difference: U={stat:.2f}, p={p_val:.6f}")
        
        # Test 2: Group differences in mean fixation duration
        exploit_dur = self.participants[self.participants['GroupLabel'] == 'Exploitation']['Mean_Duration']
        explore_dur = self.participants[self.participants['GroupLabel'] == 'Exploration']['Mean_Duration']
        
        stat, p_val = mannwhitneyu(exploit_dur, explore_dur, alternative='two-sided')
        results['fixation_duration'] = {'statistic': stat, 'p_value': p_val}
        print(f"2. Mean fixation duration difference: U={stat:.2f}, p={p_val:.6f}")
        
        # Test 3: Group differences in total fixations
        exploit_total = self.participants[self.participants['GroupLabel'] == 'Exploitation']['Total_Fixations']
        explore_total = self.participants[self.participants['GroupLabel'] == 'Exploration']['Total_Fixations']
        
        stat, p_val = mannwhitneyu(exploit_total, explore_total, alternative='two-sided')
        results['total_fixations'] = {'statistic': stat, 'p_value': p_val}
        print(f"3. Total fixations difference: U={stat:.2f}, p={p_val:.6f}")
        
        # Test 4: Target present vs absent within exploitation group
        exploit_present_rel = self.df[(self.df['GroupLabel'] == 'Exploitation') & 
                                     (self.df['ConditionLabel'] == 'Exploit_Present')]['RelSceneFix']
        exploit_absent_rel = self.df[(self.df['GroupLabel'] == 'Exploitation') & 
                                    (self.df['ConditionLabel'] == 'Exploit_Absent')]['RelSceneFix']
        
        if len(exploit_present_rel) > 0 and len(exploit_absent_rel) > 0:
            stat, p_val = mannwhitneyu(exploit_present_rel, exploit_absent_rel, alternative='two-sided')
            results['exploit_target_effect'] = {'statistic': stat, 'p_value': p_val}
            print(f"4. Exploitation target present vs absent: U={stat:.2f}, p={p_val:.6f}")
        
        # Test 5: Target present vs absent within exploration group
        explore_present_rel = self.df[(self.df['GroupLabel'] == 'Exploration') & 
                                     (self.df['ConditionLabel'] == 'Explore_Present')]['RelSceneFix']
        explore_absent_rel = self.df[(self.df['GroupLabel'] == 'Exploration') & 
                                    (self.df['ConditionLabel'] == 'Explore_Absent')]['RelSceneFix']
        
        if len(explore_present_rel) > 0 and len(explore_absent_rel) > 0:
            stat, p_val = mannwhitneyu(explore_present_rel, explore_absent_rel, alternative='two-sided')
            results['explore_target_effect'] = {'statistic': stat, 'p_value': p_val}
            print(f"5. Exploration target present vs absent: U={stat:.2f}, p={p_val:.6f}")
        
        # Test 6: Overall condition differences (Kruskal-Wallis)
        condition_groups = []
        condition_labels = []
        for group in ['Exploitation', 'Exploration']:
            for condition in ['Present', 'Absent']:
                if group == 'Exploitation':
                    condition_label = f'Exploit_{condition}'
                else:
                    condition_label = f'Explore_{condition}'
                
                condition_data = self.df[(self.df['GroupLabel'] == group) & 
                                       (self.df['ConditionLabel'] == condition_label)]['RelSceneFix']
                if len(condition_data) > 0:
                    condition_groups.append(condition_data)
                    condition_labels.append(f'{group}_{condition}')
        
        if len(condition_groups) >= 3:
            stat, p_val = kruskal(*condition_groups)
            results['all_conditions'] = {'statistic': stat, 'p_value': p_val}
            print(f"6. All four conditions comparison (Kruskal-Wallis): H={stat:.2f}, p={p_val:.6f}")
        
        return results
    
    def generate_visualizations(self, save_plots=False):
        """Generate comprehensive visualizations."""
        print("\n=== GENERATING VISUALIZATIONS ===\n")
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Fixation duration distribution by group
        plt.subplot(3, 4, 1)
        for group in self.df['GroupLabel'].unique():
            group_data = self.df[self.df['GroupLabel'] == group]['FixationDuration']
            plt.hist(group_data, alpha=0.7, bins=30, label=group, density=True)
        plt.xlabel('Fixation Duration (ms)')
        plt.ylabel('Density')
        plt.title('Fixation Duration Distribution by Group')
        plt.legend()
        plt.xlim(0, 1000)
        
        # 2. Region preferences by group
        plt.subplot(3, 4, 2)
        region_props = self.df.groupby('GroupLabel')[['RelSceneFix', 'IrrelSceneFix', 'CntlSceneFix']].mean()
        region_props.plot(kind='bar', ax=plt.gca())
        plt.title('Region Fixation Proportions by Group')
        plt.ylabel('Proportion of Fixations')
        plt.xlabel('Group')
        plt.legend(['Relevant', 'Irrelevant', 'Control'])
        plt.xticks(rotation=45)
        
        # 3. Spatial distribution (heatmap)
        plt.subplot(3, 4, 3)
        plt.hist2d(self.df['X'], self.df['Y'], bins=50, cmap='YlOrRd')
        plt.colorbar(label='Fixation Count')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Overall Spatial Distribution')
        
        # 4. Fixation duration by region type
        plt.subplot(3, 4, 4)
        region_data = []
        region_labels = []
        for region, label in [('RelSceneFix', 'Relevant'), ('IrrelSceneFix', 'Irrelevant'), ('CntlSceneFix', 'Control')]:
            region_fixations = self.df[self.df[region] == 1]['FixationDuration']
            region_data.append(region_fixations)
            region_labels.append(label)
        
        plt.boxplot(region_data, labels=region_labels)
        plt.ylabel('Fixation Duration (ms)')
        plt.title('Fixation Duration by Region Type')
        plt.xticks(rotation=45)
        
        # 5. Participant-level analysis
        if self.participants is not None:
            plt.subplot(3, 4, 5)
            for group in self.participants['GroupLabel'].unique():
                group_data = self.participants[self.participants['GroupLabel'] == group]
                plt.scatter(group_data['Prop_Rel'], group_data['Mean_Duration'], 
                          alpha=0.7, label=group, s=60)
            plt.xlabel('Proportion Relevant Fixations')
            plt.ylabel('Mean Fixation Duration (ms)')
            plt.title('Individual Differences: Relevant Fixations vs Duration')
            plt.legend()
            
            # 6. Total fixations by group
            plt.subplot(3, 4, 6)
            self.participants.boxplot(column='Total_Fixations', by='GroupLabel', ax=plt.gca())
            plt.suptitle('')
            plt.title('Total Fixations by Group')
            plt.xlabel('Group')
            
        # 7. All four conditions comparison
        plt.subplot(3, 4, 7)
        condition_means = self.df.groupby('ConditionLabel')['RelSceneFix'].mean()
        bars = plt.bar(range(len(condition_means)), condition_means.values, 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.xticks(range(len(condition_means)), condition_means.index, rotation=45, ha='right')
        plt.ylabel('Proportion Relevant Fixations')
        plt.title('Relevant Region Fixations by All Conditions')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 8. Group × Condition interaction plot
        plt.subplot(3, 4, 8)
        interaction_data = self.df.groupby(['GroupLabel', 'ConditionLabel'])['RelSceneFix'].mean().unstack()
        
        for group in interaction_data.index:
            plt.plot(range(len(interaction_data.columns)), interaction_data.loc[group], 
                    marker='o', linewidth=2, markersize=8, label=group)
        
        plt.xticks(range(len(interaction_data.columns)), interaction_data.columns, rotation=45, ha='right')
        plt.ylabel('Proportion Relevant Fixations')
        plt.xlabel('Condition')
        plt.title('Group × Condition Interaction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(3, 4, 8)
        sequence_data = []
        for participant in self.df['ParticipantID'].unique()[:10]:  # Sample of participants
            for item in self.df[self.df['ParticipantID'] == participant]['ItemNum'].unique()[:3]:
                trial_data = self.df[(self.df['ParticipantID'] == participant) & 
                                   (self.df['ItemNum'] == item)].head(20)
                if len(trial_data) >= 10:
                    sequence_data.append(trial_data['RelSceneFix'].cumsum().values[:20])
        
        if sequence_data:
            sequence_array = np.array(sequence_data)
            mean_sequence = np.mean(sequence_array, axis=0)
            std_sequence = np.std(sequence_array, axis=0)
            fixation_number = np.arange(1, len(mean_sequence) + 1)
            
            plt.plot(fixation_number, mean_sequence, 'b-', linewidth=2)
            plt.fill_between(fixation_number, mean_sequence - std_sequence, 
                           mean_sequence + std_sequence, alpha=0.3)
            plt.xlabel('Fixation Number')
            plt.ylabel('Cumulative Relevant Fixations')
            plt.title('Temporal Pattern: Relevant Region Fixations')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('eyetracking_analysis.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'eyetracking_analysis.png'")
        
        plt.show()
    
    def comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("=" * 60)
        print("           EYE-TRACKING DATA ANALYSIS REPORT")
        print("=" * 60)
        
        # Run all analyses
        basic_stats = self.basic_descriptives()
        participant_data = self.participant_level_analysis()
        region_analysis = self.region_analysis()
        duration_analysis = self.fixation_duration_analysis()
        condition_analysis = self.condition_analysis()
        spatial_analysis = self.spatial_analysis()
        statistical_results = self.statistical_tests()
        
        # Summary findings
        print("\n=== KEY FINDINGS SUMMARY ===\n")
        
        if self.participants is not None:
            exploit_mean_rel = self.participants[self.participants['GroupLabel'] == 'Exploitation']['Prop_Rel'].mean()
            explore_mean_rel = self.participants[self.participants['GroupLabel'] == 'Exploration']['Prop_Rel'].mean()
            
            print(f"• Exploitation group fixates on relevant regions {exploit_mean_rel:.3f} of the time")
            print(f"• Exploration group fixates on relevant regions {explore_mean_rel:.3f} of the time")
            
            if statistical_results['relevant_region_preference']['p_value'] < 0.05:
                print(f"• Significant difference in relevant region preference (p={statistical_results['relevant_region_preference']['p_value']:.6f})")
            else:
                print(f"• No significant difference in relevant region preference (p={statistical_results['relevant_region_preference']['p_value']:.6f})")
        
        print(f"• Total dataset contains {len(self.df):,} fixations from {self.df['ParticipantID'].nunique()} participants")
        print(f"• Mean fixation duration: {self.df['FixationDuration'].mean():.2f}ms")
        
        return {
            'basic_stats': basic_stats,
            'participants': participant_data,
            'statistical_results': statistical_results
        }
