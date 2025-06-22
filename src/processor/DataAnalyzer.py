#!/usr/bin/env python3
"""
Eye-Tracking Data Analysis Runner
This script loads the enhanced DataAnalyzer class and runs the complete analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to display plots
plt.ion()  # Turn on interactive mode

class DataAnalyzer:
    """
    Enhanced comprehensive eye-tracking data analyzer for exploration vs exploitation experiment.
    """
    
    def __init__(self, data_path):
        """Initialize the analyzer with the dataset."""
        self.data_path = data_path
        print(f"Loading data from: {data_path}")
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
            4: 'Explore_Absent'
        })
        
        # Calculate derived measures
        self.calculate_derived_measures()
        
        print(f"Dataset loaded: {len(self.df)} fixations from {self.df['ParticipantID'].nunique()} participants")
        
    def calculate_derived_measures(self):
        """Calculate additional measures for analysis."""
        # Region fixation indicators
        self.df['RelSceneFix'] = self.df['Rel_Scene_Region_RelSceneFix']
        self.df['IrrelSceneFix'] = self.df['Irrev_Scene_Region_IrrelSceneFix'] 
        self.df['CntlSceneFix'] = self.df['Cntl_Scene_Region_CntlSceneFix']
        
        # Calculate center coordinates for regions
        self.df['Rel_Center_X'] = (self.df['Rel_Scene_Region_Rel_X1'] + self.df['Rel_Scene_Region_Rel_X2']) / 2
        self.df['Rel_Center_Y'] = (self.df['Rel_Scene_Region_Rel_Y1'] + self.df['Rel_Scene_Region_Rel_Y2']) / 2
        
    def basic_descriptives(self):
        """Generate basic descriptive statistics."""
        print("=== BASIC DESCRIPTIVE STATISTICS ===\n")
        
        print(f"Total fixations: {len(self.df):,}")
        print(f"Unique participants: {self.df['ParticipantID'].nunique()}")
        print(f"Unique items: {self.df['ItemNum'].nunique()}")
        print(f"Groups: {self.df['GroupLabel'].value_counts().to_dict()}")
        print(f"Conditions: {self.df['ConditionLabel'].value_counts().to_dict()}\n")
        
        print("Fixation Duration Statistics:")
        print(self.df['FixationDuration'].describe())
        print(f"Median: {self.df['FixationDuration'].median():.2f} ms\n")
        
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
    
    def create_enhanced_visualizations(self, save_plots=True):
        """Generate comprehensive visualizations focused on condition and group aggregations."""
        print("\n=== GENERATING ENHANCED VISUALIZATIONS ===")
        print("Creating comprehensive plots...")
        
        # Ensure we have participant data
        if self.participants is None:
            self.participant_level_analysis()
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Enhanced Eye-Tracking Data Analysis', fontsize=16, fontweight='bold')
        
        # Color schemes for consistency
        group_colors = {'Exploitation': '#2E86AB', 'Exploration': '#A23B72'}
        condition_colors = ['#F18F01', '#C73E1D', '#6A994E', '#577590']
        
        # 1. Mean fixation duration by group and condition
        plt.subplot(3, 4, 1)
        try:
            condition_means = self.df.groupby(['GroupLabel', 'ConditionLabel'])['FixationDuration'].mean().unstack()
            condition_means.plot(kind='bar', ax=plt.gca(), color=condition_colors, width=0.8)
            plt.title('Mean Fixation Duration\nby Group & Condition', fontsize=10)
            plt.ylabel('Duration (ms)')
            plt.xlabel('Group')
            plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.xticks(rotation=45)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 1\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 2. Relevant region fixations by condition
        plt.subplot(3, 4, 2)
        try:
            condition_rel_means = self.df.groupby('ConditionLabel')['RelSceneFix'].mean()
            bars = plt.bar(range(len(condition_rel_means)), condition_rel_means.values, color=condition_colors)
            plt.xticks(range(len(condition_rel_means)), condition_rel_means.index, rotation=45, ha='right')
            plt.ylabel('Proportion Relevant Fixations')
            plt.title('Relevant Region Fixations\nby Condition', fontsize=10)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 2\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 3. Region distribution by group
        plt.subplot(3, 4, 3)
        try:
            region_data = self.df.groupby('GroupLabel')[['RelSceneFix', 'IrrelSceneFix', 'CntlSceneFix']].mean()
            region_data.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], width=0.8)
            plt.title('Region Fixation Proportions\nby Group', fontsize=10)
            plt.ylabel('Proportion of Fixations')
            plt.xlabel('Group')
            plt.legend(['Relevant', 'Irrelevant', 'Control'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.xticks(rotation=45)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 3\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 4. Group × Condition interaction
        plt.subplot(3, 4, 4)
        try:
            interaction_data = self.df.groupby(['GroupLabel', 'ConditionLabel'])['RelSceneFix'].mean().unstack()
            for group in interaction_data.index:
                plt.plot(range(len(interaction_data.columns)), interaction_data.loc[group], 
                        marker='o', linewidth=3, markersize=8, label=group, 
                        color=group_colors.get(group, 'black'))
            plt.xticks(range(len(interaction_data.columns)), interaction_data.columns, rotation=45, ha='right')
            plt.ylabel('Proportion Relevant Fixations')
            plt.xlabel('Condition')
            plt.title('Group × Condition\nInteraction', fontsize=10)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 4\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 5. Participant scatter plot
        plt.subplot(3, 4, 5)
        try:
            for group in self.participants['GroupLabel'].unique():
                group_data = self.participants[self.participants['GroupLabel'] == group]
                plt.scatter(group_data['Total_Fixations'], group_data['Prop_Rel'], 
                          alpha=0.7, label=group, s=60, color=group_colors.get(group, 'black'))
            plt.xlabel('Total Fixations')
            plt.ylabel('Proportion Relevant Fixations')
            plt.title('Individual Differences', fontsize=10)
            plt.legend(fontsize=8)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 5\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 6. Duration by region type and group
        plt.subplot(3, 4, 6)
        try:
            duration_by_region_group = []
            for group in ['Exploitation', 'Exploration']:
                group_data = self.df[self.df['GroupLabel'] == group]
                rel_dur = group_data[group_data['RelSceneFix'] == 1]['FixationDuration'].mean()
                irrel_dur = group_data[group_data['IrrelSceneFix'] == 1]['FixationDuration'].mean()
                cntl_dur = group_data[group_data['CntlSceneFix'] == 1]['FixationDuration'].mean()
                duration_by_region_group.append([rel_dur, irrel_dur, cntl_dur])
            
            x = np.arange(3)
            width = 0.35
            regions = ['Relevant', 'Irrelevant', 'Control']
            
            plt.bar(x - width/2, duration_by_region_group[0], width, label='Exploitation', 
                   color=group_colors['Exploitation'])
            plt.bar(x + width/2, duration_by_region_group[1], width, label='Exploration', 
                   color=group_colors['Exploration'])
            plt.xlabel('Region Type')
            plt.ylabel('Mean Duration (ms)')
            plt.title('Duration by Region\n& Group', fontsize=10)
            plt.xticks(x, regions)
            plt.legend(fontsize=8)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 6\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 7. Boxplot of relevant fixations by condition
        plt.subplot(3, 4, 7)
        try:
            condition_data = []
            condition_labels = []
            for condition in sorted(self.df['ConditionLabel'].unique()):
                data = self.df[self.df['ConditionLabel'] == condition]['RelSceneFix']
                if len(data) > 0:
                    condition_data.append(data)
                    condition_labels.append(condition)
            
            if condition_data:
                bp = plt.boxplot(condition_data, labels=condition_labels, patch_artist=True)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(condition_colors[i % len(condition_colors)])
                plt.ylabel('Relevant Region Fixations')
                plt.title('Relevant Fixations\nDistribution', fontsize=10)
                plt.xticks(rotation=45)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 7\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 8. Target Present vs Absent
        plt.subplot(3, 4, 8)
        try:
            present_absent_means = []
            groups = ['Exploitation', 'Exploration']
            
            for group in groups:
                present_cond = f'{group.split("ation")[0]}_Present'
                absent_cond = f'{group.split("ation")[0]}_Absent'
                
                present_mean = self.df[(self.df['GroupLabel'] == group) & 
                                     (self.df['ConditionLabel'] == present_cond)]['RelSceneFix'].mean()
                absent_mean = self.df[(self.df['GroupLabel'] == group) & 
                                    (self.df['ConditionLabel'] == absent_cond)]['RelSceneFix'].mean()
                present_absent_means.append([present_mean, absent_mean])
            
            x = np.arange(2)
            width = 0.35
            
            plt.bar(x - width/2, [present_absent_means[0][0], present_absent_means[1][0]], 
                   width, label='Target Present', color='#2E86AB')
            plt.bar(x + width/2, [present_absent_means[0][1], present_absent_means[1][1]], 
                   width, label='Target Absent', color='#A23B72')
            plt.xlabel('Group')
            plt.ylabel('Proportion Relevant Fixations')
            plt.title('Target Present vs Absent', fontsize=10)
            plt.xticks(x, groups)
            plt.legend(fontsize=8)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 8\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 9. Heatmap of condition × group
        plt.subplot(3, 4, 9)
        try:
            heatmap_data = self.df.groupby(['GroupLabel', 'ConditionLabel'])['RelSceneFix'].mean().unstack()
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=plt.gca(), cbar_kws={'shrink': 0.8})
            plt.title('Heatmap: Relevant Fixations\nby Group × Condition', fontsize=10)
            plt.ylabel('Group')
            plt.xlabel('Condition')
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 9\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 10. Total fixations by group and condition
        plt.subplot(3, 4, 10)
        try:
            fixation_counts = self.df.groupby(['GroupLabel', 'ConditionLabel']).size().unstack()
            fixation_counts.plot(kind='bar', ax=plt.gca(), color=condition_colors, width=0.8)
            plt.title('Total Fixations\nby Group & Condition', fontsize=10)
            plt.ylabel('Number of Fixations')
            plt.xlabel('Group')
            plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.xticks(rotation=45)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 10\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 11. Spatial distribution
        plt.subplot(3, 4, 11)
        try:
            for group in ['Exploitation', 'Exploration']:
                group_data = self.df[self.df['GroupLabel'] == group]
                # Sample data for better visualization
                if len(group_data) > 1000:
                    group_data = group_data.sample(1000)
                plt.scatter(group_data['X'], group_data['Y'], alpha=0.5, 
                          label=group, s=10, color=group_colors[group])
            
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Spatial Distribution\nby Group', fontsize=10)
            plt.legend(fontsize=8)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 11\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 12. Summary statistics
        plt.subplot(3, 4, 12)
        try:
            summary_data = []
            for group in ['Exploitation', 'Exploration']:
                group_df = self.df[self.df['GroupLabel'] == group]
                summary_data.append([
                    group_df['ParticipantID'].nunique(),
                    len(group_df),
                    group_df['FixationDuration'].mean(),
                    group_df['RelSceneFix'].mean()
                ])
            
            table_data = np.array(summary_data).T
            columns = ['Exploitation', 'Exploration']
            rows = ['Participants', 'Total Fixations', 'Mean Duration', 'Prop. Relevant']
            
            # Create table
            table = plt.table(cellText=[[f'{val:.0f}' if i < 2 else f'{val:.3f}' for val in row] 
                                      for i, row in enumerate(table_data)],
                            rowLabels=rows,
                            colLabels=columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            plt.axis('off')
            plt.title('Summary Statistics', fontsize=10)
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot 12\nError: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        if save_plots:
            try:
                plt.savefig('enhanced_eyetracking_analysis.png', dpi=300, bbox_inches='tight')
                print(f"\n✓ Plots saved as 'enhanced_eyetracking_analysis.png'")
            except Exception as e:
                print(f"\n✗ Could not save plots: {e}")
        
        plt.show()
        print("✓ Visualizations displayed!")
        
        return True
    
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
        
        return results
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 60)
        print("        COMPREHENSIVE EYE-TRACKING DATA ANALYSIS")
        print("=" * 60)
        
        # Run analyses
        basic_stats = self.basic_descriptives()
        participant_data = self.participant_level_analysis()
        statistical_results = self.statistical_tests()
        
        # Generate visualizations
        print("\n" + "="*60)
        print("               GENERATING VISUALIZATIONS")
        print("="*60)
        
        viz_success = self.create_enhanced_visualizations(save_plots=True)
        
        # Summary
        print("\n" + "="*60)
        print("                    ANALYSIS COMPLETE")
        print("="*60)
        
        if self.participants is not None:
            exploit_mean_rel = self.participants[self.participants['GroupLabel'] == 'Exploitation']['Prop_Rel'].mean()
            explore_mean_rel = self.participants[self.participants['GroupLabel'] == 'Exploration']['Prop_Rel'].mean()
            
            print(f"• Exploitation group fixates on relevant regions {exploit_mean_rel:.3f} of the time")
            print(f"• Exploration group fixates on relevant regions {explore_mean_rel:.3f} of the time")
            print(f"• Difference: {abs(exploit_mean_rel - explore_mean_rel):.3f}")
            
            if statistical_results['relevant_region_preference']['p_value'] < 0.05:
                print(f"• *** SIGNIFICANT difference in relevant region preference (p={statistical_results['relevant_region_preference']['p_value']:.6f}) ***")
            else:
                print(f"• No significant difference in relevant region preference (p={statistical_results['relevant_region_preference']['p_value']:.6f})")
        
        print(f"• Total dataset: {len(self.df):,} fixations from {self.df['ParticipantID'].nunique()} participants")
        print(f"• Mean fixation duration: {self.df['FixationDuration'].mean():.2f}ms")
        
        if viz_success:
            print("• ✓ All visualizations generated successfully!")
        else:
            print("• ⚠ Some visualizations may have issues")
        
        return {
            'basic_stats': basic_stats,
            'participants': participant_data,
            'statistical_results': statistical_results
        }


