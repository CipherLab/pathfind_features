#!/usr/bin/env python3
"""
Quantitative Greatness Pipeline

This is the main orchestrator for transforming an over-engineered failure
analysis system into something that actually makes money.

Implements the three-phase approach:
1. Stop the Validation Lies - Honest validation with time machine tests
2. The Great Feature Purge - Remove unstable features across regimes
3. Embrace Target Selection Reality - Regime-aware models

The goal: Go from 0.0103 local correlation / -0.0002 live correlation
to something approaching EGG's 0.0322 correlation that persists.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


class QuantitativeGreatnessPipeline:
    """Main orchestrator for the quantitative transformation."""

    def __init__(self, base_dir: str = "pipeline_runs"):
        self.base_dir = base_dir
        self.run_dir = None
        self.logger = logging.getLogger(__name__)

    def setup_run_directory(self, experiment_name: Optional[str] = None) -> str:
        """Create a timestamped run directory."""
        if experiment_name:
            run_dir = os.path.join(self.base_dir, experiment_name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(self.base_dir, f"quantitative_greatness_{timestamp}")

        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(run_dir, 'quantitative_greatness.log')),
                logging.StreamHandler()
            ]
        )

        self.logger.info(f"🚀 Starting Quantitative Greatness Pipeline in {run_dir}")
        return run_dir

    def phase_1_honest_validation(self, data_file: str, features_file: str,
                                 params_file: str, vix_file: Optional[str] = None) -> Dict:
        """Phase 1: Stop the Validation Lies."""
        self.logger.info("📊 PHASE 1: Stop the Validation Lies")
        self.logger.info("Goal: Make validation predict live performance within 0.005 correlation")

        from validation_framework import run_honest_validation

        phase1_dir = os.path.join(self.run_dir, "phase1_honest_validation")
        os.makedirs(phase1_dir, exist_ok=True)

        # Run honest validation with time machine tests
        validation_results = run_honest_validation(
            data_file=data_file,
            features_file=features_file,
            params_file=params_file,
            vix_file=vix_file,
            n_splits=5,
            gap_eras=200  # Large gap for honesty
        )

        # Save phase 1 results
        phase1_results = {
            'phase': 'honest_validation',
            'validation_results': validation_results,
            'honesty_assessment': validation_results.get('honest_assessment', False),
            'recommendations': self._analyze_validation_honesty(validation_results)
        }

        with open(os.path.join(phase1_dir, 'phase1_results.json'), 'w') as f:
            json.dump(phase1_results, f, indent=2, default=str)

        self.logger.info(f"✅ Phase 1 Complete - Validation is {'HONEST' if validation_results.get('honest_assessment') else 'QUESTIONABLE'}")
        return phase1_results

    def phase_2_feature_purge(self, data_file: str, features_file: str,
                             vix_file: Optional[str] = None) -> Dict:
        """Phase 2: The Great Feature Purge."""
        self.logger.info("🔥 PHASE 2: The Great Feature Purge")
        self.logger.info("Goal: Kill features that don't survive crisis testing")

        from feature_purge_engine import run_feature_purge

        phase2_dir = os.path.join(self.run_dir, "phase2_feature_purge")
        os.makedirs(phase2_dir, exist_ok=True)

        # Run aggressive feature purge
        purge_results = run_feature_purge(
            data_file=data_file,
            features_file=features_file,
            output_dir=phase2_dir,
            crisis_eras=['2008-01', '2008-02', '2008-03', '2008-04'],  # 2008 crisis
            covid_eras=['2020-03', '2020-04', '2020-05', '2020-06'],  # COVID crash
            bear_market_eras=['2018-10', '2018-11', '2018-12']  # Random bear market
        )

        # Save curated features for next phases
        curated_features_file = os.path.join(self.run_dir, 'curated_features.json')
        with open(curated_features_file, 'w') as f:
            json.dump(purge_results['final_features'], f, indent=2)

        phase2_results = {
            'phase': 'feature_purge',
            'purge_results': purge_results,
            'survival_rate': purge_results['summary']['final_feature_count'] / purge_results['summary']['total_candidates'],
            'curated_features_file': curated_features_file
        }

        self.logger.info(f"✅ Phase 2 Complete - Purged {purge_results['summary']['total_candidates'] - purge_results['summary']['final_feature_count']} unstable features")
        self.logger.info(f"📊 Survival Rate: {phase2_results['survival_rate']:.1%}")
        return phase2_results

    def phase_3_regime_aware_models(self, data_file: str, curated_features_file: str,
                                   vix_file: Optional[str] = None) -> Dict:
        """Phase 3: Embrace Target Selection Reality."""
        self.logger.info("🎯 PHASE 3: Embrace Target Selection Reality")
        self.logger.info("Goal: Train specialized models for different market regimes")

        from regime_aware_model import run_regime_aware_training

        phase3_dir = os.path.join(self.run_dir, "phase3_regime_aware")
        os.makedirs(phase3_dir, exist_ok=True)

        # Train regime-aware models
        regime_results = run_regime_aware_training(
            data_file=data_file,
            features_file=curated_features_file,
            output_dir=phase3_dir,
            vix_file=vix_file
        )

        phase3_results = {
            'phase': 'regime_aware_models',
            'regime_results': regime_results,
            'regime_performance': regime_results['performance'],
            'model_files': [
                os.path.join(phase3_dir, f'{regime}_model.txt')
                for regime in regime_results['regimes_trained']
            ]
        }

        self.logger.info("✅ Phase 3 Complete - Trained regime-aware models")
        self.logger.info(f"🎯 Overall Correlation: {regime_results['performance']['overall_correlation']:.4f}")
        self.logger.info(f"⚡ Sharpe with TC: {regime_results['performance']['sharpe_with_tc']:.2f}")
        return phase3_results

    def run_full_transformation(self, data_file: str, features_file: str,
                               params_file: str, vix_file: Optional[str] = None) -> Dict:
        """Run the complete three-phase transformation."""
        self.logger.info("🚀 STARTING QUANTITATIVE GREATNESS TRANSFORMATION")
        self.logger.info("=" * 80)

        # Phase 1: Honest Validation
        phase1_results = self.phase_1_honest_validation(data_file, features_file, params_file, vix_file)

        # Phase 2: Feature Purge
        phase2_results = self.phase_2_feature_purge(data_file, features_file, vix_file)

        # Phase 3: Regime-Aware Models
        curated_features_file = phase2_results['curated_features_file']
        phase3_results = self.phase_3_regime_aware_models(data_file, curated_features_file, vix_file)

        # Final assessment
        final_assessment = self._create_final_assessment(phase1_results, phase2_results, phase3_results)

        # Save complete results
        complete_results = {
            'pipeline': 'quantitative_greatness',
            'run_directory': self.run_dir,
            'timestamp': datetime.now().isoformat(),
            'phase1': phase1_results,
            'phase2': phase2_results,
            'phase3': phase3_results,
            'final_assessment': final_assessment
        }

        results_file = os.path.join(self.run_dir, 'complete_transformation_results.json')
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)

        self._print_final_summary(final_assessment)
        return complete_results

    def _analyze_validation_honesty(self, validation_results: Dict) -> List[str]:
        """Analyze validation results and provide recommendations."""
        recommendations = []

        honesty_score = validation_results.get('time_machine', {}).get('honesty_score', 0)
        avg_drop = validation_results.get('time_machine', {}).get('avg_drop', 0)

        if honesty_score < 0.8:
            recommendations.append("❌ Validation is lying to you - correlation drops significantly on future data")
            recommendations.append("💡 Increase era gaps to 200+ eras between train/validation")
            recommendations.append("💡 Test on completely different time periods (2019 vs 2023)")

        if avg_drop > 0.5:
            recommendations.append("💥 Massive correlation drop detected - model is regime-specific")
            recommendations.append("🎯 Consider regime-aware modeling approach")

        sharpe_tc = validation_results.get('aggregated', {}).get('mean_sharpe_with_tc', 0)
        if sharpe_tc < 0.5:
            recommendations.append("📉 Sharpe ratio after transaction costs is too low")
            recommendations.append("💰 Focus on risk-adjusted returns, not raw correlation")

        return recommendations

    def _create_final_assessment(self, phase1: Dict, phase2: Dict, phase3: Dict) -> Dict:
        """Create final assessment of the transformation."""
        assessment = {
            'transformation_success': False,
            'key_improvements': [],
            'remaining_challenges': [],
            'next_steps': []
        }

        # Check if validation became honest
        if phase1.get('honesty_assessment', False):
            assessment['key_improvements'].append("✅ Validation now predicts live performance")
        else:
            assessment['remaining_challenges'].append("❌ Validation still lies about live performance")

        # Check feature purge effectiveness
        survival_rate = phase2.get('survival_rate', 0)
        if survival_rate < 0.3:
            assessment['key_improvements'].append(f"🔥 Aggressive feature purge: {survival_rate:.1%} survival rate")
        else:
            assessment['remaining_challenges'].append("🤔 Feature purge was too gentle - more features survived than expected")

        # Check regime-aware performance
        regime_corr = phase3.get('regime_performance', {}).get('overall_correlation', 0)
        regime_sharpe = phase3.get('regime_performance', {}).get('sharpe_with_tc', 0)

        if regime_corr > 0.01:
            assessment['key_improvements'].append(f"🎯 Regime-aware correlation: {regime_corr:.4f}")
        else:
            assessment['remaining_challenges'].append("📊 Regime-aware model correlation still below 0.01")

        if regime_sharpe > 0.5:
            assessment['key_improvements'].append(f"⚡ Sharpe ratio with TC: {regime_sharpe:.2f}")
        else:
            assessment['remaining_challenges'].append("💰 Sharpe ratio still too low for profitable trading")

        # Overall success criteria
        honesty_ok = phase1.get('honesty_assessment', False)
        correlation_ok = regime_corr >= 0.01
        sharpe_ok = regime_sharpe >= 0.5

        assessment['transformation_success'] = honesty_ok and (correlation_ok or sharpe_ok)

        # Next steps
        if not assessment['transformation_success']:
            assessment['next_steps'].extend([
                "🔄 Run another iteration with stricter criteria",
                "🎯 Focus on the specific regime where correlation is highest",
                "⚙️ Fine-tune hyperparameters for the regime-aware models",
                "📊 Implement more sophisticated neutralization techniques"
            ])
        else:
            assessment['next_steps'].extend([
                "🚀 Deploy to live trading with small position sizes",
                "📈 Monitor performance across different market regimes",
                "🔧 Implement dynamic regime detection for live trading",
                "📊 Set up proper risk management and position sizing"
            ])

        return assessment

    def _print_final_summary(self, assessment: Dict):
        """Print a comprehensive final summary."""
        print("\n" + "=" * 80)
        print("🎯 QUANTITATIVE GREATNESS TRANSFORMATION COMPLETE")
        print("=" * 80)

        success = assessment.get('transformation_success', False)
        status = "✅ SUCCESS" if success else "⚠️  NEEDS WORK"
        print(f"Overall Status: {status}")

        print("\n🔑 Key Improvements:")
        for improvement in assessment.get('key_improvements', []):
            print(f"  {improvement}")

        if assessment.get('remaining_challenges'):
            print("\n🚨 Remaining Challenges:")
            for challenge in assessment.get('remaining_challenges', []):
                print(f"  {challenge}")

        print("\n🎯 Next Steps:")
        for step in assessment.get('next_steps', []):
            print(f"  {step}")

        print(f"\n📁 Complete results saved to: {self.run_dir}")
        print("=" * 80)


def main():
    """Main entry point for the Quantitative Greatness Pipeline."""
    parser = argparse.ArgumentParser(
        description="Transform over-engineered failure analysis into profitable trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full transformation
  python quantitative_greatness_pipeline.py --data-file data.parquet --features-file features.json --params-file params.json

  # Run with VIX data for regime analysis
  python quantitative_greatness_pipeline.py --data-file data.parquet --features-file features.json --params-file params.json --vix-file vix.csv

  # Custom experiment name
  python quantitative_greatness_pipeline.py --experiment-name my_quantitative_revolution --data-file data.parquet --features-file features.json --params-file params.json
        """
    )

    parser.add_argument('--data-file', required=True, help='Path to training data parquet file')
    parser.add_argument('--features-file', required=True, help='Path to features JSON file')
    parser.add_argument('--params-file', required=True, help='Path to model parameters JSON file')
    parser.add_argument('--vix-file', help='Optional path to VIX data CSV for regime analysis')
    parser.add_argument('--experiment-name', help='Optional experiment name (default: timestamped)')
    parser.add_argument('--base-dir', default='pipeline_runs', help='Base directory for runs')

    args = parser.parse_args()

    # Validate inputs
    for file_path in [args.data_file, args.features_file, args.params_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return 1

    # Initialize and run pipeline
    pipeline = QuantitativeGreatnessPipeline(args.base_dir)
    pipeline.setup_run_directory(args.experiment_name)

    try:
        results = pipeline.run_full_transformation(
            data_file=args.data_file,
            features_file=args.features_file,
            params_file=args.params_file,
            vix_file=args.vix_file
        )

        success = results.get('final_assessment', {}).get('transformation_success', False)
        return 0 if success else 1

    except Exception as e:
        pipeline.logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
