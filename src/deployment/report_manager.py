import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from .deploy_verifier import DeploymentVerifier

class DeploymentReportManager:
    def __init__(self):
        self.root_dir = Path("c:/devdrive/thInk")
        self.reports_dir = self.root_dir / "reports" / "deployment"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_deployment_report(self, deployment_path: Path) -> Dict:
        """Generate comprehensive deployment report"""
        verifier = DeploymentVerifier(deployment_path)
        verification_status = verifier.verify_deployment()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'deployment_path': str(deployment_path),
            'verification_status': verification_status,
            'verification_results': verifier.verification_results,
            'size_metrics': self._get_size_metrics(deployment_path),
            'dependency_analysis': self._analyze_dependencies(deployment_path),
            'asset_inventory': self._inventory_assets(deployment_path)
        }
        
        self._save_report(report)
        self._generate_visualizations(report)
        return report
        
    def _get_size_metrics(self, path: Path) -> Dict:
        """Calculate size metrics for deployment"""
        metrics = {'total_size': 0, 'file_count': 0, 'by_type': {}}
        
        for file in path.rglob('*'):
            if file.is_file():
                size = file.stat().st_size
                metrics['total_size'] += size
                metrics['file_count'] += 1
                
                file_type = file.suffix or 'no_extension'
                if file_type not in metrics['by_type']:
                    metrics['by_type'][file_type] = {'count': 0, 'size': 0}
                metrics['by_type'][file_type]['count'] += 1
                metrics['by_type'][file_type]['size'] += size
                
        return metrics
        
    def _analyze_dependencies(self, path: Path) -> Dict:
        """Analyze deployment dependencies"""
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return {}
            
        with open(manifest_path) as f:
            manifest = json.load(f)
            
        return {
            'required': manifest.get('requirements', {}),
            'optional': manifest.get('optional_requirements', {}),
            'conflicts': self._check_dependency_conflicts(
                manifest.get('requirements', {})
            )
        }
        
    def _inventory_assets(self, path: Path) -> Dict:
        """Create inventory of deployment assets"""
        inventory = {
            'models': [],
            'configs': [],
            'resources': []
        }
        
        for file in path.rglob('*'):
            if file.is_file():
                if file.suffix == '.pth':
                    inventory['models'].append(str(file.relative_to(path)))
                elif file.suffix == '.json':
                    inventory['configs'].append(str(file.relative_to(path)))
                elif file.suffix in ['.ico', '.png', '.jpg']:
                    inventory['resources'].append(str(file.relative_to(path)))
                    
        return inventory
        
    def _generate_visualizations(self, report: Dict):
        """Generate visual reports"""
        # Size distribution pie chart
        plt.figure(figsize=(10, 6))
        sizes = [d['size'] for d in report['size_metrics']['by_type'].values()]
        labels = list(report['size_metrics']['by_type'].keys())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Deployment Size Distribution by File Type')
        plt.savefig(self.reports_dir / f"size_distribution_{datetime.now():%Y%m%d}.png")
        plt.close()
        
    def _save_report(self, report: Dict):
        """Save deployment report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"deployment_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
    def get_historical_metrics(self) -> pd.DataFrame:
        """Get historical deployment metrics"""
        reports = []
        for report_file in self.reports_dir.glob("*.json"):
            with open(report_file) as f:
                report = json.load(f)
                reports.append({
                    'timestamp': report['timestamp'],
                    'status': report['verification_status'],
                    'size': report['size_metrics']['total_size'],
                    'files': report['size_metrics']['file_count']
                })
                
        return pd.DataFrame(reports)
        def _generate_visualizations(self, report: Dict):
        """Generate visual reports"""
        # Size distribution pie chart
        self._create_size_distribution_chart(report)
        
        # Dependency status chart
        self._create_dependency_chart(report)
        
        # Asset distribution chart
        self._create_asset_distribution_chart(report)
        
        # Historical trends
        self._create_trend_analysis()
        
    def _create_size_distribution_chart(self, report: Dict):
        plt.figure(figsize=(12, 8))
        sizes = [d['size'] for d in report['size_metrics']['by_type'].values()]
        labels = list(report['size_metrics']['by_type'].keys())
        
        plt.subplot(1, 2, 1)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Size Distribution by File Type')
        
        plt.subplot(1, 2, 2)
        counts = [d['count'] for d in report['size_metrics']['by_type'].values()]
        plt.bar(labels, counts)
        plt.title('File Count by Type')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"size_analysis_{datetime.now():%Y%m%d}.png")
        plt.close()
        
    def _create_dependency_chart(self, report: Dict):
        deps = report['dependency_analysis']
        if not deps:
            return
            
        plt.figure(figsize=(10, 6))
        categories = ['Required', 'Optional', 'Conflicts']
        counts = [
            len(deps['required']),
            len(deps['optional']),
            len(deps.get('conflicts', []))
        ]
        
        plt.bar(categories, counts, color=['blue', 'green', 'red'])
        plt.title('Dependency Analysis')
        plt.ylabel('Count')
        
        plt.savefig(self.reports_dir / f"dependency_analysis_{datetime.now():%Y%m%d}.png")
        plt.close()
        
    def _create_asset_distribution_chart(self, report: Dict):
        inventory = report['asset_inventory']
        
        plt.figure(figsize=(8, 6))
        categories = ['Models', 'Configs', 'Resources']
        counts = [
            len(inventory['models']),
            len(inventory['configs']),
            len(inventory['resources'])
        ]
        
        plt.bar(categories, counts, color=['purple', 'orange', 'cyan'])
        plt.title('Asset Distribution')
        plt.ylabel('Count')
        
        plt.savefig(self.reports_dir / f"asset_distribution_{datetime.now():%Y%m%d}.png")
        plt.close()
        
    def _create_trend_analysis(self):
        df = self.get_historical_metrics()
        if df.empty:
            return
            
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        df.plot(x='timestamp', y='size', kind='line')
        plt.title('Deployment Size Trend')
        plt.ylabel('Size (bytes)')
        
        plt.subplot(2, 1, 2)
        df.plot(x='timestamp', y='files', kind='line')
        plt.title('File Count Trend')
        plt.ylabel('Number of Files')
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"deployment_trends_{datetime.now():%Y%m%d}.png")
        plt.close()