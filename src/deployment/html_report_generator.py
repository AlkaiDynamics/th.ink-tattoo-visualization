from pathlib import Path
import jinja2
import base64
import json
from datetime import datetime
from typing import Dict, List

class HTMLReportGenerator:
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.template_dir = Path("c:/devdrive/thInk/src/deployment/templates")
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir))
        )
        
    def generate_html_report(self, report: Dict) -> str:
        """Generate HTML report from deployment data"""
        template = self.env.get_template("deployment_report.html")
        
        # Encode images to base64 for embedding
        images = self._get_report_images(report['timestamp'])
        
        report_data = {
            **report,
            'images': images,
            'formatted_time': datetime.fromisoformat(report['timestamp']).strftime(
                '%Y-%m-%d %H:%M:%S'
            ),
            'size_summary': self._format_size_metrics(report['size_metrics']),
            'verification_summary': self._format_verification_results(
                report['verification_results']
            )
        }
        
        return template.render(**report_data)
        
    def _get_report_images(self, timestamp: str) -> Dict[str, str]:
        """Get base64 encoded images for the report"""
        date_str = datetime.fromisoformat(timestamp).strftime('%Y%m%d')
        images = {}
        
        image_types = [
            'size_analysis',
            'dependency_analysis',
            'asset_distribution',
            'deployment_trends'
        ]
        
        for img_type in image_types:
            img_path = self.reports_dir / f"{img_type}_{date_str}.png"
            if img_path.exists():
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                    images[img_type] = f"data:image/png;base64,{img_data}"
                    
        return images
        
    def _format_size_metrics(self, metrics: Dict) -> Dict:
        """Format size metrics for display"""
        def format_size(size_bytes: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024
            return f"{size_bytes:.2f} TB"
            
        return {
            'total': format_size(metrics['total_size']),
            'by_type': {
                ext: {
                    'size': format_size(data['size']),
                    'count': data['count']
                }
                for ext, data in metrics['by_type'].items()
            }
        }
        
    def _format_verification_results(self, results: List[Dict]) -> Dict:
        """Format verification results for display"""
        summary = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        for result in results:
            category = result['status']
            if category in summary:
                summary[category].append(result)
                
        return summary