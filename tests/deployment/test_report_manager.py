import pytest
from pathlib import Path
import json
import shutil
from datetime import datetime
from src.deployment.report_manager import DeploymentReportManager

class TestDeploymentReportManager:
    @pytest.fixture
    def report_manager(self, tmp_path):
        manager = DeploymentReportManager()
        # Override paths for testing
        manager.reports_dir = tmp_path / "reports"
        manager.reports_dir.mkdir(parents=True)
        return manager
        
    @pytest.fixture
    def mock_deployment(self, tmp_path):
        deploy_path = tmp_path / "deployment"
        deploy_path.mkdir()
        
        # Create mock files
        (deploy_path / "main.exe").touch()
        (deploy_path / "config").mkdir()
        (deploy_path / "config/settings.json").write_text("{}")
        (deploy_path / "models").mkdir()
        (deploy_path / "models/model.pth").touch()
        
        # Create manifest
        manifest = {
            "version": "1.0.0",
            "requirements": {"numpy": "1.24.0"},
            "optional_requirements": {"opencv": "4.8.0"}
        }
        (deploy_path / "manifest.json").write_text(json.dumps(manifest))
        
        return deploy_path
        
    def test_report_generation(self, report_manager, mock_deployment):
        report = report_manager.generate_deployment_report(mock_deployment)
        
        assert report['verification_status'] is not None
        assert 'size_metrics' in report
        assert 'dependency_analysis' in report
        assert 'asset_inventory' in report
        
    def test_size_metrics_calculation(self, report_manager, mock_deployment):
        report = report_manager.generate_deployment_report(mock_deployment)
        metrics = report['size_metrics']
        
        assert metrics['total_size'] > 0
        assert metrics['file_count'] == 3  # main.exe, settings.json, model.pth
        assert '.json' in metrics['by_type']
        assert '.pth' in metrics['by_type']
        
    def test_asset_inventory(self, report_manager, mock_deployment):
        report = report_manager.generate_deployment_report(mock_deployment)
        inventory = report['asset_inventory']
        
        assert len(inventory['models']) == 1
        assert len(inventory['configs']) == 1
        assert len(inventory['resources']) == 0
        
    def test_dependency_analysis(self, report_manager, mock_deployment):
        report = report_manager.generate_deployment_report(mock_deployment)
        deps = report['dependency_analysis']
        
        assert 'numpy' in deps['required']
        assert 'opencv' in deps['optional']
        
    def test_historical_metrics(self, report_manager, mock_deployment):
        # Generate multiple reports
        for _ in range(3):
            report_manager.generate_deployment_report(mock_deployment)
            
        metrics_df = report_manager.get_historical_metrics()
        
        assert len(metrics_df) == 3
        assert all(col in metrics_df.columns 
                  for col in ['timestamp', 'status', 'size', 'files'])
        
    def test_visualization_generation(self, report_manager, mock_deployment):
        report = report_manager.generate_deployment_report(mock_deployment)
        
        expected_files = [
            "size_analysis_",
            "dependency_analysis_",
            "asset_distribution_",
            "deployment_trends_"
        ]
        
        for prefix in expected_files:
            matching_files = list(report_manager.reports_dir.glob(f"{prefix}*.png"))
            assert len(matching_files) > 0