import subprocess
import time
import json
import requests
import psutil
import matplotlib.pyplot as plt
from datetime import datetime


class LoadTestRunner:
    """Automated load testing suite"""
    
    def __init__(self, base_url="http://localhost:8003"):
        self.base_url = base_url
        self.results = {}
    
    def run_baseline_test(self):
        """Run baseline performance test"""
        print("🔄 Running baseline test (50 users, 5 spawn rate)...")
        
        cmd = [
            "locust", "-f", "locustfile.py",
            "--host", self.base_url,
            "--users", "50",
            "--spawn-rate", "5",
            "--run-time", "2m",
            "--headless",
            "--csv", "results/baseline"
        ]
        
        return self._run_test("baseline", cmd)
    
    def run_stress_test(self):
        """Run stress test with high load"""
        print("🔄 Running stress test (200 users, 20 spawn rate)...")
        
        cmd = [
            "locust", "-f", "locustfile.py",
            "--host", self.base_url,
            "--users", "200", 
            "--spawn-rate", "20",
            "--run-time", "3m",
            "--headless",
            "--csv", "results/stress"
        ]
        
        return self._run_test("stress", cmd)
    
    def run_scaling_test(self):
        """Test with multiple Docker containers"""
        print("🔄 Running container scaling test...")
        
        # This would be run against load-balanced setup
        cmd = [
            "locust", "-f", "locustfile.py", 
            "--host", "http://localhost",  # Nginx load balancer
            "--users", "500",
            "--spawn-rate", "50", 
            "--run-time", "5m",
            "--headless",
            "--csv", "results/scaling"
        ]
        
        return self._run_test("scaling", cmd)
    
    def _run_test(self, test_name, cmd):
        """Execute load test and collect metrics"""
        start_time = time.time()
        
        # Monitor system resources
        initial_cpu = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().percent
        
        try:
            # Run locust test
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Collect final metrics
            final_cpu = psutil.cpu_percent() 
            final_memory = psutil.virtual_memory().percent
            end_time = time.time()
            
            # Store results
            self.results[test_name] = {
                "duration": end_time - start_time,
                "cpu_usage": {
                    "initial": initial_cpu,
                    "final": final_cpu,
                    "peak": max(initial_cpu, final_cpu)
                },
                "memory_usage": {
                    "initial": initial_memory,
                    "final": final_memory,
                    "peak": max(initial_memory, final_memory)
                },
                "locust_output": result.stdout,
                "errors": result.stderr,
                "success": result.returncode == 0
            }
            
            print(f"✅ {test_name} test completed")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} test timed out")
            return False
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("📊 Generating load test report...")
        
        report = {
            "test_date": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results.values() if r["success"]),
            "results": self.results
        }
        
        # Save detailed report
        with open("results/load_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate summary
        self._create_summary_report()
        
        print("✅ Load test report saved to results/load_test_report.json")
    
    def _create_summary_report(self):
        """Create human-readable summary"""
        summary = []
        summary.append("# Rock Paper Scissors AI - Load Test Results")
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        for test_name, result in self.results.items():
            summary.append(f"## {test_name.title()} Test Results")
            
            if result["success"]:
                summary.append(f"- **Status**: ✅ PASSED")
                summary.append(f"- **Duration**: {result['duration']:.1f} seconds")
                summary.append(f"- **Peak CPU**: {result['cpu_usage']['peak']:.1f}%")
                summary.append(f"- **Peak Memory**: {result['memory_usage']['peak']:.1f}%")
                
                # Parse Locust output for key metrics
                output = result["locust_output"]
                if "requests/s" in output:
                    # Extract RPS if available
                    lines = output.split('\n')
                    for line in lines:
                        if "requests/s" in line:
                            summary.append(f"- **Performance**: {line.strip()}")
                            break
            else:
                summary.append(f"- **Status**: ❌ FAILED")
                if result["errors"]:
                    summary.append(f"- **Error**: {result['errors'][:200]}...")
            
            summary.append("")
        
        # Save summary
        with open("results/load_test_summary.md", "w") as f:
            f.write("\n".join(summary))


def setup_test_environment():
    """Setup directories and files for testing"""
    import os
    
    os.makedirs("results", exist_ok=True)
    
    print("📁 Test environment setup complete")


if __name__ == "__main__":
    print("🧪 Rock Paper Scissors AI - Load Test Suite")
    print("=" * 50)
    
    # Setup
    setup_test_environment()
    runner = LoadTestRunner()
    
    # Run test suite
    print("Starting comprehensive load testing...")
    
    # Run tests in sequence
    tests_to_run = [
        ("baseline", runner.run_baseline_test),
        ("stress", runner.run_stress_test),
        ("scaling", runner.run_scaling_test)
    ]
    
    successful_tests = 0
    for test_name, test_func in tests_to_run:
        print(f"\n{'='*30}")
        print(f"Running {test_name} test...")
        print(f"{'='*30}")
        
        if test_func():
            successful_tests += 1
        
        # Cool down between tests
        time.sleep(10)
    
    # Generate report
    runner.generate_report()
    
    print(f"\n🎯 Load Testing Complete!")
    print(f"   Tests Run: {len(tests_to_run)}")
    print(f"   Successful: {successful_tests}")
    print(f"   Report: results/load_test_report.json")
    print(f"   Summary: results/load_test_summary.md")
