"""Test script to see real-time SDK errors from deployed API"""
import httpx
import json
import time

API_URL = "https://ptr-langgraph-watsonx-api.onrender.com"

def test_sdk_error():
    """Send a chat message and inspect the detailed error from SDK"""
    print("=" * 80)
    print("Testing IBM watsonx.governance SDK on Render - Real-Time Error Detection")
    print("=" * 80)
    
    # Send chat message
    print("\n1. Sending chat message...")
    chat_payload = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "What is 5 + 5?"}],
        "stream": False
    }
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{API_URL}/v1/chat/completions", json=chat_payload)
            response.raise_for_status()
            chat_result = response.json()
            print(f"✓ Chat response: {chat_result['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"✗ Chat failed: {e}")
        return
    
    # Wait a moment for metrics to be processed
    time.sleep(2)
    
    # Fetch metrics with detailed error information
    print("\n2. Fetching metrics with error details...")
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{API_URL}/api/metrics")
            response.raise_for_status()
            metrics = response.json()
            
            print(f"✓ Metrics received")
            print(f"\nSummary:")
            print(f"  Total Evaluations: {metrics['summary']['total_metrics']}")
            print(f"  SDK Available: {metrics.get('summary', {}).get('sdk_available', 'N/A')}")
            
            # Check for errors in recent metrics
            if metrics.get("recent_metrics"):
                latest_metric = metrics["recent_metrics"][0]
                print(f"\n3. Latest Metric Analysis:")
                print(f"  SDK Available: {latest_metric.get('sdk_available', 'N/A')}")
                print(f"  Faithfulness: {latest_metric.get('faithfulness_score', 'N/A'):.3f}")
                print(f"  Relevance: {latest_metric.get('relevance_score', 'N/A'):.3f}")
                
                # Show error details if present
                if not latest_metric.get('sdk_available', False):
                    print(f"\n⚠ SDK NOT AVAILABLE - Error Details:")
                    print(f"  Error Type: {latest_metric.get('error_type', 'Unknown')}")
                    print(f"  Error Message: {latest_metric.get('error_message', 'No message')}")
                    
                    if 'error_traceback' in latest_metric:
                        print(f"\n  Full Traceback:")
                        print("  " + "-" * 76)
                        for line in latest_metric['error_traceback'].split('\n'):
                            print(f"  {line}")
                        print("  " + "-" * 76)
                else:
                    print(f"\n✓ SDK is working correctly!")
            
            # Print full metric for inspection
            print(f"\n4. Full Latest Metric JSON:")
            print(json.dumps(latest_metric, indent=2))
            
    except Exception as e:
        print(f"✗ Metrics fetch failed: {e}")

if __name__ == "__main__":
    test_sdk_error()
