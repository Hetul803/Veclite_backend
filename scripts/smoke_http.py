#!/usr/bin/env python3
"""
Smoke test script for Veclite backend API (HTTP-based).
Tests that POST endpoints don't hang and respond quickly.

Usage:
    python scripts/smoke_http.py <API_URL> <API_KEY>

Example:
    python scripts/smoke_http.py https://memryxbackend-production.up.railway.app veclite_sk_...
"""
import sys
import time
import requests
from typing import Optional


def test_health(url: str) -> bool:
    """Test GET /health endpoint (should be fast)."""
    print("Testing GET /health...")
    try:
        start = time.time()
        response = requests.get(f"{url}/health", timeout=5)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            print(f"✅ /health passed in {elapsed:.2f}s: {response.json()}")
            return True
        else:
            print(f"❌ /health failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ /health failed: {e}")
        return False


def test_finalize_no_body(url: str, api_key: str) -> bool:
    """Test POST /finalize with NO BODY, only X-API-Key header (must return fast, no hang)."""
    print("\nTesting POST /finalize (no body, X-API-Key header only)...")
    try:
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        start = time.time()
        # Send POST with empty body (or no body at all)
        response = requests.post(
            f"{url}/finalize",
            headers=headers,
            json={},  # Empty body
            timeout=10
        )
        elapsed = time.time() - start
        
        if response.status_code in [200, 409]:  # 409 = build already in progress (ok)
            print(f"✅ /finalize (no body) passed in {elapsed:.2f}s: {response.json()}")
            return True
        else:
            print(f"❌ /finalize (no body) failed: {response.status_code} - {response.text}")
            return False
    except requests.Timeout:
        print(f"❌ /finalize (no body) TIMED OUT after 10s (this is the bug!)")
        return False
    except Exception as e:
        print(f"❌ /finalize (no body) failed: {e}")
        return False


def test_add_single_vector(url: str, api_key: str) -> bool:
    """Test POST /add with single vector + X-API-Key header (must return fast, no hang)."""
    print("\nTesting POST /add (single vector, X-API-Key header)...")
    try:
        # Create a single 384-dim vector
        vector = [0.1] * 384
        
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "vectors": [
                {
                    "id": "test_vector_1",
                    "values": vector,
                    "metadata": {"test": True}
                }
            ]
        }
        
        start = time.time()
        response = requests.post(
            f"{url}/add",
            headers=headers,
            json=payload,
            timeout=10
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            print(f"✅ /add (single vector) passed in {elapsed:.2f}s: {response.json()}")
            return True
        else:
            print(f"❌ /add (single vector) failed: {response.status_code} - {response.text}")
            return False
    except requests.Timeout:
        print(f"❌ /add (single vector) TIMED OUT after 10s (this is the bug!)")
        return False
    except Exception as e:
        print(f"❌ /add (single vector) failed: {e}")
        return False


def test_finalize_with_authorization_header(url: str, api_key: str) -> bool:
    """Test POST /finalize with Authorization: Bearer header."""
    print("\nTesting POST /finalize (Authorization: Bearer header)...")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        start = time.time()
        response = requests.post(
            f"{url}/finalize",
            headers=headers,
            json={},
            timeout=10
        )
        elapsed = time.time() - start
        
        if response.status_code in [200, 409]:  # 409 = build already in progress (ok)
            print(f"✅ /finalize (Bearer) passed in {elapsed:.2f}s: {response.json()}")
            return True
        else:
            print(f"❌ /finalize (Bearer) failed: {response.status_code} - {response.text}")
            return False
    except requests.Timeout:
        print(f"❌ /finalize (Bearer) TIMED OUT after 10s")
        return False
    except Exception as e:
        print(f"❌ /finalize (Bearer) failed: {e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/smoke_http.py <API_URL> <API_KEY>")
        print("Example: python scripts/smoke_http.py https://memryxbackend-production.up.railway.app veclite_sk_...")
        sys.exit(1)
    
    url = sys.argv[1].rstrip('/')
    api_key = sys.argv[2]
    
    print("=" * 60)
    print("Smoke Testing Veclite Backend (HTTP)")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"API Key: {api_key[:20]}...")
    print("=" * 60)
    
    results = []
    
    # Test 1: GET /health (should be fast)
    results.append(("GET /health", test_health(url)))
    
    # Test 2: POST /finalize with no body, X-API-Key header only
    results.append(("POST /finalize (no body)", test_finalize_no_body(url, api_key)))
    
    # Test 3: POST /add single vector with X-API-Key header
    results.append(("POST /add (single vector)", test_add_single_vector(url, api_key)))
    
    # Test 4: POST /finalize with Authorization: Bearer
    results.append(("POST /finalize (Bearer)", test_finalize_with_authorization_header(url, api_key)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! POST endpoints are working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. POST endpoints may still be hanging.")
        sys.exit(1)


if __name__ == "__main__":
    main()

