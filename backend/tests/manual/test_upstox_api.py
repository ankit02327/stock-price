#!/usr/bin/env python3
"""
Test script to verify Upstox API credentials and connectivity with automatic token refresh
"""

import os
import sys
from dotenv import load_dotenv
import requests

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Import token manager for testing
try:
    from shared.upstox_token_manager import UpstoxTokenManager
    TOKEN_MANAGER_AVAILABLE = True
except ImportError:
    TOKEN_MANAGER_AVAILABLE = False
    print("⚠ Token manager not available - testing with static token only")

def test_upstox_connection():
    """Test Upstox API connection with static token."""
    api_key = os.getenv('UPSTOX_API_KEY')
    access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
    
    print("=" * 60)
    print("UPSTOX API CONNECTION TEST (Static Token)")
    print("=" * 60)
    
    # Check if credentials exist
    print(f"\n1. Checking credentials...")
    print(f"   API Key: {'✓ Found' if api_key else '✗ Missing'}")
    print(f"   Access Token: {'✓ Found' if access_token else '✗ Missing'}")
    
    if not access_token:
        print("\n❌ Access token not found in .env file!")
        print("   Please run: python backend/scripts/setup_upstox_oauth.py")
        return False
    
    # Test API call
    print(f"\n2. Testing Upstox API connection...")
    
    try:
        url = "https://api.upstox.com/v2/market-quote/ltp"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        # Test with RELIANCE stock
        params = {'symbol': 'NSE_EQ|INE002A01018'}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data}")
            if data.get('status') == 'success':
                print("\n✓ Upstox API is working correctly!")
                return True
            else:
                print(f"\n✗ API returned error: {data.get('message')}")
                return False
        elif response.status_code == 401:
            print("\n✗ Authentication failed - Token may be expired")
            if TOKEN_MANAGER_AVAILABLE:
                print("   → Try running the token refresh test below")
            else:
                print("   → Please run: python backend/scripts/setup_upstox_oauth.py")
            return False
        elif response.status_code == 403:
            print("\n✗ Access forbidden - API not enabled in Upstox dashboard")
            return False
        else:
            print(f"\n✗ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Connection error: {str(e)}")
        return False

def test_token_manager():
    """Test token manager functionality."""
    if not TOKEN_MANAGER_AVAILABLE:
        print("\n⚠ Token manager not available - skipping token refresh tests")
        return False
    
    print("\n" + "=" * 60)
    print("UPSTOX TOKEN MANAGER TEST")
    print("=" * 60)
    
    try:
        # Initialize token manager
        token_manager = UpstoxTokenManager()
        
        print(f"\n1. Token Manager Status:")
        token_info = token_manager.get_token_info()
        
        print(f"   Has Access Token: {'✓' if token_info['has_access_token'] else '✗'}")
        print(f"   Has Refresh Token: {'✓' if token_info['has_refresh_token'] else '✗'}")
        print(f"   Has Client Credentials: {'✓' if token_info['has_client_credentials'] else '✗'}")
        print(f"   Token Expired: {'✗' if not token_info['is_expired'] else '✓ (needs refresh)'}")
        
        if token_info['token_expiry']:
            print(f"   Expiry: {token_info['token_expiry']}")
            print(f"   Minutes Until Expiry: {token_info['minutes_until_expiry']:.1f}")
        
        if not token_manager.has_valid_credentials():
            print("\n❌ Missing OAuth credentials!")
            print("   Please run: python backend/scripts/setup_upstox_oauth.py")
            return False
        
        # Test getting valid token
        print(f"\n2. Testing Token Retrieval:")
        access_token = token_manager.get_valid_token()
        
        if access_token:
            print(f"   ✓ Valid token obtained: {access_token[:20]}...")
            
            # Test API call with managed token
            print(f"\n3. Testing API with Managed Token:")
            
            url = "https://api.upstox.com/v2/market-quote/ltp"
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }
            params = {'symbol': 'NSE_EQ|INE002A01018'}  # RELIANCE
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print("   ✓ API call successful with managed token!")
                    
                    # Extract price for display
                    data_key = 'NSE_EQ|INE002A01018'
                    if data_key in data.get('data', {}):
                        price = data['data'][data_key]['last_price']
                        print(f"   ✓ RELIANCE price: ₹{price}")
                    
                    return True
                else:
                    print(f"   ✗ API error: {data.get('message')}")
                    return False
            else:
                print(f"   ✗ HTTP error: {response.status_code}")
                return False
        else:
            print("   ✗ Failed to get valid token")
            return False
            
    except Exception as e:
        print(f"\n✗ Token manager error: {str(e)}")
        return False

def test_token_refresh_simulation():
    """Simulate token refresh by testing the refresh mechanism."""
    if not TOKEN_MANAGER_AVAILABLE:
        print("\n⚠ Token manager not available - skipping refresh simulation")
        return False
    
    print("\n" + "=" * 60)
    print("TOKEN REFRESH SIMULATION")
    print("=" * 60)
    
    try:
        token_manager = UpstoxTokenManager()
        
        if not token_manager.has_valid_credentials():
            print("❌ Missing OAuth credentials for refresh test")
            return False
        
        print("\n1. Testing 401 Error Handling:")
        
        # Simulate a 401 error by calling handle_401_error
        print("   Simulating 401 error...")
        refresh_result = token_manager.handle_401_error()
        
        if refresh_result:
            print("   ✓ Token refresh successful!")
            
            # Verify new token works
            new_token = token_manager.get_valid_token()
            if new_token:
                print("   ✓ New token obtained and ready for use")
                return True
            else:
                print("   ✗ Failed to get new token after refresh")
                return False
        else:
            print("   ✗ Token refresh failed")
            print("   This might be expected if the refresh token is invalid")
            return False
            
    except Exception as e:
        print(f"\n✗ Refresh simulation error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("UPSTOX API & TOKEN REFRESH TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Static token connection
    result1 = test_upstox_connection()
    results.append(("Static Token Connection", result1))
    
    # Test 2: Token manager functionality
    result2 = test_token_manager()
    results.append(("Token Manager", result2))
    
    # Test 3: Token refresh simulation (optional - may fail if refresh token is invalid)
    print("\n" + "=" * 60)
    print("OPTIONAL: Token Refresh Simulation")
    print("(This may fail if refresh token is invalid - that's normal)")
    print("=" * 60)
    
    result3 = test_token_refresh_simulation()
    results.append(("Token Refresh Simulation", result3))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"   {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if result1 or result2:  # At least one main test passed
        print("\n✓ Upstox integration is working!")
        if result2:
            print("✓ Automatic token refresh is available")
        return True
    else:
        print("\n❌ Upstox integration needs setup")
        print("   Run: python backend/scripts/setup_upstox_oauth.py")
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        sys.exit(1)
