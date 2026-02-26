#!/usr/bin/env python3
"""
Quick Upstox Token Generation Helper

This script helps you generate a new Upstox access token manually.
"""

import os
import sys
import webbrowser
from urllib.parse import urlencode

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utilities import Config

def generate_new_token():
    """Generate a new Upstox access token."""
    print("=" * 60)
    print("UPSTOX TOKEN GENERATION HELPER")
    print("=" * 60)
    
    # Load config
    config = Config()
    
    if not config.upstox_client_id:
        print("❌ No Upstox Client ID found in .env file")
        print("   Please add UPSTOX_CLIENT_ID to your .env file")
        return
    
    print(f"✅ Client ID found: {config.upstox_client_id}")
    print(f"✅ Redirect URI: {config.upstox_redirect_uri}")
    
    # Build authorization URL
    auth_params = {
        'response_type': 'code',
        'client_id': config.upstox_client_id,
        'redirect_uri': config.upstox_redirect_uri,
        'state': 'upstox_auth'
    }
    
    auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?{urlencode(auth_params)}"
    
    print("\n🔗 AUTHORIZATION URL:")
    print(auth_url)
    print("\n📋 INSTRUCTIONS:")
    print("1. Click the URL above or copy it to your browser")
    print("2. Log in to your Upstox account")
    print("3. Authorize the application")
    print("4. Copy the 'code' parameter from the redirect URL")
    print("5. Run: python scripts/setup_upstox_oauth.py")
    print("\n🌐 Opening browser...")
    
    try:
        webbrowser.open(auth_url)
        print("✅ Browser opened with authorization URL")
    except Exception as e:
        print(f"⚠️  Could not open browser: {e}")
        print("   Please manually copy the URL above")
    
    print("\n" + "=" * 60)
    print("WHY DO YOU NEED A NEW TOKEN?")
    print("=" * 60)
    print("• Upstox tokens expire DAILY at 3:30 AM IST")
    print("• Your current token likely expired this morning")
    print("• This is why you're getting 401 Unauthorized errors")
    print("• Generate a new token to get live data again")

if __name__ == "__main__":
    generate_new_token()
