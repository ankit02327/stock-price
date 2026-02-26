#!/usr/bin/env python3
"""
Manual OAuth setup with your credentials
"""

import os
import sys
import json
import webbrowser
import http.server
import socketserver
import threading
import time
import urllib.parse
import requests
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.upstox_token_manager import UpstoxTokenManager

# Your credentials - Load from cache or enter manually
# Get these from: https://upstox.com/developer/apps
CLIENT_ID = "your_client_id_here"  # Replace with your actual client ID
CLIENT_SECRET = "your_client_secret_here"  # Replace with your actual client secret
REDIRECT_URI = "http://localhost:3000"

class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, callback_data=None, **kwargs):
        self.callback_data = callback_data
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path.startswith('/?code='):
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            if 'code' in query_params:
                code = query_params['code'][0]
                self.callback_data['code'] = code
                self.callback_data['received'] = True
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                response_html = """
                <!DOCTYPE html>
                <html>
                <head><title>Success</title></head>
                <body style="font-family: Arial; text-align: center; margin-top: 100px;">
                    <h2 style="color: green;">✓ Authorization successful!</h2>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
                """
                self.wfile.write(response_html.encode())
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Waiting for callback...</h1></body></html>")
    
    def log_message(self, format, *args):
        pass

def start_callback_server(port=8080, timeout=300):
    callback_data = {'code': None, 'received': False}
    
    def handler(*args, **kwargs):
        return OAuthCallbackHandler(*args, callback_data=callback_data, **kwargs)
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Starting callback server on port {port}...")
            
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            start_time = time.time()
            while not callback_data['received'] and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            httpd.shutdown()
            server_thread.join(timeout=5)
            
            if callback_data['received']:
                return callback_data['code']
            else:
                print(f"Timeout waiting for OAuth callback ({timeout} seconds)")
                return None
                
    except Exception as e:
        print(f"Error starting callback server: {e}")
        return None

def exchange_code_for_tokens(client_id, client_secret, code, redirect_uri):
    try:
        url = "https://api.upstox.com/index/oauth/token"
        
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'client_id': client_id,
            'client_secret': client_secret,
            'redirect_uri': redirect_uri
        }
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        print("Exchanging authorization code for tokens...")
        response = requests.post(url, data=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            token_data = response.json()
            
            access_token = token_data.get('access_token')
            refresh_token = token_data.get('refresh_token')
            expires_in = token_data.get('expires_in', 3600)
            
            if access_token and refresh_token:
                print("✓ Successfully obtained tokens")
                return access_token, refresh_token, expires_in
            else:
                print(f"✗ Missing tokens in response: {token_data}")
                return None
        else:
            print(f"✗ Token exchange failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Token exchange error: {e}")
        return None

def test_upstox_api(access_token):
    try:
        print("Testing Upstox API with new token...")
        
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
                price = data['data']['NSE_EQ|INE002A01018']['last_price']
                print(f"✓ Upstox API test successful! RELIANCE: ₹{price}")
                return True
            else:
                print(f"✗ API returned error: {data.get('message')}")
                return False
        else:
            print(f"✗ API test failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ API test error: {e}")
        return False

def main():
    print("=" * 60)
    print("UPSTOX OAUTH SETUP (AUTOMATED)")
    print("=" * 60)
    print(f"Client ID: {CLIENT_ID}")
    print(f"Redirect URI: {REDIRECT_URI}")
    print()
    
    # Build authorization URL
    auth_url = (
        f"https://api.upstox.com/index/oauth/authorize?"
        f"response_type=code&"
        f"client_id={CLIENT_ID}&"
        f"redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
    )
    
    print("Opening authorization URL in your browser...")
    print(f"URL: {auth_url}")
    print()
    
    # Start callback server
    print("Starting local server to receive authorization callback...")
    print(f"Listening on: {REDIRECT_URI}")
    print()
    
    # Manual browser opening
    print("Please open the following URL manually in your browser:")
    print(f"URL: {auth_url}")
    print()
    print("Steps:")
    print("1. Copy the URL above")
    print("2. Paste it in your browser")
    print("3. Log in to Upstox with your credentials")
    print("4. Authorize the application")
    print("5. You will be redirected to a success page")
    
    print()
    print("Please complete the authorization in your browser...")
    print("The local server will automatically receive the callback.")
    print()
    
    # Wait for callback
    auth_code = start_callback_server(port=3000, timeout=300)
    
    if not auth_code:
        print("✗ Authorization failed or timed out")
        return False
    
    print("✓ Authorization code received")
    print()
    
    print("=" * 60)
    print("STEP 2: Token Exchange")
    print("=" * 60)
    
    # Exchange code for tokens
    token_result = exchange_code_for_tokens(CLIENT_ID, CLIENT_SECRET, auth_code, REDIRECT_URI)
    
    if not token_result:
        print("✗ Token exchange failed")
        return False
    
    access_token, refresh_token, expires_in = token_result
    
    print("=" * 60)
    print("STEP 3: Save Tokens")
    print("=" * 60)
    
    # Initialize token manager and save tokens
    try:
        token_manager = UpstoxTokenManager()
        token_manager.set_oauth_credentials(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
        token_manager._save_tokens(access_token, refresh_token, expires_in)
        
        print("✓ Tokens saved to .env and cache files")
        
    except Exception as e:
        print(f"✗ Failed to save tokens: {e}")
        return False
    
    print()
    print("=" * 60)
    print("STEP 4: Test API")
    print("=" * 60)
    
    # Test the API
    if test_upstox_api(access_token):
        print()
        print("=" * 60)
        print("✓ SETUP COMPLETE!")
        print("=" * 60)
        print()
        print("Your Upstox OAuth credentials have been successfully configured.")
        print("The system will now automatically refresh tokens when needed.")
        print()
        print("Next steps:")
        print("1. Test with: python test_upstox_api.py")
        print("2. Your backend will automatically use the new tokens")
        print("3. No manual token refresh needed - it's all automatic!")
        print()
        return True
    else:
        print()
        print("✗ API test failed, but tokens were saved.")
        print("Please check your Upstox app permissions and try again.")
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
