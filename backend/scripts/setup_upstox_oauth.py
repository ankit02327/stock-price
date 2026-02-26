#!/usr/bin/env python3
"""
Upstox OAuth Setup Script

Interactive script to help set up Upstox OAuth2 credentials.
Guides user through the authorization flow and saves tokens automatically.
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

class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Handle OAuth callback from Upstox."""
    
    def __init__(self, *args, callback_data=None, **kwargs):
        self.callback_data = callback_data
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET request from OAuth callback."""
        if self.path.startswith('/?code='):
            # Extract authorization code
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            if 'code' in query_params:
                code = query_params['code'][0]
                self.callback_data['code'] = code
                self.callback_data['received'] = True
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                response_html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Upstox OAuth Success</title>
                    <style>
                        body { font-family: Arial, sans-serif; text-align: center; margin-top: 100px; }
                        .success { color: green; font-size: 18px; }
                        .info { color: #666; margin-top: 20px; }
                    </style>
                </head>
                <body>
                    <div class="success">✓ Authorization successful!</div>
                    <div class="info">You can close this window and return to the terminal.</div>
                </body>
                </html>
                """
                self.wfile.write(response_html.encode())
            else:
                # Send error response
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                error_html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Upstox OAuth Error</title>
                    <style>
                        body { font-family: Arial, sans-serif; text-align: center; margin-top: 100px; }
                        .error { color: red; font-size: 18px; }
                        .info { color: #666; margin-top: 20px; }
                    </style>
                </head>
                <body>
                    <div class="error">✗ Authorization failed!</div>
                    <div class="info">Please check the terminal for error details.</div>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
        else:
            # Send default response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            default_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Upstox OAuth Setup</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; margin-top: 100px; }
                    .info { color: #666; }
                </style>
            </head>
            <body>
                <div class="info">Waiting for OAuth callback...</div>
            </body>
            </html>
            """
            self.wfile.write(default_html.encode())
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

def start_callback_server(port=3000, timeout=300):
    """
    Start HTTP server to receive OAuth callback.
    
    Args:
        port: Port to listen on
        timeout: Timeout in seconds
    
    Returns:
        Authorization code or None if timeout
    """
    callback_data = {'code': None, 'received': False}
    
    def handler(*args, **kwargs):
        return OAuthCallbackHandler(*args, callback_data=callback_data, **kwargs)
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Starting callback server on port {port}...")
            
            # Start server in a separate thread
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            # Wait for callback or timeout
            start_time = time.time()
            while not callback_data['received'] and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            # Shutdown server
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
    """
    Exchange authorization code for access and refresh tokens.
    
    Args:
        client_id: Upstox client ID
        client_secret: Upstox client secret
        code: Authorization code from callback
        redirect_uri: OAuth redirect URI
    
    Returns:
        Tuple of (access_token, refresh_token, expires_in) or None
    """
    try:
        url = "https://api.upstox.com/v2/login/authorization/token"
        
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
            refresh_token = token_data.get('refresh_token', '')  # Upstox v2 doesn't provide refresh tokens
            expires_in = token_data.get('expires_in', 3600)
            
            if access_token:
                print("✓ Successfully obtained access token")
                if not refresh_token:
                    print("  (Note: Upstox v2 uses JWT tokens without refresh tokens)")
                return access_token, refresh_token, expires_in
            else:
                print(f"✗ Missing access_token in response: {token_data}")
                return None
        else:
            print(f"✗ Token exchange failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Token exchange error: {e}")
        return None

def test_upstox_api(access_token):
    """
    Test Upstox API with the new access token.
    
    Args:
        access_token: Access token to test
    
    Returns:
        True if API test successful, False otherwise
    """
    try:
        print("Testing Upstox API with new token...")
        
        # Test with RELIANCE stock
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
                print("✓ Upstox API test successful!")
                return True
            else:
                print(f"✗ API returned error: {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"✗ API test failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ API test error: {e}")
        return False

def main():
    """Main OAuth setup flow."""
    print("=" * 60)
    print("UPSTOX OAUTH SETUP")
    print("=" * 60)
    print()
    
    # Get OAuth credentials
    print("Please provide your Upstox OAuth credentials:")
    print("(Get these from: https://upstox.com/developer/)")
    print()
    
    client_id = input("Client ID: ").strip()
    if not client_id:
        print("✗ Client ID is required")
        return False
    
    client_secret = input("Client Secret: ").strip()
    if not client_secret:
        print("✗ Client Secret is required")
        return False
    
    redirect_uri = input("Redirect URI [http://localhost:3000]: ").strip()
    if not redirect_uri:
        redirect_uri = "http://localhost:3000"
    
    print()
    print("=" * 60)
    print("STEP 1: Authorization")
    print("=" * 60)
    
    # Build authorization URL
    auth_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog?"
        f"response_type=code&"
        f"client_id={client_id}&"
        f"redirect_uri={urllib.parse.quote(redirect_uri)}"
    )
    
    print(f"Opening authorization URL in your browser...")
    print(f"URL: {auth_url}")
    print()
    print("If the browser doesn't open automatically, copy and paste the URL above.")
    print()
    
    # Start callback server
    print("Starting local server to receive authorization callback...")
    print(f"Listening on: {redirect_uri}")
    print()
    
    # Open browser
    try:
        webbrowser.open(auth_url)
        print("✓ Browser opened with authorization page")
    except Exception as e:
        print(f"⚠ Could not open browser automatically: {e}")
        print("Please open the URL manually in your browser.")
    
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
    token_result = exchange_code_for_tokens(client_id, client_secret, auth_code, redirect_uri)
    
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
        token_manager.set_oauth_credentials(client_id, client_secret, redirect_uri)
        
        # Calculate proper expiry time (next day 3:30 AM IST)
        now = datetime.now()
        tomorrow_330am = (now + timedelta(days=1)).replace(hour=3, minute=30, second=0, microsecond=0)
        expires_in_seconds = int((tomorrow_330am - now).total_seconds())
        
        token_manager._save_tokens(access_token, refresh_token, expires_in_seconds)
        
        print("✓ Tokens saved to JSON cache")
        print(f"✓ Token expires at: {tomorrow_330am.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print("✓ Token will automatically refresh before expiry")
        
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
        print("1. Test with: python backend/test_upstox_api.py")
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
