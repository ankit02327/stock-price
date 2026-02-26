"""
Test suite for Upstox token refresh functionality.

Tests token loading, proactive refresh, reactive refresh, and error handling.
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.upstox_token_manager import UpstoxTokenManager

class TestUpstoxTokenManager(unittest.TestCase):
    """Test cases for UpstoxTokenManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_env_file = os.path.join(self.test_dir, '.env')
        self.test_cache_file = os.path.join(self.test_dir, 'upstox_tokens.json')
        
        # Create cache directory
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Sample test data
        self.test_client_id = "test_client_id"
        self.test_client_secret = "test_client_secret"
        self.test_access_token = "test_access_token_12345"
        self.test_refresh_token = "test_refresh_token_67890"
        self.test_redirect_uri = "http://localhost:8080"
        
        # Initialize token manager with test paths
        self.token_manager = UpstoxTokenManager(env_file_path=self.test_env_file)
        self.token_manager.cache_file = self.test_cache_file
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_test_env_file(self, **kwargs):
        """Create test .env file with given values."""
        env_content = []
        
        defaults = {
            'UPSTOX_CLIENT_ID': self.test_client_id,
            'UPSTOX_CLIENT_SECRET': self.test_client_secret,
            'UPSTOX_ACCESS_TOKEN': self.test_access_token,
            'UPSTOX_REFRESH_TOKEN': self.test_refresh_token,
            'UPSTOX_REDIRECT_URI': self.test_redirect_uri,
            'UPSTOX_TOKEN_EXPIRY': (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        defaults.update(kwargs)
        
        for key, value in defaults.items():
            if value is not None:
                env_content.append(f"{key}={value}")
        
        with open(self.test_env_file, 'w') as f:
            f.write('\n'.join(env_content))
    
    def create_test_cache_file(self, **kwargs):
        """Create test cache file with given values."""
        cache_data = {
            'client_id': self.test_client_id,
            'client_secret': self.test_client_secret,
            'access_token': self.test_access_token,
            'refresh_token': self.test_refresh_token,
            'redirect_uri': self.test_redirect_uri,
            'token_expiry': (datetime.now() + timedelta(hours=1)).isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        cache_data.update(kwargs)
        
        with open(self.test_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def test_load_tokens_from_env(self):
        """Test loading tokens from .env file."""
        self.create_test_env_file()
        self.token_manager._load_tokens()
        
        self.assertEqual(self.token_manager.client_id, self.test_client_id)
        self.assertEqual(self.token_manager.client_secret, self.test_client_secret)
        self.assertEqual(self.token_manager.access_token, self.test_access_token)
        self.assertEqual(self.token_manager.refresh_token, self.test_refresh_token)
        self.assertEqual(self.token_manager.redirect_uri, self.test_redirect_uri)
        self.assertIsNotNone(self.token_manager.token_expiry)
    
    def test_load_tokens_from_cache_fallback(self):
        """Test loading tokens from cache when .env is missing."""
        self.create_test_cache_file()
        self.token_manager._load_tokens()
        
        self.assertEqual(self.token_manager.client_id, self.test_client_id)
        self.assertEqual(self.token_manager.client_secret, self.test_client_secret)
        self.assertEqual(self.token_manager.access_token, self.test_access_token)
        self.assertEqual(self.token_manager.refresh_token, self.test_refresh_token)
    
    def test_load_tokens_priority(self):
        """Test that .env takes priority over cache."""
        # Create both files with different values
        self.create_test_env_file(UPSTOX_ACCESS_TOKEN="env_token")
        self.create_test_cache_file(access_token="cache_token")
        
        self.token_manager._load_tokens()
        
        # Should use .env value
        self.assertEqual(self.token_manager.access_token, "env_token")
    
    def test_is_token_expired(self):
        """Test token expiry detection."""
        # Test with future expiry
        future_expiry = datetime.now() + timedelta(hours=1)
        self.token_manager.token_expiry = future_expiry
        self.assertFalse(self.token_manager.is_token_expired())
        
        # Test with past expiry
        past_expiry = datetime.now() - timedelta(hours=1)
        self.token_manager.token_expiry = past_expiry
        self.assertTrue(self.token_manager.is_token_expired())
        
        # Test with expiry within threshold (15 minutes)
        near_expiry = datetime.now() + timedelta(minutes=10)
        self.token_manager.token_expiry = near_expiry
        self.assertTrue(self.token_manager.is_token_expired())
        
        # Test with no expiry info
        self.token_manager.token_expiry = None
        self.assertTrue(self.token_manager.is_token_expired())
    
    @patch('requests.post')
    def test_refresh_access_token_success(self, mock_post):
        """Test successful token refresh."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'access_token': 'new_access_token',
            'refresh_token': 'new_refresh_token',
            'expires_in': 3600
        }
        mock_post.return_value = mock_response
        
        # Set up initial tokens
        self.token_manager.client_id = self.test_client_id
        self.token_manager.client_secret = self.test_client_secret
        self.token_manager.refresh_token = self.test_refresh_token
        self.token_manager.redirect_uri = self.test_redirect_uri
        
        # Test refresh
        result = self.token_manager.refresh_access_token()
        
        self.assertTrue(result)
        self.assertEqual(self.token_manager.access_token, 'new_access_token')
        self.assertEqual(self.token_manager.refresh_token, 'new_refresh_token')
        self.assertIsNotNone(self.token_manager.token_expiry)
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn('data', call_args[1])
        self.assertEqual(call_args[1]['data']['grant_type'], 'refresh_token')
    
    @patch('requests.post')
    def test_refresh_access_token_failure(self, mock_post):
        """Test token refresh failure."""
        # Mock failure response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {'error': 'invalid_grant'}
        mock_response.text = '{"error": "invalid_grant"}'
        mock_post.return_value = mock_response
        
        # Set up initial tokens
        self.token_manager.client_id = self.test_client_id
        self.token_manager.client_secret = self.test_client_secret
        self.token_manager.refresh_token = self.test_refresh_token
        
        # Test refresh
        result = self.token_manager.refresh_access_token()
        
        self.assertFalse(result)
    
    def test_get_valid_token_with_fresh_token(self):
        """Test getting valid token when token is still fresh."""
        # Set up fresh token
        future_expiry = datetime.now() + timedelta(hours=1)
        self.token_manager.access_token = self.test_access_token
        self.token_manager.token_expiry = future_expiry
        
        # Mock refresh to ensure it's not called
        with patch.object(self.token_manager, 'refresh_access_token') as mock_refresh:
            result = self.token_manager.get_valid_token()
            
            self.assertEqual(result, self.test_access_token)
            mock_refresh.assert_not_called()
    
    def test_get_valid_token_with_expired_token(self):
        """Test getting valid token when token is expired."""
        # Set up expired token
        past_expiry = datetime.now() - timedelta(hours=1)
        self.token_manager.access_token = self.test_access_token
        self.token_manager.token_expiry = past_expiry
        
        # Mock successful refresh
        with patch.object(self.token_manager, 'refresh_access_token', return_value=True):
            with patch.object(self.token_manager, '_save_tokens') as mock_save:
                result = self.token_manager.get_valid_token()
                
                # Should return the token (refresh is mocked to succeed)
                self.assertEqual(result, self.test_access_token)
    
    def test_handle_401_error(self):
        """Test handling 401 error with token refresh."""
        # Mock successful refresh
        with patch.object(self.token_manager, 'refresh_access_token', return_value=True) as mock_refresh:
            result = self.token_manager.handle_401_error()
            
            self.assertTrue(result)
            mock_refresh.assert_called_once()
    
    def test_handle_401_error_failure(self):
        """Test handling 401 error when refresh fails."""
        # Mock failed refresh
        with patch.object(self.token_manager, 'refresh_access_token', return_value=False) as mock_refresh:
            result = self.token_manager.handle_401_error()
            
            self.assertFalse(result)
            mock_refresh.assert_called_once()
    
    def test_save_tokens(self):
        """Test saving tokens to both .env and cache."""
        # Set up token manager
        self.token_manager.client_id = self.test_client_id
        self.token_manager.client_secret = self.test_client_secret
        
        # Test saving tokens
        self.token_manager._save_tokens('new_access', 'new_refresh', 3600)
        
        # Verify cache file was created
        self.assertTrue(os.path.exists(self.test_cache_file))
        
        # Verify cache content
        with open(self.test_cache_file, 'r') as f:
            cache_data = json.load(f)
        
        self.assertEqual(cache_data['access_token'], 'new_access')
        self.assertEqual(cache_data['refresh_token'], 'new_refresh')
        self.assertEqual(cache_data['client_id'], self.test_client_id)
        self.assertIn('token_expiry', cache_data)
        
        # Verify .env file was updated
        self.assertTrue(os.path.exists(self.test_env_file))
        
        # Read .env content
        with open(self.test_env_file, 'r') as f:
            env_content = f.read()
        
        self.assertIn('UPSTOX_ACCESS_TOKEN=new_access', env_content)
        self.assertIn('UPSTOX_REFRESH_TOKEN=new_refresh', env_content)
    
    def test_set_oauth_credentials(self):
        """Test setting OAuth credentials."""
        self.token_manager.set_oauth_credentials(
            'new_client_id', 
            'new_client_secret', 
            'http://localhost:9000'
        )
        
        self.assertEqual(self.token_manager.client_id, 'new_client_id')
        self.assertEqual(self.token_manager.client_secret, 'new_client_secret')
        self.assertEqual(self.token_manager.redirect_uri, 'http://localhost:9000')
        
        # Verify cache file was created/updated
        self.assertTrue(os.path.exists(self.test_cache_file))
        
        with open(self.test_cache_file, 'r') as f:
            cache_data = json.load(f)
        
        self.assertEqual(cache_data['client_id'], 'new_client_id')
        self.assertEqual(cache_data['client_secret'], 'new_client_secret')
    
    def test_has_valid_credentials(self):
        """Test checking if valid credentials are available."""
        # Test with missing credentials
        self.assertFalse(self.token_manager.has_valid_credentials())
        
        # Test with all credentials
        self.token_manager.client_id = self.test_client_id
        self.token_manager.client_secret = self.test_client_secret
        self.token_manager.access_token = self.test_access_token
        self.token_manager.refresh_token = self.test_refresh_token
        
        self.assertTrue(self.token_manager.has_valid_credentials())
    
    def test_get_token_info(self):
        """Test getting token information for debugging."""
        # Set up test data
        future_expiry = datetime.now() + timedelta(hours=1)
        self.token_manager.access_token = self.test_access_token
        self.token_manager.refresh_token = self.test_refresh_token
        self.token_manager.client_id = self.test_client_id
        self.token_manager.client_secret = self.test_client_secret
        self.token_manager.token_expiry = future_expiry
        
        info = self.token_manager.get_token_info()
        
        self.assertTrue(info['has_access_token'])
        self.assertTrue(info['has_refresh_token'])
        self.assertTrue(info['has_client_credentials'])
        self.assertFalse(info['is_expired'])
        self.assertIsNotNone(info['token_expiry'])
        self.assertGreater(info['minutes_until_expiry'], 0)

class TestTokenManagerIntegration(unittest.TestCase):
    """Integration tests for token manager with real file operations."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_env_file = os.path.join(self.test_dir, '.env')
        
        # Create cache directory
        cache_dir = os.path.join(self.test_dir, '_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize token manager
        self.token_manager = UpstoxTokenManager(env_file_path=self.test_env_file)
        self.token_manager.cache_file = os.path.join(cache_dir, 'upstox_tokens.json')
    
    def tearDown(self):
        """Clean up integration test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_dual_storage_sync(self):
        """Test that tokens are synced between .env and cache."""
        # Create initial .env file
        with open(self.test_env_file, 'w') as f:
            f.write("UPSTOX_CLIENT_ID=test_client\n")
            f.write("UPSTOX_CLIENT_SECRET=test_secret\n")
        
        # Set credentials and save tokens
        self.token_manager.set_oauth_credentials('test_client', 'test_secret')
        self.token_manager._save_tokens('access123', 'refresh456', 3600)
        
        # Verify both files exist and have correct content
        self.assertTrue(os.path.exists(self.test_env_file))
        self.assertTrue(os.path.exists(self.token_manager.cache_file))
        
        # Check .env content
        with open(self.test_env_file, 'r') as f:
            env_content = f.read()
        
        self.assertIn('UPSTOX_ACCESS_TOKEN=access123', env_content)
        self.assertIn('UPSTOX_REFRESH_TOKEN=refresh456', env_content)
        
        # Check cache content
        with open(self.token_manager.cache_file, 'r') as f:
            cache_data = json.load(f)
        
        self.assertEqual(cache_data['access_token'], 'access123')
        self.assertEqual(cache_data['refresh_token'], 'refresh456')
    
    def test_cache_fallback_when_env_missing(self):
        """Test that cache is used when .env file is missing."""
        # Create only cache file
        with open(self.token_manager.cache_file, 'w') as f:
            json.dump({
                'client_id': 'cache_client',
                'client_secret': 'cache_secret',
                'access_token': 'cache_access',
                'refresh_token': 'cache_refresh',
                'token_expiry': (datetime.now() + timedelta(hours=1)).isoformat()
            }, f)
        
        # Load tokens (should use cache)
        self.token_manager._load_tokens()
        
        self.assertEqual(self.token_manager.client_id, 'cache_client')
        self.assertEqual(self.token_manager.access_token, 'cache_access')

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
