"""
Browser compatibility and cross-platform testing
"""

import pytest
from unittest.mock import patch


class TestBrowserCompatibility:
    """Test browser compatibility and responsive design"""
    
    def test_responsive_meta_tags(self, client, mock_api_response):
        """Test responsive design meta tags"""
        with patch('app.requests.get') as mock_get:
            mock_get.return_value = mock_api_response(200, {'groups': []})
            
            response = client.get('/')
            
            # Check for responsive viewport meta tag
            assert b'name="viewport"' in response.data
            assert b'width=device-width' in response.data
            assert b'initial-scale=1' in response.data
    
    def test_css_framework_compatibility(self, client, mock_api_response):
        """Test CSS framework compatibility"""
        with patch('app.requests.get') as mock_get:
            mock_get.return_value = mock_api_response(200, {'groups': []})
            
            response = client.get('/')
            
            # Check for Bootstrap classes
            bootstrap_classes = [
                b'container', b'row', b'col-', b'btn', b'card',
                b'navbar', b'alert', b'modal', b'form-control'
            ]
            
            for css_class in bootstrap_classes:
                assert css_class in response.data, f"Missing Bootstrap class: {css_class}"
    
    def test_javascript_compatibility(self, client, mock_api_response):
        """Test JavaScript compatibility features"""
        with patch('app.requests.get') as mock_get:
            mock_get.return_value = mock_api_response(200, {'groups': []})
            
            response = client.get('/chat/general')
            
            # Check for modern JavaScript features with fallbacks
            assert b'addEventListener' in response.data
            assert b'fetch(' in response.data
            
            # Check for error handling in JavaScript
            assert b'catch' in response.data
            assert b'try' in response.data
    
    def test_progressive_enhancement(self, client, mock_api_response):
        """Test progressive enhancement principles"""
        with patch('app.requests.get') as mock_get:
            mock_get.return_value = mock_api_response(200, {'groups': []})
            
            response = client.get('/')
            
            # Core functionality should work without JavaScript
            # Forms should have proper action attributes
            assert b'method="POST"' in response.data
            assert b'action=' in response.data
            
            # Links should have proper href attributes
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.data, 'html.parser')
            links = soup.find_all('a')
            
            for link in links:
                href = link.get('href')
                if href and not href.startswith('#'):
                    assert href.startswith(('/', 'http')), f"Invalid link href: {href}"

