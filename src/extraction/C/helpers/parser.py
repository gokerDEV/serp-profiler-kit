from bs4 import BeautifulSoup
import json
import os
import re
import trafilatura
from urllib.parse import urlparse

# Patterns for cleanup
BAD_TAGS = ["script", "style", "noscript", "svg", "canvas", "iframe", "form", "nav", "footer", "aside", "header"]
BAD_ATTR_PATTERN = re.compile(r"cookie|consent|newsletter|login|subscribe|paywall|popup|ad-wrapper", re.I)

def simple_tokenize(text):
    """Regex based word count."""
    if not text:
        return 0
    return len(re.findall(r'\w+', text, re.UNICODE))

def clean_soup(soup):
    """
    Remove boilerplate elements from soup.
    """
    # 1. Remove specific tags
    for tag in soup(BAD_TAGS):
        tag.decompose()
    
    # 2. Remove elements by ID/Class pattern
    # Doing a full pass might be expensive, so we use find_all with regex
    # Removing by ID
    for tag in soup.find_all(id=BAD_ATTR_PATTERN):
        tag.decompose()
        
    # Removing by Class (class is a list in bs4)
    # regex matches against the string representation of class attribute
    for tag in soup.find_all(class_=BAD_ATTR_PATTERN):
        tag.decompose()
    
    # Removing by ARIA label
    for tag in soup.find_all(attrs={"aria-label": BAD_ATTR_PATTERN}):
        tag.decompose()
        
    return soup

def extract_targeted(soup):
    """
    Try to find main content in common semantic containers.
    """
    selectors = [
        "article", 
        "main", 
        '[role="main"]', 
        ".post-content", 
        ".entry-content", 
        ".article-body", 
        ".story-body",
        ".content-body",
        "#main-content"
    ]
    
    candidates = []
    
    for sel in selectors:
        elements = soup.select(sel)
        for el in elements:
            text = el.get_text(" ", strip=True)
            # Threshold to consider candidate valid
            if len(text) > 200: 
                candidates.append((len(text), text))
    
    if candidates:
        # Sort by length descending and return longest
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
        
    return None

def extract_content(html_path):
    """
    Extracts structural information, content, and metadata from an HTML file.
    Returns a dictionary suitable for JSON serialization.
    """
    result = {
        'status': 'ok',
        'status_reason': None,
        'extractor_used': None,
        'text_quality_flag': 'ok'
    }

    try:
        # Check for .html vs .html.html extension issue if direct path fails
        if not os.path.exists(html_path):
            if os.path.exists(html_path + ".html"):
                html_path += ".html"
            else:
                 return {'status': 'error', 'error': 'file_not_found', 'status_reason': 'file_not_found'}

        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html = f.read()

        soup = BeautifulSoup(html, 'html.parser')

        # 1. Metadata (Expanded Fallback)
        # Title
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        
        if not title:
            meta_title = soup.find('meta', attrs={'property': 'og:title'}) or \
                         soup.find('meta', attrs={'name': 'twitter:title'})
            if meta_title:
                title = meta_title.get('content', '').strip()
                
        # Description
        description = None
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or \
                    soup.find('meta', attrs={'property': 'og:description'}) or \
                    soup.find('meta', attrs={'name': 'twitter:description'})
        if meta_desc:
            description = meta_desc.get('content', '').strip()

        # Canonical
        canonical = None
        link_canon = soup.find('link', attrs={'rel': 'canonical'})
        if link_canon:
            canonical = link_canon.get('href', '').strip()

        # 2. Structure Counts (Memory Optimized)
        tag_count = sum(1 for _ in soup.find_all(True))
        
        # Links structure
        links = soup.find_all('a', href=True)
        link_list = []
        for l in links:
             link_list.append({
                 'href': l['href'],
                 'text': l.get_text(" ", strip=True),
                 'title': l.attrs.get('title', '')
             })
        link_count = len(links)
        
        # Images structure
        images = soup.find_all('img')
        image_list = []
        for i in images:
            image_list.append({
                'url': i.get('src', ''),
                'alt': i.get('alt', ''),
                'title': i.get('title', '')
            })
        image_count = len(images)
        
        # Scripts
        scripts = soup.find_all('script')
        script_list = [s.get('src') for s in scripts if s.get('src')] 
        script_count = len(scripts)

        # 3. Headings (H1-H6)
        h_tags = {}
        h_counts = {}
        for i in range(1, 7):
            h_tag = f'h{i}'
            headers = soup.find_all(h_tag)
            h_tags[h_tag] = [h.get_text(" ", strip=True) for h in headers]
            h_counts[h_tag] = len(headers)

        # 4. Content Extraction (3-Stage Fallback)
        main_content = None
        
        # Stage 1: Trafilatura (Best)
        try:
            main_content = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=True) # no_fallback=True to rely on our own fallback
            if main_content:
                result['extractor_used'] = 'trafilatura'
        except Exception:
            pass # Continue to fallback

        # Fallback Logic
        if not main_content:
            # Prepare clean soup (Boilerplate removal)
            clean_s = clean_soup(soup) # Note: this modifies soup in place!
            
            # Stage 2: Targeted Extraction
            main_content = extract_targeted(clean_s)
            if main_content:
                result['extractor_used'] = 'targeted_fallback'
            else:
                # Stage 3: Global Text (Cleaned)
                main_content = clean_s.get_text(" ", strip=True)
                if main_content:
                    result['extractor_used'] = 'fallback_global'
                    result['text_quality_flag'] = 'noisy_fallback'
                else:
                    result['extractor_used'] = 'failed'
                    result['text_quality_flag'] = 'empty'

        # Word Count (Regex based)
        word_count = simple_tokenize(main_content)
        
        # 5. Schema / Metadata Flags
        has_schema = bool(soup.find('script', attrs={'type': 'application/ld+json'}) or soup.find(attrs={"itemtype": True}))
        
        # Meta Published Time
        meta_pub = soup.find('meta', attrs={'property': 'article:published_time'}) or \
                   soup.find('meta', attrs={'itemprop': 'datePublished'}) or \
                   soup.find('meta', attrs={'name': 'date'})
        published_time = meta_pub.get('content') if meta_pub else None
        has_published_time = bool(published_time)
        
        # Meta Author
        meta_auth = soup.find('meta', attrs={'name': 'author'}) or \
                    soup.find('meta', attrs={'property': 'article:author'})
        
        author = None
        if meta_auth:
            author = meta_auth.get('content')
            
        if not author:
            # Fallback to link rel=author
            link_auth = soup.find('link', attrs={'rel': 'author'})
            if link_auth:
                author = link_auth.get('href') # Fallback to href if content not present (it usually isn't on link tags)

        has_author = bool(author)

        # Assemble Result
        result.update({
            'title': title,
            'description': description,
            'canonical': canonical,
            'content': main_content,
            'content_length': len(main_content) if main_content else 0,
            'word_count': word_count,
            'tag_count': tag_count,
            'link_count': link_count,
            'image_count': image_count,
            'script_count': script_count,
            'h_tags': h_tags,
            'h_counts': h_counts,
            'has_schema': has_schema,
            'has_published_time': has_published_time,
            'published_time': published_time,
            'has_author': has_author,
            'author': author,
            # Lists
            'links': link_list[:500], 
            'images': image_list[:100],
            'scripts': script_list[:100],
        })
        
        return result

    except Exception as e:
        return {'status': 'error', 'error': str(e), 'status_reason': 'script_exception'}
