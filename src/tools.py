"""
Tools for web search and content reading
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import httpx
from bs4 import BeautifulSoup
import json
import hashlib
import pickle
import asyncio
import logging

# Try to load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip



class SearchTool:
    """
    Tool for searching the web using Tavily API with caching
    """

    def __init__(self, api_key: Optional[str] = None):
        # Configure search APIs
        self.tavily_api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")  # Fallback

        # Cache settings
        self.cache_dir = os.path.expanduser("~/.research_weaver_cache")
        self.cache_duration = timedelta(hours=24)  # Cache for 24 hours
        os.makedirs(self.cache_dir, exist_ok=True)

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web and return results with caching

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, url, and snippet
        """
        # Check cache first
        cache_key = self._get_cache_key(query, max_results)
        cached_results = self._get_cached_results(cache_key)

        if cached_results is not None:
            print(f"Using cached results for query: {query}")
            return cached_results

        # Try different search backends in priority order
        results = []

        if self.tavily_api_key:
            results = await self._search_tavily(query, max_results)
        elif self.serper_api_key:
            results = await self._search_serper(query, max_results)
        else:
            raise ValueError("Please set TAVILY_API_KEY or SERPER_API_KEY environment variable.")
            return []

        # Cache the results
        if results:
            self._cache_results(cache_key, results)

        return results

    async def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Search using Tavily API
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Content-Type": "application/json"},
                    json={
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": "basic",
                        "include_answer": False,
                        "include_images": False,
                        "include_raw_content": False,
                        "max_results": max_results,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    results = []

                    for item in data.get("results", []):
                        results.append(
                            {
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "snippet": item.get("content", ""),
                            }
                        )

                    print(f"Tavily search returned {len(results)} results for: {query}")
                    return results
                else:
                    print(f"Tavily API error: {response.status_code} - {response.text}")
                    return []

            except Exception as e:
                print(f"Error searching with Tavily: {e}")
                return []

    async def _search_serper(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Search using Serper API
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": max_results},
                )

                if response.status_code == 200:
                    data = response.json()
                    results = []

                    for item in data.get("organic", [])[:max_results]:
                        results.append(
                            {
                                "title": item.get("title", ""),
                                "url": item.get("link", ""),
                                "snippet": item.get("snippet", ""),
                            }
                        )

                    return results
                else:
                    print(f"Serper API error: {response.status_code}")
                    return []

            except Exception as e:
                print(f"Error searching with Serper: {e}")
                return []

    def _get_cache_key(self, query: str, max_results: int) -> str:
        """
        Generate cache key for search query
        """
        cache_string = f"{query}|{max_results}"
        return hashlib.md5(cache_string.encode("utf-8")).hexdigest()

    def _get_cached_results(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search results if they exist and are not expired
        """
        cache_file = os.path.join(self.cache_dir, f"search_{cache_key}.pkl")

        try:
            if os.path.exists(cache_file):
                # Check if cache is still valid
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < self.cache_duration:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                else:
                    # Cache expired, remove file
                    os.remove(cache_file)
        except Exception as e:
            print(f"Error reading cache: {e}")

        return None

    def _cache_results(self, cache_key: str, results: List[Dict[str, Any]]):
        """
        Cache search results
        """
        cache_file = os.path.join(self.cache_dir, f"search_{cache_key}.pkl")

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error caching results: {e}")


class ReadTool:
    """
    Tool for reading and extracting content from web pages
    """

    def __init__(self, search_tool=None, max_concurrent_reads=5):
        self.timeout = 30
        self.search_tool = search_tool  # Reference to SearchTool for content cache
        self.max_concurrent_reads = max_concurrent_reads
        self.semaphore = asyncio.Semaphore(max_concurrent_reads)

        # Web scraping cache settings
        self.cache_dir = os.path.expanduser("~/.research_weaver_cache")
        self.cache_duration = timedelta(hours=24)  # Cache for 24 hours
        os.makedirs(self.cache_dir, exist_ok=True)

    async def read_url(self, url: str) -> str:
        """
        Read and extract main content from a URL

        Args:
            url: URL to read

        Returns:
            Extracted text content
        """
        # Check web scraping cache
        scraped_content = self._get_cached_content(url)
        if scraped_content:
            logging.debug(f"Using cached content for {url[:50]}...")
            return scraped_content

        # Fetch and parse content from web
        logging.debug(f"Fetching content from {url[:50]}...")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                response = await client.get(
                    url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
                )

                if response.status_code == 200:
                    # Parse HTML
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Try to find main content
                    main_content = None

                    # Common content selectors
                    content_selectors = [
                        "main",
                        "article",
                        '[role="main"]',
                        "#main-content",
                        ".main-content",
                        "#content",
                        ".content",
                        ".post-content",
                        ".entry-content",
                    ]

                    for selector in content_selectors:
                        main_content = soup.select_one(selector)
                        if main_content:
                            break

                    # If no main content found, use body
                    if not main_content:
                        main_content = soup.body

                    if main_content:
                        # Extract text
                        text = main_content.get_text(separator="\n", strip=True)

                        # Clean up excessive whitespace
                        lines = [line.strip() for line in text.split("\n") if line.strip()]
                        text = "\n".join(lines)

                        # Limit length
                        if len(text) > 5000:
                            text = text[:5000] + "...\n[Content truncated]"

                        result = f"Content from {url}:\n\n{text}"

                        # Cache the scraped content
                        self._cache_content(url, result)

                        return result
                    else:
                        return f"Could not extract content from {url}"

                else:
                    return f"Error reading {url}: HTTP {response.status_code}"

            except httpx.TimeoutException:
                return f"Timeout reading {url}"
            except Exception as e:
                return f"Error reading {url}: {str(e)}"

    async def read_urls_concurrent(self, urls: List[str]) -> Dict[str, str]:
        """
        Read multiple URLs concurrently with controlled concurrency
        
        Args:
            urls: List of URLs to read
            
        Returns:
            Dictionary mapping URL to content
        """
        logging.info(f"Starting concurrent read of {len(urls)} URLs")
        
        # Create tasks for all URLs
        tasks = [self._read_url_with_semaphore(url) for url in urls]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        url_content_map = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logging.error(f"Error reading {url}: {result}")
                url_content_map[url] = f"Error reading {url}: {str(result)}"
            else:
                url_content_map[url] = result
                
        logging.info(f"Completed concurrent read of {len(urls)} URLs")
        return url_content_map

    async def _read_url_with_semaphore(self, url: str) -> str:
        """
        Read URL with semaphore control for concurrency limiting
        """
        async with self.semaphore:
            return await self.read_url(url)

    async def read_urls_batch(self, urls: List[str], batch_size: int = None) -> Dict[str, str]:
        """
        Read URLs in batches to avoid overwhelming the system
        
        Args:
            urls: List of URLs to read
            batch_size: Size of each batch (defaults to max_concurrent_reads)
            
        Returns:
            Dictionary mapping URL to content
        """
        if batch_size is None:
            batch_size = self.max_concurrent_reads
            
        logging.info(f"Reading {len(urls)} URLs in batches of {batch_size}")
        
        all_results = {}
        
        # Process URLs in batches
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}: {len(batch)} URLs")
            
            batch_results = await self.read_urls_concurrent(batch)
            all_results.update(batch_results)
            
            # Small delay between batches to be respectful to servers
            if i + batch_size < len(urls):
                await asyncio.sleep(0.5)
                
        return all_results

    def _get_cache_key(self, url: str) -> str:
        """
        Generate cache key for URL content
        """
        return hashlib.md5(url.encode("utf-8")).hexdigest()

    def _get_cached_content(self, url: str) -> Optional[str]:
        """
        Get cached content for URL if it exists and is not expired
        """
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"readtool_{cache_key}.json")

        try:
            if os.path.exists(cache_file):
                # Check if cache is still valid
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < self.cache_duration:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                        return cache_data.get("content", "")
                else:
                    # Cache expired, remove file
                    os.remove(cache_file)
        except Exception as e:
            print(f"Error reading ReadTool cache: {e}")

        return None

    def _cache_content(self, url: str, content: str):
        """
        Cache scraped content for URL
        """
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"readtool_{cache_key}.json")

        try:
            cache_data = {"url": url, "content": content, "cached_at": datetime.now().isoformat()}
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logging.debug(f"Cached scraped content for {url[:50]}...")
        except Exception as e:
            logging.error(f"Error caching ReadTool content: {e}")


# Tool registry for easy access
def create_tools(max_concurrent_reads=5):
    """
    Create and return tool instances
    
    Args:
        max_concurrent_reads: Maximum number of concurrent read operations
    """
    search_tool = SearchTool()
    read_tool = ReadTool(search_tool=search_tool, max_concurrent_reads=max_concurrent_reads)

    return {"search": search_tool, "read": read_tool}


def create_llm_client():
    """
    Create LLM client using DeepSeek as default
    """
    from .llm_adapter import create_llm_adapter, CompatibleLLMClient

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set DEEPSEEK_API_KEY environment variable. " "Get your API key from https://platform.deepseek.com/"
        )

    adapter = create_llm_adapter(provider="deepseek", api_key=api_key)
    return CompatibleLLMClient(adapter)
