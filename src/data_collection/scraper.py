import asyncio
import csv
import os
import re
from typing import List, Dict, Set
from dataclasses import dataclass
import logging
from datetime import datetime
import pandas as pd
from pyppeteer.launcher import launch
import crawler
import sys
from pathlib import Path

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

@dataclass
class Result:
    url: str
    file_name: str
    success: bool
    error: str = None

class Scraper:
    def __init__(self, 
                 input_file: str,
                 output_dir: str,
                 exceptions_file: str,
                 screenshots_dir: str,
                 batch_size: int = 5,
                 skip_exceptions: bool = False,
                 quiet: bool = False,
                 browser_type: str = 'chromium',
                 headless: bool = True,
                 visit: str = None,
                 overwrite: bool = False):
        # Initialize paths
        self.input_file = input_file
        self.output_dir = output_dir
        self.screenshots_dir = screenshots_dir
        self.exceptions_file = exceptions_file
        self.log_file = 'data/logs/scraper.csv'
        
        # Create required directories
        for directory in [self.output_dir, self.screenshots_dir, os.path.dirname(self.log_file)]:
            if directory:
                os.makedirs(directory, exist_ok=True)
        
        # Initialize settings
        self.batch_size = batch_size
        self.skip_exceptions = skip_exceptions
        self.browser_type = browser_type.lower()
        self.headless = headless
        self.overwrite = overwrite
        
        # Setup logging
        self.logger = Utils.set_colorful_logging("Scraper")
        self.logger.setLevel(logging.ERROR if quiet else logging.INFO)
        
        # Initialize URLs
        self.urls = self._initialize_urls(visit)
        
        # Log configuration
        self._log_initialization()

    def _initialize_urls(self, visit: str = None) -> pd.DataFrame:
        """Initialize URLs DataFrame from input file or single URL"""
        if visit:
            return pd.DataFrame({'url': [visit]})
        urls = Utils.get_unique_urls(self.input_file)
        return pd.DataFrame({'url': urls})

    def _log_initialization(self):
        """Log initialization parameters"""
        params = {
            "Overwrite": self.overwrite,
            "Skip Exceptions": self.skip_exceptions,
            "Input File": self.input_file,
            "Output Directory": self.output_dir,
            "Screenshots Directory": self.screenshots_dir,
            "Browser": self.browser_type,
            "Headless": self.headless,
            "Exceptions File": self.exceptions_file,
            "Batch Size": self.batch_size
        }
        
        self.logger.info("="*50)
        self.logger.info("🔧 Initialization Parameters:")
        for key, value in params.items():
            self.logger.info(f"• {key}: {value}")
        self.logger.info("="*50)

    def get_unprocessed_urls(self) -> List[str]:
        """Get URLs that haven't been processed yet"""
        processed_files = self._get_processed_urls()
        exception_urls = self._get_exception_urls() if self.skip_exceptions else set()
        processed_files.update(exception_urls)
        
        unprocessed = []
        for url in self.urls['url']:
            if url != 'url' and (not processed_files or url not in processed_files):
                unprocessed.append(url)
        return unprocessed

    def _get_processed_urls(self) -> Set[str]:
        """Get set of already processed URLs from log file"""
        if not os.path.exists(self.log_file):
            return set()
        try:
            df = pd.read_csv(self.log_file)
            return set(df['url'].unique())
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load log file: {str(e)}")
            return set()

    def _get_exception_urls(self) -> Set[str]:
        """Get set of URLs with previous exceptions"""
        if not os.path.exists(self.exceptions_file):
            return set()
        try:
            df = pd.read_csv(self.exceptions_file)
            urls = set(df['url'].unique())
            self.logger.info(f"📋 Loaded {len(urls)} URLs with previous exceptions")
            return urls
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load exceptions file: {str(e)}")
            return set()

    async def _process_url(self, url: str, browser) -> Result:
        """Process a single URL"""
        idx = self._get_url_index(url)
        file_name = self._url_to_filename(url)
        ss_file_name = self._url_to_filename(url, self.screenshots_dir)
        
        self.logger.info(f"[{idx}] 🌐 Processing: {url}")
        
        try:
            await crawler.scraper(
                url=url, 
                browser=browser, 
                file_name=file_name, 
                ss_file_name=ss_file_name
            )
            self._save_log(url, file_name)
            return Result(url=url, file_name=file_name, success=True)
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[{idx}] ❌ Error crawling {url}: {error_msg}")
            self._save_exception(url, file_name, error_msg)
            return Result(url=url, file_name=file_name, success=False, error=error_msg)

    def _save_log(self, url: str, file_name: str):
        """Save successful processing to log file"""
        self._save_to_csv(
            self.log_file,
            {
                "url": url,
                "file_name": file_name,
                "timestamp": datetime.now()
            }
        )

    def _save_exception(self, url: str, file_name: str, error: str):
        """Save exception details to exceptions file"""
        self._save_to_csv(
            self.exceptions_file,
            {
                "url": url,
                "file_name": file_name,
                "exception": error,
                "timestamp": datetime.now()
            }
        )

    def _save_to_csv(self, file_path: str, data: Dict):
        """Generic method to save data to CSV file"""
        pd.DataFrame([data]).to_csv(
            file_path,
            mode='a',
            header=not os.path.exists(file_path),
            encoding='utf-8',
            index=True
        )

    def _url_to_filename(self, url: str, dir: str = None) -> str:
        """Convert URL to a safe filename and join with directory path"""
        safe_name = Utils.url_to_safe_filename(url)
        target_dir = dir if dir is not None else self.output_dir
        return os.path.join(target_dir, safe_name)

    def _get_url_index(self, url: str) -> int:
        """Get index of URL in the DataFrame"""
        try:
            idx = self.urls.index[self.urls['url'].str.contains(url, regex=False)].tolist()
            return idx[0] if idx else -1
        except Exception as e:
            self.logger.error(f"Error getting index for URL {url}: {str(e)}")
            return -1

    async def process_all(self):
        """Process all unprocessed URLs in batches"""
        self.unprocessed_items = self.get_unprocessed_urls()
        if not self.unprocessed_items:
            self.logger.info("📭 No URLs to process")
            return

        self.logger.info(f"📋 Found {len(self.unprocessed_items)} URLs to process")
        browser = None
        
        try:
            browser = await self._create_browser()
            
            for i in range(0, len(self.unprocessed_items), self.batch_size):
                batch = self.unprocessed_items[i:i + self.batch_size]
                batch_range = f"[{i}-{min(i + self.batch_size - 1, len(self.unprocessed_items) - 1)}]"
                
                try:
                    tasks = [self._process_url(url, browser) for url in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    successful = sum(1 for r in results if r.success)
                    self.logger.info(
                        f"Batch {i // self.batch_size + 1}: "
                        f"{successful}/{len(batch)} successful {batch_range}"
                    )
                except Exception as e:
                    self.logger.error(f"💥 Error processing batch {batch_range}: {str(e)}")
                    continue
        finally:
            if browser and self.headless:
                await browser.close()
                self.logger.info("🔒 Browser closed")

        self.show_crawl_stats()

    def show_crawl_stats(self):
        """Show comprehensive crawling statistics"""
        total_urls = len(self.urls)
        failed_urls = len(self._get_exception_urls())
        processed_count = len(self.unprocessed_items)
        success_rate = ((processed_count - failed_urls) / total_urls * 100) if total_urls > 0 else 0
        
        print("\n" + "="*50)
        print("📊 Crawling Statistics:")
        print("\n📈 Summary:")
        print(f"  • Total URLs in input file: {total_urls}")
        print(f"  • Processed URLs: {processed_count}")
        print(f"  • Failed URLs (unique in exceptions): {failed_urls}")
        print(f"  • Success Rate: {success_rate:.1f}%")
        print("="*50 + "\n")

    async def _create_browser(self):
        """Create and configure browser instance"""
        launch_args = [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-site-isolation-trials',
            '--ignore-certificate-errors',
            '--disable-blink-features=AutomationControlled',
            '--disable-infobars',
            '--window-size=1920,1080',
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]

        if self.browser_type == 'chrome':
            launch_args.extend([
                '--start-maximized',
                '--enable-automation',
                '--no-default-browser-check',
            ])
            chrome_path = self._get_chrome_path()
            return await launch(
                headless=self.headless,
                executablePath=chrome_path,
                args=launch_args,
                ignoreHTTPSErrors=True,
                defaultViewport=None if not self.headless else {
                    'width': 1920,
                    'height': 1080,
                    'deviceScaleFactor': 1,
                }
            )
        else:
            return await launch(
                headless=self.headless,
                args=launch_args,
                ignoreHTTPSErrors=True,
                pipe=True,
                defaultViewport={
                    'width': 1920,
                    'height': 1080,
                    'deviceScaleFactor': 1,
                }
            )

    def _get_chrome_path(self) -> str:
        """Get Chrome executable path based on OS"""
        if os.name == 'posix':
            if os.path.exists('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'):
                return '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
            elif os.path.exists('/usr/bin/google-chrome'):
                return '/usr/bin/google-chrome'
            raise Exception("Chrome not found. Please install Google Chrome.")
        return 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'  # Windows path

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process websites from CSV file')
    parser.add_argument('--input', type=str, default='data/processed/websites.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='data/raw/site_links/', help='Directory to save the results as JSON')
    parser.add_argument('--exceptions', type=str, default='data/exceptions/scraper.csv', 
                        help='CSV file to log exceptions')
    parser.add_argument('--screenshots', type=str, 
                        help='Directory to save the screenshots')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of URLs to process in parallel')
    parser.add_argument('--skip-exceptions', type=int, choices=[0, 1], default=0, 
                       help='Skip URLs that previously had exceptions (0=no, 1=yes)')
    parser.add_argument('--overwrite', type=int, choices=[0, 1], default=0,
                       help='Overwrite existing files (1 to overwrite, 0 to skip)')
    parser.add_argument('--show-results', type=int, choices=[0, 1], default=0,
                       help='Show crawling statistics (0=no, 1=yes)')
    parser.add_argument('--quiet', type=int, choices=[0, 1], default=0,
                       help='Suppress all non-error logs (0=no, 1=yes)')
    parser.add_argument('--visit', type=str, help='Visit a single URL from the input file')
    parser.add_argument('--browser', type=str, choices=['chrome', 'chromium'], default='chromium',
                       help='Browser to use (chrome or chromium)')
    parser.add_argument('--headless', type=int, choices=[0, 1], default=1,
                       help='Run browser in headless mode (0=no, 1=yes)')
    args = parser.parse_args()

    scraper = Scraper(
        input_file=args.input,
        output_dir=args.output,
        exceptions_file=args.exceptions,
        batch_size=args.batch_size,
        screenshots_dir=args.screenshots,
        skip_exceptions=bool(args.skip_exceptions),
        overwrite=bool(args.overwrite),
        quiet=bool(args.quiet),
        browser_type=args.browser,
        headless=bool(args.headless),
        visit=args.visit
    )

    await scraper.process_all()

if __name__ == "__main__":
    asyncio.run(main())
