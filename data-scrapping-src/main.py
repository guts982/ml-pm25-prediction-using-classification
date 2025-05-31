#!/usr/bin/env python3
"""
Kathmandu Weather Data Scraper - PARALLEL VERSION
High-speed parallel scraping with real-time CSV writing and progress tracking.

Requirements:
- selenium
- webdriver-manager
- beautifulsoup4
- pandas

Install with uv:
uv add selenium webdriver-manager beautifulsoup4 pandas

For Google Colab:
!pip install selenium webdriver-manager beautifulsoup4 pandas
!apt-get update
!apt-get install -y chromium-browser
"""

import time
from datetime import datetime, timedelta
import csv
import os
from typing import Dict, List, Optional
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import signal
import sys

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
    import pandas as pd
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not available. Install with: pip install selenium webdriver-manager beautifulsoup4 pandas")
    exit()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelWundergroundScraper:
    def __init__(self, max_workers: int = 4):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.max_workers = max_workers
        self.csv_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.stats = {
            'total_dates': 0,
            'completed_dates': 0,
            'successful_dates': 0,
            'failed_dates': 0,
            'total_records': 0
        }
        self.start_time = time.time()

    def setup_selenium(self) -> Optional[webdriver.Chrome]:
        """Setup Selenium WebDriver with optimized settings"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1280,720')  # Smaller window
            chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--disable-features=VizDisplayCompositor')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-images')  # Speed boost
            chrome_options.add_argument('--disable-javascript')  # May break some sites but speeds up loading
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Performance optimizations
            chrome_options.add_argument('--aggressive-cache-discard')
            chrome_options.add_argument('--memory-pressure-off')
            chrome_options.add_argument('--max_old_space_size=4096')

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.implicitly_wait(2)

            # Execute script to hide webdriver property
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            return driver
        except Exception as e:
            logger.error(f"❌ Failed to initialize Selenium: {e}")
            return None

    def handle_cookie_consent(self, driver):
        """Handle cookie consent with minimal delay"""
        try:
            cookie_selectors = [
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
                "[data-testid='cookie-banner-accept']",
            ]

            for selector in cookie_selectors:
                try:
                    accept_button = WebDriverWait(driver, 1).until(
                        EC.element_to_be_clickable((By.XPATH if selector.startswith("//") else By.CSS_SELECTOR, selector))
                    )
                    accept_button.click()
                    return
                except TimeoutException:
                    continue
                except Exception:
                    continue
        except Exception:
            pass

    def find_hourly_table(self, driver):
        """Find hourly data table with prioritized selectors"""
        table_selectors = [
            "//table[contains(@class, 'observation-table')]",
            "//div[contains(@class, 'observation')]//table",
            "//table[contains(@class, 'history-table')]",
            "//table[.//th[contains(text(), 'Time')] or .//th[contains(text(), 'Temp')]]",
            "//table[tbody/tr/td]"
        ]

        for selector in table_selectors:
            try:
                element = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                if element.is_displayed():
                    return element
            except TimeoutException:
                continue
            except Exception:
                continue
        return None

    def extract_number(self, text: str) -> Optional[float]:
        """Extract number from text"""
        if not text:
            return None
        try:
            cleaned = re.sub(r'[^\d\.-]+', '', text)
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None

    def extract_cell_value(self, cell):
        """Extract value from table cell"""
        if not cell:
            return None
        text = cell.get_text(strip=True)
        return text if text else None

    def scrape_single_date(self, date: datetime) -> Optional[List[Dict]]:
        """Scrape hourly weather data for a single date"""
        driver = self.setup_selenium()
        if not driver:
            return None

        url = f"https://www.wunderground.com/history/daily/np/kathmandu/VNKT/date/{date.strftime('%Y-%m-%d')}"
        hourly_data = []

        try:
            driver.get(url)
            self.handle_cookie_consent(driver)
            
            # Quick page load check
            WebDriverWait(driver, 8).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            hourly_table_element = self.find_hourly_table(driver)
            if not hourly_table_element:
                logger.warning(f"No table found for {date.strftime('%Y-%m-%d')}")
                return None

            # Parse table
            soup = BeautifulSoup(hourly_table_element.get_attribute('outerHTML'), 'html.parser') #type:ignore
            tbody = soup.find('tbody')

            if tbody:
                rows = tbody.find_all('tr')  #type:ignore
                for row in rows:
                    cells = row.find_all('td')  #type:ignore
                    if len(cells) >= 6:
                        try:
                            timestamp_str = self.extract_cell_value(cells[0])
                            if not timestamp_str:
                                continue

                            time_obj = None
                            for fmt in ['%I:%M %p', '%H:%M', '%I:%M%p']:
                                try:
                                    time_obj = datetime.strptime(timestamp_str, fmt)
                                    break
                                except ValueError:
                                    continue

                            if not time_obj:
                                continue

                            datetime_obj = datetime(date.year, date.month, date.day, time_obj.hour, time_obj.minute)

                            
                            hourly_data.append  ({
                                 'timestamp': datetime_obj.isoformat(),
                                 'temperature': self.extract_number(self.extract_cell_value(cells[1]) if len(cells) > 1 else None), #type:ignore
                                 'dew_point': self.extract_number(self.extract_cell_value(cells[2]) if len(cells) > 2 else None), #type:ignore
                                 'humidity': self.extract_number(self.extract_cell_value(cells[3]) if len(cells) > 3 else None), #type:ignore
                                 'wind': self.extract_cell_value(cells[4]) if len(cells) > 4 else None,
                                 'wind_speed': self.extract_number(self.extract_cell_value(cells[5]) if len(cells) > 5 else None),  #type:ignore
                                 'wind_gust': self.extract_number(self.extract_cell_value(cells[6]) if len(cells) > 6 else None),  #type:ignore
                                 'pressure': self.extract_number(self.extract_cell_value(cells[7]) if len(cells) > 7 else None),  #type:ignore
                                 'precipitation': self.extract_number(self.extract_cell_value(cells[8]) if len(cells) > 8 else None),  #type:ignore
                                 'condition': self.extract_cell_value(cells[9]) if len(cells) > 9 else None,
                            })
                        except Exception as e:
                            continue
            # print("HOURLY DATA", hourly_data)
            return hourly_data if hourly_data else None

        except Exception as e:
            logger.error(f"Error scraping {date.strftime('%Y-%m-%d')}: {e}")
            return None
        finally:
            try:
                driver.quit()
            except:
                pass

    def append_to_csv(self, data: List[Dict], filename: str):
        """Thread-safe CSV appending"""
        if not data:
            return

        with self.csv_lock:
            file_exists = os.path.isfile(filename)
            try:
                with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
                    # fieldnames = ['timestamp', 'temperature', 'dew_point', 'humidity', 'wind', 'pressure', 'conditions', 'precipitation']
                    fieldnames = ['timestamp', 'temperature', 'dew_point', 'humidity', 'wind', 'wind_speed', 'wind_gust', 'pressure', 'precipitation', 'condition']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerows(data)
                    
            except Exception as e:
                logger.error(f"Error writing to CSV: {e}")

    def update_progress(self, date: datetime, success: bool, record_count: int = 0):
        """Thread-safe progress tracking"""
        with self.progress_lock:
            self.stats['completed_dates'] += 1
            if success:
                self.stats['successful_dates'] += 1
                self.stats['total_records'] += record_count
            else:
                self.stats['failed_dates'] += 1
            
            # Print progress
            elapsed = time.time() - self.start_time
            progress = (self.stats['completed_dates'] / self.stats['total_dates']) * 100
            rate = self.stats['completed_dates'] / elapsed if elapsed > 0 else 0
            eta = (self.stats['total_dates'] - self.stats['completed_dates']) / rate if rate > 0 else 0
            
            print(f"\r🚀 Progress: {progress:.1f}% | "
                  f"✅ {self.stats['successful_dates']} | "
                  f"❌ {self.stats['failed_dates']} | "
                  f"📊 {self.stats['total_records']} records | "
                  f"⏱️  {rate:.2f} dates/sec | "
                  f"ETA: {eta/60:.1f}min", end='', flush=True)

    def process_date_batch(self, dates: List[datetime], filename: str):
        """Process a batch of dates in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all dates
            future_to_date = {executor.submit(self.scrape_single_date, date): date for date in dates}
            
            # Process results as they complete
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    hourly_data = future.result()
                    if hourly_data:
                        self.append_to_csv(hourly_data, filename)
                        self.update_progress(date, True, len(hourly_data))
                        logger.info(f"✅ {date.strftime('%Y-%m-%d')}: {len(hourly_data)} records")
                    else:
                        self.update_progress(date, False)
                        logger.warning(f"❌ {date.strftime('%Y-%m-%d')}: No data")
                except Exception as e:
                    self.update_progress(date, False)
                    logger.error(f"❌ {date.strftime('%Y-%m-%d')}: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n⚠️ Received interrupt signal. Cleaning up...")
    sys.exit(0)

def main():
    """Main function with parallel processing"""
    print("🚀 PARALLEL KATHMANDU WEATHER SCRAPER")
    print("=" * 60)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Configuration
    start_date = datetime(2025, 4, 26) #datetime(2020, 1, 1)
    end_date = datetime(2025, 5, 30)  # datetime(2025, 4, 25)
    max_workers = 8 # 6  # Adjust based on your system - more workers = faster but more resource intensive
    batch_size =  60 # 50  # Process dates in batches to manage memory
    
    # Generate date list
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Total dates: {len(dates)}")
    print(f"⚡ Workers: {max_workers}")
    print(f"🔄 Batch size: {batch_size}")
    
    # Initialize scraper
    scraper = ParallelWundergroundScraper(max_workers=max_workers)
    scraper.stats['total_dates'] = len(dates)
    
    # Create output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # filename = f'kathmandu_weather_2020_1_to_2025_4_parallel_{timestamp}.csv'
    filename = f'kathmandu_weather_2020_4_to_2025_5_parallel_{timestamp}.csv'
    
    print(f"💾 Output file: {filename}")
    print(f"🚀 Starting parallel scraping...\n")
    
    try:
        # Process dates in batches
        for i in range(0, len(dates), batch_size):
            batch = dates[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(dates) + batch_size - 1) // batch_size
            
            print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} dates)")
            scraper.process_date_batch(batch, filename)
            
            # Small delay between batches to be nice to the server
            if i + batch_size < len(dates):
                time.sleep(2)
        
        # Final summary
        elapsed = time.time() - scraper.start_time
        print(f"\n\n✅ SCRAPING COMPLETED!")
        print("=" * 50)
        print(f"⏱️  Total time: {elapsed/60:.2f} minutes")
        print(f"📊 Total records: {scraper.stats['total_records']}")
        print(f"✅ Successful dates: {scraper.stats['successful_dates']}")
        print(f"❌ Failed dates: {scraper.stats['failed_dates']}")
        print(f"🚀 Average speed: {scraper.stats['completed_dates']/(elapsed/60):.1f} dates/minute")
        print(f"💾 Data saved to: {filename}")
        
        # Show sample of scraped data
        if os.path.isfile(filename):
            try:
                df = pd.read_csv(filename)
                print(f"\n📋 Sample data (first 5 rows):")
                print(df.head())
                print(f"\n📈 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            except Exception as e:
                print(f"Error reading final CSV: {e}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print(f"\n🎉 All done! Check {filename} for your data.")

if __name__ == "__main__":
    main()

# For Google Colab users
def colab_setup():
    """Setup function for Google Colab"""
    import subprocess
    import sys

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                             "selenium", "webdriver-manager", "beautifulsoup4", "pandas"])
        print("✅ Packages installed successfully!")
        subprocess.check_call(["apt-get", "update"])
        subprocess.check_call(["apt-get", "install", "-y", "chromium-browser"])
        print("✅ Chrome installed for Colab!")
        main()
    except Exception as e:
        print(f"❌ Setup error: {e}")

# Uncomment the line below if running in Google Colab
# colab_setup()