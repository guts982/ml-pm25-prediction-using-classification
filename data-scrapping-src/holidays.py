import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from datetime import datetime, timedelta
import os
import threading
import re
import csv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NepalHolidayScraper:
    def __init__(self, max_workers=4, csv_filename="nepal_holidays_2020_1_to_2025_4.csv"):
        self.max_workers = max_workers
        self.base_url = "https://www.timeanddate.com/holidays/nepal/{}"
        self.all_holidays = []
        self.lock = threading.Lock()
        self.csv_filename = csv_filename
        self.csv_initialized = False
        
    def init_csv(self):
        """Initialize CSV file with headers"""
        if not self.csv_initialized:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['date', 'name', 'type', 'year'])
            self.csv_initialized = True
            logger.info(f"Initialized CSV file: {self.csv_filename}")
    
    def write_to_csv(self, holidays):
        """Write holidays to CSV file immediately"""
        if not holidays:
            return
            
        with self.lock:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                for holiday in holidays:
                    writer.writerow([
                        holiday['date'],
                        holiday['name'],
                        holiday['type'],
                        holiday['year']
                    ])
            logger.info(f"Written {len(holidays)} holidays to CSV")
        
    def setup_driver(self):
        """Setup Chrome driver with optimized options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User agent to avoid detection
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(10)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            logger.info("Please ensure ChromeDriver is installed and in PATH")
            raise

    def parse_date_with_year(self, date_str, year):
        """Parse various date formats and return standardized date"""
        date_str = date_str.strip()
        
        # Skip day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if date_str in day_names:
            return None
        
        # Common date formats to try
        formats = [
            f"%d %b",      # 15 Jan
            f"%b %d",      # Jan 15
            f"%d %B",      # 15 January
            f"%B %d",      # January 15
        ]
        
        for fmt in formats:
            try:
                # Parse the date with the current year
                parsed_date = datetime.strptime(f"{date_str} {year}", f"{fmt} %Y")
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format worked, return None silently for day names
        return None

    def scrape_year(self, year):
        """Scrape holidays for a specific year"""
        driver = None
        holidays = []
        
        try:
            driver = self.setup_driver()
            url = self.base_url.format(year)
            logger.info(f"Scraping holidays for year {year}: {url}")
            
            driver.get(url)
            time.sleep(3)  # Allow page to load
            
            # Try different selectors for the holiday table
            table = None
            selectors = [
                "table.table",
                "table.zebra", 
                "#holidays-table",
                "table",
                ".fixed_table table"
            ]
            
            for selector in selectors:
                try:
                    table = driver.find_element(By.CSS_SELECTOR, selector)
                    if table:
                        logger.info(f"Found table using selector: {selector}")
                        break
                except NoSuchElementException:
                    continue
            
            if not table:
                logger.warning(f"Could not find holiday table for year {year}")
                return []
            
            # Extract holiday data from table rows
            rows = table.find_elements(By.TAG_NAME, "tr")
            logger.info(f"Found {len(rows)} rows in table for year {year}")
            
            for i, row in enumerate(rows):
                try:
                    # Skip header rows
                    if i == 0:
                        continue
                    
                    # Get all cells in the row
                    ths = row.find_elements(By.TAG_NAME, "th")  # Date is in <th>
                    tds = row.find_elements(By.TAG_NAME, "td")  # Other data in <td>
                    
                    # Based on the HTML structure: <th>15 Jan</th><td>Wednesday</td><td><a>Holiday Name</a></td><td>Type</td>
                    if len(ths) >= 1 and len(tds) >= 2:
                        # Date is in the first <th>
                        date_text = ths[0].text.strip()
                        
                        # Holiday name is in the second <td> (index 1), usually as a link
                        name_cell = tds[1]  # Skip the day name cell (index 0)
                        
                        # Try to get holiday name from link first
                        name_text = ""
                        links = name_cell.find_elements(By.TAG_NAME, "a")
                        if links:
                            name_text = links[0].text.strip()
                        else:
                            name_text = name_cell.text.strip()
                        
                        # Holiday type is in the third <td> (index 2)
                        type_text = "Holiday"
                        if len(tds) >= 3:
                            type_text = tds[2].text.strip()
                        
                        # Skip if essential data is missing
                        if not date_text or not name_text:
                            continue
                        
                        # Parse the date
                        parsed_date = self.parse_date_with_year(date_text, year)
                        if not parsed_date:
                            continue  # Skip invalid dates (like day names)
                        
                        holiday = {
                            'date': parsed_date,
                            'name': name_text,
                            'type': type_text,
                            'year': year
                        }
                        
                        holidays.append(holiday)
                        logger.debug(f"Added holiday: {parsed_date} - {name_text}")
                        
                except Exception as e:
                    logger.debug(f"Error processing row {i} in year {year}: {e}")
                    continue
            
            logger.info(f"Successfully scraped {len(holidays)} holidays for year {year}")
            
            # Write to CSV immediately
            if holidays:
                self.write_to_csv(holidays)
            
        except Exception as e:
            logger.error(f"Error scraping year {year}: {e}")
            
        finally:
            if driver:
                driver.quit()
                
        return holidays
    
    def scrape_date_range(self, start_date, end_date):
        """Scrape holidays for a date range using parallel processing"""
        start_year = start_date.year
        end_year = end_date.year
        
        years_to_scrape = list(range(start_year, end_year + 1))
        logger.info(f"Scraping holidays from {start_year} to {end_year} ({len(years_to_scrape)} years)")
        
        # Initialize CSV file
        self.init_csv()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scraping tasks
            future_to_year = {executor.submit(self.scrape_year, year): year for year in years_to_scrape}
            
            # Collect results as they complete
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    holidays = future.result()
                    with self.lock:
                        self.all_holidays.extend(holidays)
                    logger.info(f"Completed scraping year {year} - {len(holidays)} holidays")
                except Exception as e:
                    logger.error(f"Error scraping year {year}: {e}")
        
        # Filter holidays to exact date range and remove duplicates
        filtered_holidays = []
        seen_holidays = set()
        
        for holiday in self.all_holidays:
            try:
                holiday_date = datetime.strptime(holiday['date'], '%Y-%m-%d').date()
                if start_date <= holiday_date <= end_date:
                    # Create a unique key for deduplication
                    unique_key = (holiday['date'], holiday['name'])
                    if unique_key not in seen_holidays:
                        filtered_holidays.append(holiday)
                        seen_holidays.add(unique_key)
            except ValueError:
                logger.warning(f"Invalid date format: {holiday['date']}")
                continue
        
        self.all_holidays = filtered_holidays
        logger.info(f"Filtered to {len(self.all_holidays)} unique holidays within date range")
        
        return self.all_holidays
    
    def finalize_csv(self):
        """Sort and clean up the final CSV file"""
        try:
            # Read the CSV back
            df = pd.read_csv(self.csv_filename)
            
            if df.empty:
                logger.warning("No data in CSV file")
                return
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['date', 'name'])
            
            # Sort by date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Rewrite the sorted CSV
            df.to_csv(self.csv_filename, index=False, encoding='utf-8')
            logger.info(f"Finalized CSV with {len(df)} unique holidays")
            
            # Display sample data
            print(f"\n✅ Successfully scraped Nepal holidays!")
            print(f"📄 Data saved to: {self.csv_filename}")
            print(f"📊 Total holidays: {len(df)}")
            print(f"\nSample data:")
            print(df.head(10).to_string(index=False))
            
            return df
            
        except Exception as e:
            logger.error(f"Error finalizing CSV: {e}")
            return None

def main():
    """Main function to run the scraper"""
    # Date range
    start_date = datetime(2025, 4, 26).date() #datetime(2020, 1, 1).date()
    end_date = datetime(2025, 5, 30).date() #datetime(2025, 4, 25).date()
    
    # Initialize scraper with parallel processing and CSV filename
    # csv_filename = "nepal_holidays_2020_1_to_2025_4.csv"
    csv_filename = "nepal_holidays_2025_4_to_2025_5.csv"
    scraper = NepalHolidayScraper(max_workers=5, csv_filename=csv_filename)
    
    try:
        print(f"🚀 Starting Nepal holiday scraping from {start_date} to {end_date}")
        print(f"📝 Writing data to: {csv_filename}")
        print("⏰ This may take several minutes...")
        
        start_time = time.time()
        
        # Scrape holidays
        holidays = scraper.scrape_date_range(start_date, end_date)
        
        # Finalize and sort the CSV
        final_df = scraper.finalize_csv()
        
        end_time = time.time()
        
        if final_df is not None and len(final_df) > 0:
            print(f"\n🎉 Scraping completed successfully!")
            print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
            print(f"📈 Average: {len(final_df)/(end_time - start_time):.2f} holidays/second")
        else:
            print("❌ No holidays were scraped. Please check the logs for errors.")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        print("\n⚠️  Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        print(f"❌ Scraping failed: {e}")

if __name__ == "__main__":
    main()