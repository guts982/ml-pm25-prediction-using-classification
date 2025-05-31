import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeekendHolidayAdder:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        
    def load_csv(self):
        #Load the CSV file and validate structure
        try:
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.df)} holidays from {self.csv_file}")
            
            # Validate required columns
            required_columns = ['date', 'name', 'type', 'year']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert date column to datetime
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            print(f"📊 Original data:")
            print(f"   Total holidays: {len(self.df)}")
            print(f"   Date range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return False
    
    def get_date_range(self):
        # Get the date range from the existing data
        if self.df is None or self.df.empty:
            return None, None
        
        start_date = self.df['date'].min().date()
        end_date = self.df['date'].max().date()
        
        return start_date, end_date
    
    def generate_weekends(self, start_date, end_date):
        """Generate all Saturdays and Sundays in the date range"""
        weekends = []
        current_date = start_date
        
        while current_date <= end_date:
            # Check if it's Saturday (5) or Sunday (6)
            if current_date.weekday() == 5:  # Saturday
                weekends.append({
                    'date': current_date,
                    'name': 'Saturday',
                    'type': 'Local Holiday',
                    'year': current_date.year
                })
            elif current_date.weekday() == 6:  # Sunday
                weekends.append({
                    'date': current_date,
                    'name': 'Sunday',
                    'type': 'Local Holiday Optional',
                    'year': current_date.year
                })
            
            current_date += timedelta(days=1)
        
        return weekends
    
    def add_weekends(self, custom_start_date=None, custom_end_date=None):
        """Add weekends to the holiday dataset"""
        if self.df is None:
            logger.error("No data loaded. Please load CSV first.")
            return False
        
        # Get date range
        if custom_start_date and custom_end_date:
            start_date = datetime.strptime(custom_start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(custom_end_date, '%Y-%m-%d').date()
            logger.info(f"Using custom date range: {start_date} to {end_date}")
        else:
            start_date, end_date = self.get_date_range()
            logger.info(f"Using existing data date range: {start_date} to {end_date}")
        
        if not start_date or not end_date:
            logger.error("Could not determine date range")
            return False
        
        # Generate weekend data
        logger.info("Generating weekend holidays...")
        weekend_data = self.generate_weekends(start_date, end_date)
        
        if not weekend_data:
            logger.warning("No weekends generated")
            return False
        
        # Convert to DataFrame
        weekend_df = pd.DataFrame(weekend_data)
        weekend_df['date'] = pd.to_datetime(weekend_df['date'])
        
        # Check for existing weekend dates to avoid duplicates
        existing_dates = set(self.df['date'].dt.date)
        new_weekends = []
        skipped_count = 0
        
        for _, weekend in weekend_df.iterrows():
            weekend_date = weekend['date'].date()
            if weekend_date not in existing_dates:
                new_weekends.append(weekend)
            else:
                skipped_count += 1
        
        if new_weekends:
            # Convert new weekends to DataFrame
            new_weekend_df = pd.DataFrame(new_weekends)
            
            # Combine with existing data
            self.df = pd.concat([self.df, new_weekend_df], ignore_index=True)
            
            # Sort by date
            self.df = self.df.sort_values('date')
            
            logger.info(f"Added {len(new_weekends)} weekend holidays")
            logger.info(f"Skipped {skipped_count} weekends that already exist")
        else:
            logger.info("No new weekends to add")
        
        print(f"\n📈 Updated data:")
        print(f"   Total holidays: {len(self.df)}")
        print(f"   New weekends added: {len(new_weekends) if new_weekends else 0}")
        print(f"   Weekends skipped (already exist): {skipped_count}")
        
        return True
    
    def save_csv(self, output_file=None):
        """Save the updated data to CSV"""
        if self.df is None:
            logger.error("No data to save")
            return False
        
        if not output_file:
            # Create output filename based on input
            base_name = self.csv_file.replace('.csv', '')
            output_file = f"{base_name}_with_weekends.csv"
        
        try:
            # Format date column back to string
            self.df['date'] = self.df['date'].dt.strftime('%Y-%m-%d')
            
            # Save to CSV
            self.df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"Saved updated data to {output_file}")
            print(f"💾 Data saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            return False
    
    def show_sample_data(self, num_rows=10):
        """Display sample data"""
        if self.df is None:
            logger.warning("No data to display")
            return
        
        print(f"\n📋 Sample data ({num_rows} rows):")
        print(self.df.head(num_rows).to_string(index=False))
        
        # Show some weekend examples
        weekend_data = self.df[self.df['name'].isin(['Saturday', 'Sunday'])]
        if not weekend_data.empty:
            print(f"\n🗓️  Weekend examples:")
            print(weekend_data.head(5).to_string(index=False))
    
    def get_statistics(self):
        """Get statistics about the data"""
        if self.df is None:
            return
        
        total_holidays = len(self.df)
        saturdays = len(self.df[self.df['name'] == 'Saturday'])
        sundays = len(self.df[self.df['name'] == 'Sunday'])
        other_holidays = total_holidays - saturdays - sundays
        
        print(f"\n📊 Holiday Statistics:")
        print(f"   Total holidays: {total_holidays}")
        print(f"   Saturdays: {saturdays}")
        print(f"   Sundays: {sundays}")
        print(f"   Other holidays: {other_holidays}")
        
        # Holiday types breakdown
        type_counts = self.df['type'].value_counts()
        print(f"\n🏷️  Holiday Types:")
        for holiday_type, count in type_counts.items():
            print(f"   {holiday_type}: {count}")

def main():
    """Main function to run the weekend adder"""
    # Configuration
    input_csv = "nepal_holidays_2020_1_to_2025_4.csv"  # Change this to your CSV file name
    input_csv = "nepal_holidays_2025_4_to_2025_5.csv"  # Change this to your CSV file name
    
    # You can specify custom date range if needed
    # custom_start = "2020-01-01"
    # custom_end = "2025-04-25"
    custom_start = None  # Use None to auto-detect from data
    custom_end = None
    
    print("🚀 Starting Weekend Holiday Adder")
    print(f"📂 Input file: {input_csv}")
    
    try:
        # Initialize the adder
        adder = WeekendHolidayAdder(input_csv)
        
        # Load the CSV
        if not adder.load_csv():
            print("❌ Failed to load CSV file")
            return
        
        # Add weekends
        if not adder.add_weekends(custom_start, custom_end):
            print("❌ Failed to add weekends")
            return
        
        # Save the updated data
        output_file = adder.save_csv()
        if not output_file:
            print("❌ Failed to save updated data")
            return
        
        # Show statistics and sample data
        adder.get_statistics()
        adder.show_sample_data()
        
        print(f"\n✅ Successfully added weekends to holiday data!")
        print(f"📄 Output file: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{input_csv}'")
        print("   Please make sure the file exists and the path is correct")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        print(f"❌ Script failed: {e}")

if __name__ == "__main__":
    main()