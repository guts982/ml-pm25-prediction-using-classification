
First scrape weather data from www.wunderground.com by adjusting  the date range.

```
uv run main.py
```

Next, scrape the holiday data from www.timeanddate.com for same date range as above

```
uv run holidays.py
```

Finally, add weekend data in the holiday data just scrapped.

```
uv run holidayLocal.py
```
