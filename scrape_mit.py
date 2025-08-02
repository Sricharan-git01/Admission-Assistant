import requests
from bs4 import BeautifulSoup
import os

# Create output directory
os.makedirs("data", exist_ok=True)

# MIT pages categorized by topic
PAGES = {
    "Academic_Undergrad": "https://catalog.mit.edu/degree-charts/#undergraduatedegreestext",
    "Academic_Graduate": "https://catalog.mit.edu/degree-charts/#graduatedegreestext",
    "Admissions_Selection": "https://mitadmissions.org/apply/process/selection/",
    "Tuition_Annual_Budget": "https://sfs.mit.edu/undergraduate-students/the-cost-of-attendance/annual-student-budget/",
    "Tuition_Budget_Worksheet": "https://sfs.mit.edu/manage-your-money/budgeting/budgeting-worksheet/",
    "Housing_Graduate": "https://catalog.mit.edu/mit/campus-life/housing/#graduatesinglestudenthousingtext",
    "Housing_Undergrad": "https://catalog.mit.edu/mit/campus-life/housing/#undergraduatehousingtext"
}

# Extract and save top-level page text only
def extract_page(label, url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        filename = f"data/{label}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\n{text}")
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")

# Scrape each page individually (no recursion)
for label, url in PAGES.items():
    print(f"Scraping {label} from {url}")
    extract_page(label, url)

print("All MIT content scraped")
