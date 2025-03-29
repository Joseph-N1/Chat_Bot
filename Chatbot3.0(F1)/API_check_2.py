import requests

ergast_url = "http://ergast.com/api/f1/2024/1/results.json"  # Round 1 (Bahrain GP)
response = requests.get(ergast_url)

if response.status_code == 200:
    data = response.json()
    print(data)  # Check the structure of the response
else:
    print(f"Error: {response.status_code}")

